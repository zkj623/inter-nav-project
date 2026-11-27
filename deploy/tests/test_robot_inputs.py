"""Hardware-shaped messages without ROS/SDK installation or a robot connection."""
from collections import deque
import json
import struct
import threading
import time
from types import SimpleNamespace as Message

import numpy as np
import pytest

from deploy.app.config import ROOT, check_task_inputs, load
from deploy.app.tasks import Tasks
from deploy.robot.diagnostics import describe
from deploy.robot.localization import PoseContinuity, registered_floor_height
from deploy.robot.sensors import RosSensors, decode_cloud


def cloud(stamp=1_010_000_000, bigendian=False, z=.7):
    # Two organized rows, padding in both points and rows; field order != XYZ.
    order = '>' if bigendian else '<'
    raw = b''.join(struct.pack(order+'ffff', 99, x, .2, z) + b'PAD!'
                   for x in (2., 3.))
    return Message(width=1, height=2, point_step=16, row_step=20,
                   data=raw, is_bigendian=bigendian,
                   fields=[Message(name=name, offset=offset, count=1, datatype=7)
                           for name, offset in [('z', 12), ('x', 4), ('y', 8)]],
                   header=Message(frame_id='odom', stamp=stamp))


def odometry(stamp, x, frame='odom'):
    return Message(header=Message(frame_id=frame, stamp=stamp), child_frame_id='base_link',
                   pose=Message(pose=Message(position=Message(x=x, y=0., z=0.),
                                            orientation=Message(w=1., x=0., y=0., z=0.))),
                   twist=Message(twist=Message(linear=Message(x=.1, y=0., z=.02), angular=Message(x=.03, y=.04, z=0.))))


def sensor(calibrated=True):
    source = RosSensors.__new__(RosSensors)
    source.config = dict(load()['robot'], map_frame='odom', base_frame='base_link', lio_imu_frame='base_link')
    source.matched_scan = None
    source.mount = None
    source.registered_poses = deque(maxlen=8)
    source.registered_scans = deque(maxlen=8)
    source.previous_scan_end_at = None
    source.imu_samples = deque(maxlen=256)
    if calibrated:
        source.config.update(ground_z=-.3, lidar_origin_offset_base_m=[.1, 0., .1])
    else:
        source.config.update(ground_z=None, lidar_origin_offset_base_m=None)
    source.lock = threading.Lock()
    source.frame_ready = threading.Condition(source.lock)
    source.imu_tolerance_ns = 50_000_000
    source.continuity = PoseContinuity(.5, .7)
    source.last_stamps = {}
    source.last_error = None
    source.sequence = 0
    source._stamp = lambda message: (message.header.stamp, time.monotonic())
    # No TF buffer: same-frame data must work without TF publication.
    return source


def matched_packet(source, pose=None, z=.7):
    pose = pose or odometry(1_010_000_000, 1.)
    stamp = pose.header.stamp
    source._lidar(cloud(stamp-100_000_000, z=z))
    source._pose(pose)
    source._imu(Message(header=Message(stamp=stamp, frame_id=source.config['imu_frame']),
                        angular_velocity=pose.twist.twist.angular, linear_acceleration=Message(x=0., y=0., z=1.)))
    source._lidar(cloud(stamp, z=z))
    return source.latest()


@pytest.mark.parametrize('bigendian', [False, True])
def test_cloud_layout_and_endianness(bigendian):
    np.testing.assert_allclose(decode_cloud(cloud(bigendian=bigendian)), [[2, .2, .7], [3, .2, .7]])


def test_sensor_error_recovers_only_when_its_own_stream_recovers():
    source = sensor()
    def receive(ok):
        if not ok:
            raise ValueError('bad timestamp')
    odom = source._guard(receive, 'odom')
    lidar = source._guard(receive, 'lidar')
    odom(False)
    lidar(True)
    assert 'odom' in source.last_error
    odom(True)
    assert source.last_error is None


def test_observe_deduplicates_scans_without_advancing_task_timeout():
    source = sensor()
    snapshot = matched_packet(source, z=-.05)  # 25 cm above the fixture floor
    tasks = Tasks(load()['navigation'])
    for now in (10., 10.1, 10.3, 10.9, 11.2, 11.3):
        tasks.update(snapshot, now)
    assert tasks.engine.layers.scan_count == 1
    assert tasks.elapsed == 0
    np.testing.assert_allclose(snapshot.observation['body_linear_velocity'], [.1, 0, .02])
    np.testing.assert_allclose(snapshot.observation['body_angular_velocity'], [.03, .04, 0])


def test_ground_height_cap_preserves_dense_pixel_alignment_and_the_source_recording():
    from deploy.robot.observation import Snapshot
    from deploy.robot.world_boundary import WorldBoundary
    xyz = np.array([[1., .95, .279], [1., 0., .28], [1., 0., .28001]], dtype=np.float32)
    depths = np.full((1, 3), 1., dtype=np.float32)
    sensors = {'lidar': {'pointcloud': xyz}, 'camera': {'pointcloud': xyz, 'depth': depths}}
    raw = Snapshot({'sensors': sensors}, 4., 3.9, 3.9, 3.9, (), 12)
    cut = raw.below_ceiling(.28)
    filtered = cut.observation['sensors']
    assert cut.sequence == 12 and cut.lidar_at == 3.9
    assert len(filtered['lidar']['pointcloud']) == 2
    assert filtered['lidar']['pointcloud'][-1, 2] == .28
    assert filtered['camera']['pointcloud'].shape == (3, 3)
    assert np.isnan(filtered['camera']['pointcloud'][2]).all()
    np.testing.assert_equal(filtered['camera']['depth'], [[1., 1., np.nan]])
    cropped = raw.navigation_view(.28, WorldBoundary((-1.2, 5.), (-2.6, .9))).observation['sensors']
    np.testing.assert_array_equal(cropped['lidar']['pointcloud'], [[1., 0., .28]])
    assert np.isnan(cropped['camera']['pointcloud'][[0, 2]]).all()
    np.testing.assert_equal(cropped['camera']['depth'], [[np.nan, 1., np.nan]])
    assert np.isfinite(xyz).all() and np.isfinite(depths).all() and xyz[-1, 2] > .28


def test_observation_freshness_error_identifies_stream_and_direction():
    from deploy.robot.observation import Snapshot
    snapshot = Snapshot({}, 10., 9.5, 10., None, (0, 0, 0), 1)
    with pytest.raises(ValueError, match='odom observation stale: age=500.0 ms'):
        snapshot.require_fresh(10., .4)
    snapshot.pose_at, snapshot.lidar_at = 10., 10.06
    with pytest.raises(ValueError, match='lidar observation future: ahead=60.0 ms'):
        snapshot.require_fresh(10., .4)


@pytest.mark.parametrize('corruption', ['truncated', 'stride', 'missing_xyz', 'field_offset', 'all_nan'])
def test_malformed_cloud_is_not_free_space(corruption):
    message = cloud()
    if corruption == 'truncated':
        message.data = message.data[:-1]
    elif corruption == 'stride':
        message.row_step = 12
    elif corruption == 'missing_xyz':
        message.fields.pop()
    elif corruption == 'field_offset':
        message.fields[0].offset = 15
    else:
        message.data = struct.pack('<ffff', 0, float('nan'), 0, 0) + b'PAD!'
        message.height = 1
    with pytest.raises(ValueError):
        decode_cloud(message)


def test_odom_pose_needs_no_tf_and_cloud_origin_moves_with_robot():
    source = sensor()
    packet = matched_packet(source, odometry(1_010_000_000, 1.01))
    assert packet.planning_frame == 'odom' and packet.geometry_calibrated
    np.testing.assert_allclose(packet.observation['position'], [1.01, 0, .3])
    np.testing.assert_allclose(packet.observation['sensors']['lidar']['position'], [1.11, 0, .4])
    # World-frame endpoints must NOT be translated by the robot position again.
    np.testing.assert_allclose(packet.observation['sensors']['lidar']['pointcloud'], [[2, .2, 1], [3, .2, 1]])
    with pytest.raises(ValueError, match='Non-increasing'):
        source._pose(odometry(990_000_000, -4.))
    assert source.continuity.fault is None
    with pytest.raises(ValueError, match='Non-increasing'):
        source._lidar(cloud())


def test_mid360_pairs_registered_points_world_velocity_and_raw_imu():
    source = sensor()
    source.config = dict(load()['robot'], ground_z=0., base_frame='bot1/aft_mapped',
                         lidar_topic='/bot1/cloud_registered_sparse',
                         lidar_origin_offset_base_m=[-.011, -.02329, .04412])
    stamp = 1_000_000_000
    pose = odometry(stamp, 1., frame=source.config['map_frame'])
    pose.child_frame_id = source.config['base_frame']
    pose.pose.pose.orientation.w = pose.pose.pose.orientation.z = np.sqrt(.5)
    scan = cloud(stamp)
    with pytest.raises(ValueError, match='registered'):
        source._lidar(scan)
    scan.header.frame_id = source.config['map_frame']
    previous = cloud(stamp-100_000_000)
    previous.header.frame_id = scan.header.frame_id
    source._lidar(previous)
    source._lidar(scan)
    source._pose(pose)
    assert source.latest() is None
    imu = Message(header=Message(stamp=stamp-100_000_000, frame_id=source.config['imu_frame']),
                  angular_velocity=Message(x=0., y=.15, z=-.2), linear_acceleration=Message(x=0., y=0., z=1.))
    source._imu(imu)
    assert source.latest() is None
    imu.header.stamp = stamp+2_000_000
    source._imu(imu)
    packet = source.latest()
    np.testing.assert_allclose(packet.velocity, [0., -.1, -.2], atol=1e-9)
    np.testing.assert_allclose(packet.observation['body_linear_velocity'], [0., -.1, .02], atol=1e-9)
    np.testing.assert_allclose(packet.observation['sensors']['imu']['linear_acceleration_m_s2'], [0, 0, 9.80665])
    lidar = packet.observation['sensors']['lidar']
    np.testing.assert_allclose(lidar['pointcloud'], [[2., .2, .7], [3., .2, .7]])
    np.testing.assert_allclose(lidar['position'], [1.02329, -.011, .04412])
    # The next pose may arrive before its cloud. Keep the old *source* age.
    pose.header.stamp += 100_000_000
    source._pose(pose)
    assert source.wait_frame(stamp, 0) == stamp
    assert source.latest().pose_at == packet.pose_at
    scan.header.stamp = pose.header.stamp
    source._lidar(scan)
    assert source.wait_frame(stamp, 0) == pose.header.stamp
    assert source.latest() is None  # previous gyro is too old for the new scan


def test_mid360_floor_estimate_uses_world_height_and_rejects_a_single_wall():
    coords = np.arange(-1.9, 2., .09)
    x, y = np.meshgrid(coords, coords)
    floor = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 1.25)])
    assert registered_floor_height(floor, [0., 0., 1.7]) == pytest.approx(1.25)
    wall = np.column_stack([np.ones(x.size), y.ravel(), x.ravel()+1.25])
    assert registered_floor_height(wall, [0., 0., 1.7]) is None


def test_default_startup_measures_the_floor_from_registered_xyz(monkeypatch):
    from deploy.tests.test_timebase import fake_ros, message

    cfg = load()['robot']
    x, y = np.meshgrid(np.arange(-2., 2., .15), np.arange(-2., 2., .15))
    xyz = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.31))).astype('<f4')
    payload = dict(width=len(xyz), height=1, point_step=12, row_step=xyz.nbytes, data=xyz.tobytes(),
                   is_bigendian=False, fields=[Message(name=axis, datatype=7, count=1, offset=4*i)
                                              for i, axis in enumerate('xyz')])
    def sample(topic, stamp):
        header = message(stamp).header
        if topic == cfg['imu_topic']:
            header.frame_id = cfg['imu_frame']
            return Message(header=header)
        if topic == cfg['odom_topic']:
            pose = odometry(header.stamp, 0., frame=cfg['map_frame'])
            pose.child_frame_id = cfg['lio_imu_frame']
            return pose
        assert topic == '/bot1/cloud_registered_sparse'
        header.frame_id = cfg['map_frame']
        return Message(header=header, **payload)

    fake_ros(monkeypatch, sample=sample)
    source = RosSensors(cfg)
    try:
        assert source.config['ground_z'] == pytest.approx(-.31)
    finally:
        source.close()


@pytest.mark.parametrize('interaction_path', [None, ROOT/'configs/box_30x80x30.yaml'], ids=['navigation', 'first_push'])
def test_sparse_cloud_uses_world_xyz_and_matches_body_pose_when_dds_streams_overtake(monkeypatch, interaction_path):
    from deploy.robot.localization import multiply, rotation
    from deploy.robot.timebase import SourceTimeBridge
    from deploy.tests.test_timebase import EPOCH_NS, MONO_NS, fake_ros, message

    state = fake_ros(monkeypatch)
    cfg = dict(load(interaction_path=interaction_path)['robot'], ground_z=0.)
    assert cfg['lidar_topic'] == '/bot1/cloud_registered_sparse'
    source = RosSensors(cfg, timebase=SourceTimeBridge(EPOCH_NS, MONO_NS))
    period = 100_000_000
    body_q = [np.cos(.3), 0., 0., np.sin(.3)]
    world_from_body = rotation(body_q)
    pitch = np.deg2rad(cfg['mid360_mount']['pitch_deg'])
    body_from_imu = [np.cos(pitch/2), 0., np.sin(pitch/2), 0.]
    imu_q = multiply(body_q, body_from_imu)
    imu_offset = np.array(cfg['mid360_mount']['imu_in_base_m'])
    def scan(index):
        msg = cloud()
        msg.header = message(EPOCH_NS + index*period).header
        msg.header.frame_id = cfg['map_frame']
        return msg
    def pose_and_imu(index):
        body_p = [1.+index*.05, .2, .35]
        imu_p = body_p + world_from_body @ imu_offset
        msg = odometry(None, 0.)
        msg.header = scan(index).header
        msg.child_frame_id = cfg['lio_imu_frame']
        msg.pose.pose.position = Message(x=imu_p[0], y=imu_p[1], z=imu_p[2])
        msg.pose.pose.orientation = Message(w=imu_q[0], x=imu_q[1], y=imu_q[2], z=imu_q[3])
        source._pose(msg)
        gyro = rotation(body_from_imu).T @ [0., 0., .5]
        imu = Message(header=message(EPOCH_NS + index*period).header,
                      angular_velocity=Message(x=gyro[0], y=gyro[1], z=gyro[2]),
                      linear_acceleration=Message(x=0., y=0., z=1.))
        imu.header.frame_id = cfg['imu_frame']
        source._imu(imu)

    try:
        # Pose 2 arrives before cloud 1. Later, cloud 3 arrives before pose 3.
        state.mono_ns += 2*period
        source._lidar(scan(0))
        pose_and_imu(1)
        pose_and_imu(2)
        assert source.latest() is None
        source._lidar(scan(1))
        packet = source.latest()
        lidar = packet.observation['sensors']['lidar']
        np.testing.assert_array_equal(lidar['pointcloud'], decode_cloud(scan(1)))
        assert lidar['physics_step'] == lidar['source_stamp_ns'] == EPOCH_NS+period
        assert packet.pose_at - packet.lidar_at == pytest.approx(.1)
        np.testing.assert_allclose(packet.observation['position'], [1.05, .2, .35])
        np.testing.assert_allclose(packet.observation['orientation'], body_q)
        np.testing.assert_allclose(lidar['position'], [1.05, .2, .35] + world_from_body @ cfg['lidar_origin_offset_base_m'])
        np.testing.assert_allclose(packet.observation['body_angular_velocity'], [0., 0., .5], atol=1e-12)
        np.testing.assert_allclose(packet.observation['body_linear_velocity'],
                                   world_from_body.T @ [.1, 0., .02] - np.cross([0., 0., .5], imu_offset))
        source._lidar(scan(2))
        second = source.latest()
        state.mono_ns += period
        source._lidar(scan(3))
        assert source.latest().lidar_at == second.lidar_at
        pose_and_imu(3)
        assert source.latest().observation['sensors']['lidar']['physics_step'] == EPOCH_NS+3*period

        # A gap ages the first subsequent sweep conservatively. The next recovers.
        state.mono_ns += 5*period
        pose_and_imu(8)
        source._lidar(scan(8))
        with pytest.raises(ValueError, match='lidar observation stale'):
            source.latest().require_fresh(state.mono_ns*1e-9, cfg['max_data_age_s'])
        state.mono_ns += period
        pose_and_imu(9)
        source._lidar(scan(9))
        source.latest().require_fresh(state.mono_ns*1e-9, cfg['max_data_age_s'])
    finally:
        source.close()


def test_uncalibrated_data_recordable_but_cannot_build_or_drive_map():
    source = sensor(calibrated=False)
    packet = matched_packet(source)
    assert not packet.geometry_calibrated
    engine = Tasks(load()['navigation'])
    assert engine.update(packet, packet.captured_at) is None
    assert not engine.map_ready
    with pytest.raises(ValueError, match='ground_z'):
        check_task_inputs(source.config, packet)
    engine.start('navigate', (2, 0, .3), packet.captured_at)
    with pytest.raises(ValueError, match='calibrated'):
        engine.update(packet, packet.captured_at)


def test_video_only_profile_supports_navigation_but_has_no_interaction():
    source = sensor()
    packet = matched_packet(source)
    check_task_inputs(source.config, packet)
    assert load()['navigation']['interaction_enabled'] is False


def test_diagnostic_metadata_does_not_dump_binary_payload():
    message = cloud()
    message.header.stamp = Message(sec=1, nanosec=10_000_000)
    report = describe(message, 1.02)
    assert report['payload_bytes'] == report['expected_bytes'] == 40
    assert report['age_s'] == pytest.approx(.01)
    assert 'PAD!' not in json.dumps(report)


def test_live_observe_without_rgbd_does_not_start_models_or_motion(monkeypatch, tmp_path):
    from deploy.app import keyboard, runtime
    from deploy.robot import sensors, motion
    source = sensor(calibrated=False)
    matched_packet(source)
    source.close = lambda: None
    keys = deque([None, ('q', None)])
    monkeypatch.setattr(sensors, 'RosSensors', lambda *args: source)
    monkeypatch.setattr(keyboard, 'Keyboard', lambda: Message(
        editing=None, poll=lambda: keys.popleft(), close=lambda: None))
    def forbidden(*args, **kwargs):
        pytest.fail('Observe-only video-unavailable profile must not start models or SDK')
    monkeypatch.setattr(motion, 'MotionProcess', forbidden)
    from deploy.app import session as session_module
    original = session_module.Session
    sessions = []
    def make_session(*args, **kwargs):
        obj = original(*args, **kwargs)
        close = obj.close
        obj.close = lambda summary: close(summary, grace=0)
        sessions.append(obj)
        return obj
    monkeypatch.setattr(session_module, 'Session', make_session)
    args = Message(allow_motion=False, demo=False, session_port=0, camera='none', rgb_topic=None)
    assert runtime.run_live(args, load()) == 0
    summary = sessions[0].summary
    assert summary['map_created'] is False
    assert summary['last_autonomous_stop_confirmed'] is None
    assert sessions[0].batch(0, 0)['frames']
    assert not list(tmp_path.iterdir())  # run publishes; only record owns experiment files
