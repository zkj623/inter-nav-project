"""Hardware-shaped messages without ROS/SDK installation or a robot connection."""
from collections import deque
import json
import struct
import threading
import time
from types import SimpleNamespace as Message

import numpy as np
import pytest

from deploy.app.config import check_task_inputs, check_motion_configuration, load
from deploy.app.tasks import Tasks
from deploy.robot.diagnostics import describe
from deploy.robot.localization import PoseContinuity, PoseHistory
from deploy.robot.sensors import RosSensors, decode_cloud


def cloud(stamp=1_010_000_000, bigendian=False):
    # Two organized rows, padding in both points and rows; field order != XYZ.
    order = '>' if bigendian else '<'
    raw = b''.join(struct.pack(order+'ffff', 99, x, .2, .7) + b'PAD!'
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
                   twist=Message(twist=Message(linear=Message(x=.1, y=0.), angular=Message(z=0.))))


def sensor(calibrated=True):
    source = RosSensors.__new__(RosSensors)
    source.config = load()['robot']
    if calibrated:
        source.config.update(ground_z=-.3, lidar_origin_offset_base_m=[.1, 0., .1])
    source.lock = threading.Lock()
    source.pose_history = PoseHistory(.05)
    source.continuity = PoseContinuity(.5, .7)
    source.last_stamps = {}
    source.pose = source.lidar = source.camera = source.last_error = None
    source.sequence = 0
    source._stamp = lambda message: (message.header.stamp, time.monotonic())
    # No TF buffer: same-frame data must work without TF publication.
    return source


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


def test_observe_maintenance_uses_wall_interval_without_advancing_task_timeout():
    source = sensor()
    source._pose(odometry(1_000_000_000, 1.))
    source._pose(odometry(1_020_000_000, 1.02))
    source._lidar(cloud())
    snapshot = source.latest()
    engine = Tasks(load()['navigation'])
    calls = []
    engine.component.mapping.update = lambda step, obs, **kwargs: calls.append((step, kwargs['maintenance']))
    for now in (10., 10.1, 10.3, 10.9, 11.2, 11.3):
        engine.update(snapshot, now)
    assert [step for step, _ in calls] == list(range(6))
    assert [due for _, due in calls] == [True, False, False, False, True, False]
    assert engine.elapsed == 0 and engine.last_step == -1


def test_observation_freshness_error_identifies_stream_and_direction():
    from deploy.robot.observation import Snapshot
    snapshot = Snapshot({}, 10., 9.5, 10., None, (0, 0, 0), 1)
    with pytest.raises(ValueError, match='odom observation stale: age=500.0 ms'):
        snapshot.require_fresh(10., .4)
    snapshot.pose_at, snapshot.lidar_at = 10., 10.06
    with pytest.raises(ValueError, match='lidar observation future: ahead=60.0 ms'):
        snapshot.require_fresh(10., .4)


def test_deskewed_padding_is_removed_before_tf_without_removing_real_origin():
    def message(points):
        a = np.asarray(points, dtype='<f4').reshape(-1, 3)
        return Message(width=len(a), height=1, point_step=12, row_step=len(a)*12,
                       data=a.tobytes(), is_bigendian=False,
                       fields=[Message(name=name, offset=i*4, count=1, datatype=7)
                               for i, name in enumerate(('x', 'y', 'z'))],
                       header=Message(frame_id='odom', stamp=1_010_000_000))

    returns = np.array([[1., 2., .3], [0., 0., 0.], [2., 1., .4]], dtype=np.float32)
    padded = message(np.concatenate([np.zeros((10000, 3)), returns]))
    np.testing.assert_array_equal(decode_cloud(padded, strip_utlidar_padding=True), returns)
    assert len(decode_cloud(padded)) == 10003  # Generic decoding remains lossless.
    np.testing.assert_array_equal(decode_cloud(message(returns), strip_utlidar_padding=True), returns)
    with pytest.raises(ValueError, match='no finite returns'):
        decode_cloud(message(np.zeros((10000, 3))), strip_utlidar_padding=True)

    source = sensor()
    source._lidar(padded)
    np.testing.assert_array_equal(source.lidar[1]['pointcloud'], returns)
    source.config['lidar_topic'] = '/utlidar/cloud_base'
    padded.header.stamp += 1
    source._lidar(padded)
    assert len(source.lidar[1]['pointcloud']) == 10003


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
    source._pose(odometry(1_000_000_000, 1.))
    source._pose(odometry(1_020_000_000, 1.02))
    source._lidar(cloud())
    packet = source.latest()
    assert packet.planning_frame == 'odom' and packet.geometry_calibrated
    np.testing.assert_allclose(packet.observation['position'], [1.02, 0, .3])
    np.testing.assert_allclose(packet.observation['sensors']['lidar']['position'], [1.11, 0, .4])
    # World-frame endpoints must NOT be translated by the robot position again.
    np.testing.assert_allclose(packet.observation['sensors']['lidar']['pointcloud'], [[2, .2, 1], [3, .2, 1]])
    with pytest.raises(ValueError, match='Non-increasing'):
        source._pose(odometry(990_000_000, -4.))
    assert source.continuity.fault is None
    with pytest.raises(ValueError, match='Non-increasing'):
        source._lidar(cloud())


def test_cloud_waits_for_time_aligned_odometry():
    source = sensor()
    source._pose(odometry(1_000_000_000, 1.))
    source._lidar(cloud(1_200_000_000))
    assert source.latest() is None
    source._pose(odometry(1_200_000_000, 1.02))
    assert source.latest() is not None


def test_uncalibrated_data_recordable_but_cannot_build_or_drive_map():
    source = sensor(calibrated=False)
    source._pose(odometry(1_000_000_000, 1.))
    source._lidar(cloud())
    packet = source.latest()
    assert not packet.geometry_calibrated
    engine = Tasks(load()['navigation'])
    assert engine.update(packet, packet.captured_at) is None
    assert not engine.map_ready
    with pytest.raises(ValueError, match='ground_z'):
        check_task_inputs(source.config, packet)
    engine.start('navigate', (2, 0, .3), packet.captured_at)
    with pytest.raises(ValueError, match='calibrated'):
        engine.update(packet, packet.captured_at)


def test_video_only_profile_allows_geometry_but_rejects_exploration():
    source = sensor()
    source._pose(odometry(1_000_000_000, 1.))
    source._lidar(cloud())
    packet = source.latest()
    check_task_inputs(source.config, packet, exploring=False)
    with pytest.raises(ValueError, match='RGB-D'):
        check_task_inputs(source.config, packet, exploring=True)


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
    from deploy.perception import client
    source = sensor(calibrated=False)
    source._pose(odometry(1_000_000_000, 1.))
    source._lidar(cloud())
    source.close = lambda: None
    keys = deque([None, ('q', None)])
    monkeypatch.setattr(sensors, 'RosSensors', lambda *args: source)
    monkeypatch.setattr(keyboard, 'Keyboard', lambda: Message(
        editing=None, poll=lambda: keys.popleft(), close=lambda: None))
    def forbidden(*args, **kwargs):
        pytest.fail('Observe-only video-unavailable profile must not start models or SDK')
    monkeypatch.setattr(client, 'AsyncPerception', forbidden)
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


def test_receipt_does_not_bypass_communication_acceptance():
    config = load()
    robot = config['robot']
    robot.update(calibration_verified=True, deskew_and_self_filter_verified=True,
                 remote_arbitration_verified=True, ground_z=-.3,
                 lidar_origin_offset_base_m=[.1, 0, .1], takeover_button_mask=1)
    with pytest.raises(ValueError, match='communication_verified'):
        check_motion_configuration(config)
    robot['communication_verified'] = True
    check_motion_configuration(config)


def test_time_aligned_origin_rotates_with_body():
    history = PoseHistory(.05)
    # A 20-degree turn over 40 ms; midpoint must rotate the mounting offset by 10 degrees.
    history.add(1_000_000_000, [1, 2, 0], [1, 0, 0, 0])
    history.add(1_040_000_000, [1, 2, 0], [np.cos(np.deg2rad(10)), 0, 0, np.sin(np.deg2rad(10))])
    from deploy.robot.localization import transform
    p, q = history.at(1_020_000_000)
    np.testing.assert_allclose(transform([.1, 0, .2], p, q),
                               [1+.1*np.cos(np.deg2rad(10)), 2+.1*np.sin(np.deg2rad(10)), .2])
