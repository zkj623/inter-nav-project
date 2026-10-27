"""Fixed per-run coordinates: geometry, ownership, and live/replay boundaries."""
from collections import deque
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import load, REPO
from deploy.robot.observation import ExperimentFrame, Snapshot, normalize
from deploy.robot.localization import rotation, multiply
from deploy.recording.recorder import Recorder
from deploy.recording.replay import snapshots
from deploy.app.session import encode_packet, decode_packet


def observation():
    yaw = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)])
    # Source ground is z=2; robot base must remain .3 m above it.
    raw = {'position': np.array([30., 40., 2.3]), 'orientation': yaw,
           'sensors': {
               'lidar': {'position': np.array([30., 40.1, 2.4]),
                         'pointcloud': np.array([[30., 42., 2.], [29., 42., 2.7]]),
                         'physics_step': 1, 'max_range': 8.},
               'camera': {'position': np.array([30., 40.2, 2.5]), 'orientation': yaw,
                          'rgba': np.zeros((1, 2, 4), dtype=np.uint8),
                          'depth': np.array([[2., np.nan]]),
                          'pointcloud': np.array([[29., 42., 2.7], [np.nan]*3]),
                          'horizontal_fov_degrees': 60., 'capture_step': 1}}}
    return Snapshot(normalize(raw, 2.), 10., 10., 10., 10., (.1, .2, .3), 1,
                    planning_frame='odom')


def test_all_geometry_shares_transform_with_ground_height_and_body_velocity_preserved():
    source = observation()
    original = source.observation['sensors']['lidar']['pointcloud'].copy()
    frame = ExperimentFrame(2.)
    result = frame.apply(source, 10., .4)
    assert result.planning_frame == 'experiment'
    np.testing.assert_allclose(result.observation['position'], [0, 0, .3], atol=1e-7)
    np.testing.assert_allclose(result.observation['orientation'], [1, 0, 0, 0], atol=1e-7)
    lidar = result.observation['sensors']['lidar']
    np.testing.assert_allclose(lidar['pointcloud'], [[2, 0, 0], [2, 1, .7]], atol=1e-7)
    np.testing.assert_allclose(lidar['position'], [.1, 0, .4], atol=1e-7)
    camera = result.observation['sensors']['camera']
    np.testing.assert_allclose(camera['position'], [.2, 0, .5], atol=1e-7)
    np.testing.assert_allclose(camera['pointcloud'][0], [2, 1, .7], atol=1e-7)
    assert np.isnan(camera['pointcloud'][1]).all()
    np.testing.assert_allclose(camera['orientation'], [1, 0, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(camera['depth'], source.observation['sensors']['camera']['depth'])
    assert result.velocity == source.velocity
    np.testing.assert_array_equal(source.observation['sensors']['lidar']['pointcloud'], original)
    assert source.coordinate_transform is None
    # Metadata defines the inverse even though input normalization already removed ground_z.
    meta = result.coordinate_transform
    q = [np.cos(meta['yaw_source_rad']/2), 0, 0, np.sin(meta['yaw_source_rad']/2)]
    recovered = lidar['pointcloud'] @ rotation(q).T + meta['origin_source_m']
    np.testing.assert_allclose(recovered, [[30, 42, 2], [29, 42, 2.7]], atol=1e-7)


def test_reference_is_fixed_after_robot_translates_and_turns():
    source = observation()
    frame = ExperimentFrame(2.)
    first = frame.apply(source, 10., .4)
    moved = replace(source, observation={**source.observation,
                    'position': np.array([29., 42., .3]), 'orientation': np.array([0., 0, 0, 1.])})
    later = frame.apply(moved, 10., .4)
    np.testing.assert_allclose(later.observation['position'], [2, 1, .3], atol=1e-7)
    np.testing.assert_allclose(later.observation['sensors']['lidar']['pointcloud'],
                               first.observation['sensors']['lidar']['pointcloud'])
    assert later.coordinate_transform == first.coordinate_transform
    assert later.coordinate_transform is not first.coordinate_transform
    np.testing.assert_allclose(rotation(later.observation['orientation'])[:, 0], [0, 1, 0], atol=1e-7)
    # A fresh run deliberately chooses a fresh reference.
    restarted = ExperimentFrame(2.).apply(moved, 10., .4)
    np.testing.assert_allclose(restarted.observation['position'][:2], [0, 0], atol=1e-7)


def test_yaw_alignment_keeps_roll_pitch_for_safety_checks():
    source = observation()
    roll = [.98, .2, 0, 0]
    tilted = replace(source, observation={**source.observation,
                      'orientation': multiply(source.observation['orientation'], roll)})
    result = ExperimentFrame(2.).apply(tilted, 10., .4)
    np.testing.assert_allclose(rotation(result.observation['orientation']), rotation(roll), atol=1e-7)


def test_invalid_initial_data_does_not_set_origin_and_later_frames_cannot_reset_it():
    frame = ExperimentFrame(2.)
    source = observation()
    uncalibrated = replace(source, geometry_calibrated=False)
    assert frame.apply(uncalibrated, 10., .4) is uncalibrated
    with pytest.raises(ValueError, match='stale'):
        frame.apply(source, 11., .4)
    assert frame.metadata is None
    ready = frame.apply(source, 10., .4)
    with pytest.raises(ValueError, match='twice'):
        frame.apply(ready, 10., .4)
    with pytest.raises(ValueError, match='Source coordinate frame changed'):
        frame.apply(replace(source, planning_frame='other_odom'), 10., .4)
    with pytest.raises(ValueError, match='calibration lost'):
        frame.apply(uncalibrated, 10., .4)


def test_recording_and_session_packet_keep_transform_without_rebasing_replay(tmp_path):
    source = observation()
    fixed = ExperimentFrame(2.).apply(source, 10., .4)
    restored, _ = decode_packet(encode_packet(fixed, {'ready': False}))
    assert restored.coordinate_transform == fixed.coordinate_transform
    recorder = Recorder(tmp_path, load(), 'test', REPO)
    recorder.observation(fixed)
    recorder.close({'completed': True})
    replayed = next(snapshots(recorder.directory))
    assert replayed.coordinate_transform == fixed.coordinate_transform
    np.testing.assert_array_equal(replayed.observation['sensors']['lidar']['pointcloud'],
                                  fixed.observation['sensors']['lidar']['pointcloud'])
    # Existing recordings do not gain a new reference on replay.
    index = recorder.directory / 'observations.jsonl'
    entry = json.loads(index.read_text())
    del entry['snapshot']['coordinate_transform']
    index.write_text(json.dumps(entry)+'\n')
    assert next(snapshots(recorder.directory)).coordinate_transform is None


def test_live_entrypoint_rebases_before_map_boundary_check(monkeypatch):
    from deploy.app import keyboard, runtime, session
    from deploy.robot import sensors
    from deploy.tests.test_robot_inputs import sensor, odometry, cloud
    source = sensor()
    message = odometry(1_000_000_000, 30.)
    message.pose.pose.position.y = 40.
    message.pose.pose.orientation.w = message.pose.pose.orientation.z = np.sqrt(.5)
    source._pose(message)
    source._lidar(cloud(1_000_000_000))
    source.lidar[1]['pointcloud'] = np.array([[30., 42., .7], [29., 42., .7]])
    source.close = lambda: None
    monkeypatch.setattr(sensors, 'RosSensors', lambda *args: source)
    keys = deque([None, ('q', None)])
    monkeypatch.setattr(keyboard, 'Keyboard', lambda: SimpleNamespace(
        editing=None, poll=lambda: keys.popleft(), close=lambda: None))
    sessions = []
    constructor = session.Session
    def create_session(*args, **kwargs):
        obj = constructor(*args, **kwargs)
        close = obj.close
        obj.close = lambda summary: close(summary, grace=0)
        sessions.append(obj)
        return obj
    monkeypatch.setattr(session, 'Session', create_session)
    args = SimpleNamespace(allow_motion=False, demo=False, session_port=0, camera='none', rgb_topic=None)
    config = load()
    config['robot'] = source.config
    assert runtime.run_live(args, config) == 0
    summary = sessions[0].summary
    assert summary['map_created'] and summary['planning_frame'] == 'experiment'
    assert summary['coordinate_transform']['origin_source_m'][:2] == [30., 40.]
    batch = sessions[0].batch(0, 0)
    assert len([e for e in batch['events'] if e['event'] == 'coordinate_frame_initialized']) == 1
    snap, scene = decode_packet(batch['frames'][0]['packet'])
    assert scene['ready'] and snap.planning_frame == 'experiment'
    np.testing.assert_allclose(snap.observation['position'], [0, 0, .3], atol=1e-7)
