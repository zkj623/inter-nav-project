"""Self-hit filtering without loading Isaac sensor dependencies."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'lidar_returns', Path(__file__).resolve().parents[1] / 'grutopia_extension/sensors/lidar_returns.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_self_filter_preserves_near_walls_and_other_robots():
    points = np.array([[0.3, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [8, 0, 0]])
    paths = ['/World/go2/base/collision', '/World/wall', '/World/go2_other/base', '']
    result, count = module.exclude_robot_returns(points, paths, '/World/go2')
    assert count == 1
    np.testing.assert_array_equal(result, points[1:])


def test_empty_and_mismatched_frames():
    result, count = module.exclude_robot_returns(np.empty((0, 3)), [], '/World/go2')
    assert result.shape == (0, 3) and count == 0
    with pytest.raises(ValueError, match='counts differ'):
        module.exclude_robot_returns([[0, 0, 0]], [], '/World/go2')


def test_scan_keeps_leg_identity_when_backend_advances_to_a_floor_hit():
    # The online turn diagnostic leaked a 0.22m-high rear-leg return when a
    # later read supplied a floor identity for that same beam index.
    points = np.array([[-.36, -.13, .22], [1.2, 0., .3]], dtype=np.float32)
    paths = np.array(['/World/go2/RR_thigh', '/World/wall'], dtype=object)
    position = np.array([11.17, -.5, .45])
    orientation = np.array([1., 0., 0., 0.])
    frame = module.freeze_lidar_scan(points, paths, position, orientation, 1280, 6.4)
    points[0] = [0., 0., .01]
    paths[0] = '/World/floor'
    position[:] = [12., 1., .45]
    orientation[:] = [0., 0., 0., 1.]
    result, removed = module.exclude_robot_returns(frame['point_cloud'], frame['hit_paths'], '/World/go2')
    assert removed == 1
    np.testing.assert_allclose(result, [[1.2, 0., .3]])
    np.testing.assert_allclose(frame['position'], [11.17, -.5, .45])
    np.testing.assert_array_equal(frame['orientation'], [1., 0., 0., 0.])
    assert frame['physics_step'] == 1280 and frame['time'] == 6.4


def test_next_matched_scan_retains_floor_wall_and_other_robot_returns():
    points = np.array([[0., 0., .01], [.4, 0., .2], [.3, 0., .25]])
    paths = ['/World/floor', '/World/wall', '/World/go2_other/RR_thigh']
    frame = module.freeze_lidar_scan(points, paths, [0, 0, 0], [1, 0, 0, 0], 1281, 6.405)
    result, removed = module.exclude_robot_returns(frame['point_cloud'], frame['hit_paths'], '/World/go2')
    assert removed == 0
    np.testing.assert_allclose(result, points)


def test_mismatched_scan_is_not_published():
    with pytest.raises(ValueError, match='counts differ'):
        module.freeze_lidar_scan([[0, 0, 0]], [], [0, 0, 0], [1, 0, 0, 0], 1, .005)
