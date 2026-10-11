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
