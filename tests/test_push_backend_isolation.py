"""Simulation defaults and dependency isolation at the execution boundary."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideConfig
from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
from grutopia_extension.interactive_navigation.mapping import MappingConfig
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_footprint_samples


def test_simulation_skill_loads_without_deploy_or_robot_dependencies():
    source = '''
import builtins
original = builtins.__import__
def isolated(name, *args, **kwargs):
    if name.split('.')[0] in {'deploy', 'rclpy', 'unitree_sdk2py'}:
        raise AssertionError('Simulation imported device adapter: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = isolated
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.push_aside_backend import PushAsideBackend
from grutopia_extension.interactive_navigation.go2_heading_trim import HeadingTrimControl
assert PushAsideBackend().yaw_servo(.1).__class__.__name__ == 'ContactYawServo'
'''
    result = subprocess.run([sys.executable, '-c', source], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr


def test_original_positional_skill_configuration_keeps_its_field_order():
    settings = PushAsideConfig(.4, .18, .3)
    assert (settings.front_gap, settings.front_contact_reach, settings.lateral_speed) == (.4, .18, .3)


@pytest.mark.parametrize('width', [.16, .22, .35])
def test_native_navigation_and_map_filter_keep_their_distinct_original_widths(width):
    planner = SidePushPlanner(np.ones((80, 80), bool), [[-2, 2], [-2, 2]], resolution=.05)
    planner.navigation_half_width = width
    x, y = navigation_footprint_samples(planner)
    assert x.min() == -.32 and x.max() == .5
    assert y.min() == -width and y.max() == width
    mapping = OnlineNavigationMap(planner, MappingConfig(x_limits=(-2, 2), y_limits=(-2, 2), grid_resolution=.05))
    mapping.robot_xy, mapping.robot_yaw = np.array([.5, -.4]), 0.
    footprint = mapping.robot_footprint_box(np.array([1., 0.]))
    # Online mapping historically filters .44 m irrespective of route half-width.
    assert footprint.dimensions == (.82, .44, .5)
    np.testing.assert_allclose(footprint.xy, [.59, -.4])
    mapping.measured_scan_masks(np.ones((2, 2), bool), np.zeros((2, 2), bool))
    assert not hasattr(mapping, 'measured_hit')
