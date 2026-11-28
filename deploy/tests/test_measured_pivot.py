"""Field pivot admission, command dispatch and live-map invalidation."""
import math

import numpy as np
import pytest

from deploy.app.config import load
from deploy.app.navigation import NavigationEngine
from deploy.robot.pivot import measured_pivot_clear
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear


@pytest.fixture
def field():
    engine = NavigationEngine(load()['navigation'])
    xy = np.array([3.79, -.185])
    engine.set_goal((4., -1.5), 0.)
    engine.phase = 'navigate'
    engine.position = np.r_[xy, .35]
    engine.heading = math.radians(47.5)
    p = engine.planner
    p.free = engine.layers.interior.copy()
    p.planning_observed = p.free.copy()
    p.free[p.cell((3.95, .45))] = False
    return engine, xy, math.radians(-131.4)


def test_recorded_half_turn_reaches_yaw_dispatch_and_rechecks_each_scan(field):
    engine, xy, end = field
    p, heading = engine.planner, engine.heading
    assert not body_rotation_clear(p, [], xy, heading, end, margin=.025)
    assert measured_pivot_clear(p, [], xy, heading, end)
    target = xy + 1.4*np.array([math.cos(end), math.sin(end)])
    tracker = engine._tracker((tuple(target), tuple(engine.goal[:2])))
    p.before_turn_search = lambda reason: pytest.fail(reason)
    for tick in range(8):
        p.control_time = engine.now = 1. + tick*.1
        action, detail = tracker.action(xy, heading, p, [])
        command = engine._output(action['move_by_speed'])
        assert detail['measured_pose_pivot'] and not np.any(command[:2])
        if command[2] < 0.:
            break
    else:
        pytest.fail('No yaw command after settling')
    p.free[p.cell(xy)] = False
    p.control_time += .1
    action, detail = tracker.action(xy, heading, p, [])
    np.testing.assert_array_equal(action['move_by_speed'], np.zeros(3))
    assert detail['needs_replan'] and not detail['body_motion_safe']


def test_margin_applies_to_whole_cells_boxes_and_map_edges(field):
    engine, xy, _ = field
    p = engine.planner
    p.free[:] = True
    p.free[p.cell((3.95, .45))] = False
    # With the nose pointing +Y the obstacle starts at y=.40.
    assert measured_pivot_clear(p, [], (3.95, -.13), math.pi/2, math.pi/2)
    assert not measured_pivot_clear(p, [], (3.95, -.12), math.pi/2, math.pi/2)
    p.free[:] = True
    box = BoxState('test', tuple(xy + [.3, 0.]), (.30, .80, .30))
    assert not measured_pivot_clear(p, [box], xy, 0., math.pi)
    assert not measured_pivot_clear(p, [], (p.limits[0][0]+.2, 0.), 0., 0.)


def test_obstacle_only_in_middle_of_turn_is_blocking(field):
    engine, xy, _ = field
    p = engine.planner
    p.free[:] = True
    p.free[p.cell(xy + [.05, .46])] = False
    assert all(measured_pivot_clear(p, [], xy, angle, angle) for angle in (0., math.pi))
    assert not measured_pivot_clear(p, [], xy, 0., math.pi)
