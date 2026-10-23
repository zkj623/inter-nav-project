"""Direct routing must preserve observed footprint, clearance and runtime checks."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner, MappingConfig, OccupancyGridMap, PlanningError
from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent, PointNavigationConfig


def component(margin=.10):
    config = PointNavigationConfig(
        goal=(2., .05, .3), prefer_voronoi_paths=True, velocity_control=True,
        use_scene_graph=False, safe_base_height=.22, fall_height=.16,
        replan_interval_steps=3, direct_path_tracking_margin=margin,
        mapping=MappingConfig(x_limits=(-4., 4.), y_limits=(-4., 4.),
                              robot_radius=.30, obstacle_inflation_radius=.40))
    result = PointNavigationComponent(config)
    result.mapping.map.occupancy.observed[:] = True
    result.mapping.map.lidar_frames = 1
    return result


@pytest.mark.parametrize('goal', [(-1., -2., .3), (2., -2., .3)])
def test_open_square_uses_actual_shortest_line_instead_of_central_detour(goal):
    start = (-2., -2., .3)
    runtime = component().mapping
    with patch.object(runtime.voronoi_planner, 'plan', wraps=runtime.voronoi_planner.plan) as skeleton:
        path = runtime._map_plan(start, goal)
    assert path == (goal,)
    skeleton.assert_not_called()
    assert runtime.statistics()['direct_plans'] == 1
    assert runtime.voronoi_plan_count == runtime.voronoi_plan_fallbacks == 0
    old = component(margin=None).mapping
    old_points = np.array([start] + list(old._map_plan(start, goal)))[:, :2]
    assert np.linalg.norm(np.diff(old_points, axis=0), axis=1).sum() > np.linalg.norm(np.subtract(goal, start)) + 1.
    assert old.voronoi_plan_count == 1


@pytest.mark.parametrize('hazard', ['blocking_obstacle', 'nearby_obstacle', 'nearby_unknown', 'narrow_known_strip'])
def test_center_line_alone_is_insufficient_and_voronoi_is_still_used(hazard):
    runtime = component().mapping
    occ = runtime.map.occupancy
    start, goal = (-2., .05, .3), (2., .05, .3)
    if hazard == 'blocking_obstacle':
        occ.mark_occupied((.05, .05), radius=.01, evidence=4.)
    elif hazard == 'nearby_obstacle':
        occ.mark_occupied((.05, .65), radius=.01, evidence=4.)
    elif hazard == 'nearby_unknown':
        occ.observed[occ.world_to_cell((.05, .35))] = False
    else:
        occ.observed[:] = False
        occ.observed[37:44, :] = True
    if hazard != 'blocking_obstacle':
        # Existing centre-line check passes; the new swept margin must reject it.
        assert runtime.map.planner.segment_clear(occ.world_to_cell(start[:2]), occ.world_to_cell(goal[:2]),
                                                 occ.inflated_mask() | ~occ.observed)
    assert not runtime._direct_segment_clear(start[:2], goal[:2])
    path = runtime._map_plan(start, goal)
    assert path and runtime.direct_plan_count == 0 and runtime.voronoi_plan_count == 1


@pytest.mark.parametrize('case', ['unknown_goal', 'unknown_wall', 'outside_map', 'nonfinite'])
def test_no_direct_route_or_unsafe_astar_when_observations_do_not_connect(case):
    runtime = component().mapping
    start, goal = (-2., .05, .3), (2., .05, .3)
    occ = runtime.map.occupancy
    if case == 'unknown_goal':
        occ.observed[occ.world_to_cell(goal[:2])] = False
    elif case == 'unknown_wall':
        occ.observed[:, 39:41] = False
    elif case == 'outside_map':
        goal = (5., .05, .3)
    else:
        assert not runtime._direct_segment_clear((float('nan'), 0), goal[:2])
        return
    with pytest.raises(PlanningError):
        runtime._map_plan(start, goal)
    assert runtime.direct_plan_count == 0


def test_body_must_fit_at_map_boundary_and_at_endpoints():
    runtime = component().mapping
    assert not runtime._direct_segment_clear((-3.8, 0), (-2., 0))
    occ = runtime.map.occupancy
    occ.observed[occ.world_to_cell((2.35, .05))] = False
    # Unknown cell beyond the target is inside the endpoint's body footprint.
    assert not runtime._direct_segment_clear((-2., .05), (2., .05))


def test_existing_larger_static_inflation_is_never_reduced():
    obj = component()
    config = replace(obj.config, mapping=replace(obj.config.mapping, static_obstacle_inflation_radius=.8))
    runtime = PointNavigationComponent(config).mapping
    occ = runtime.map.occupancy
    occ.observed[:] = True
    occ.mark_box_occupied((.01, .71), (.09, .79))
    assert not runtime._direct_segment_clear((-2., .05), (2., .05))


def test_capsule_checks_subcell_geometry_and_diagonal_cell_corners():
    occ = OccupancyGridMap(MappingConfig(x_limits=(0., 4.), y_limits=(0., 4.), grid_resolution=.1))
    checker = AStarMapPlanner(occ)
    blocked = np.zeros_like(occ.observed)
    blocked[10, 10] = True  # square [1, 1.1] x [1, 1.1]
    assert not checker.segment_clear_with_margin((.5, .999), (1.5, .999), blocked, .01)
    # Degenerate segment still requires endpoint clearance.
    assert not checker.segment_clear_with_margin((.99, .99), (.99, .99), blocked, .02)
    assert checker.segment_clear_with_margin((.5, .7), (1.5, .7), blocked, .1)


def test_lost_margin_replans_and_does_not_keep_using_old_direct_route():
    obj = component()
    runtime = obj.mapping
    observation = {'position': (-2., .05, .3), 'orientation': (1., 0., 0., 0.), 'sensors': {}}
    assert obj.action(0, observation)['move_by_speed'][0] > 0
    assert runtime._last_plan_used_direct
    assert runtime._plan_history[-1]['planner'] == 'direct'
    # Normal periodic checks must not repeatedly retry Voronoi for a valid direct route.
    obj.action(4, observation)
    assert runtime.direct_plan_count == 1
    runtime.map.occupancy.mark_occupied((.05, .65), radius=.01, evidence=4.)
    assert runtime._remaining_path_blocked(observation['position'])
    assert not runtime._tracking_segment_clear((-.25, .05), (.3, .05))
    obj.action(5, observation)
    assert not runtime._last_plan_used_direct
    assert runtime.voronoi_plan_count == 1
    assert runtime._plan_history[-1]['planner'] == 'voronoi'
    assert runtime.replan_reasons['blocked'] == 1


@pytest.mark.parametrize('margin', [0., -1., float('nan'), float('inf')])
def test_invalid_margin_is_rejected(margin):
    with pytest.raises(ValueError, match='direct_path_tracking_margin'):
        component(margin)


def test_deploy_point_task_enables_direct():
    from deploy.app.config import load
    from deploy.app.tasks import Tasks
    tasks = Tasks(load()['navigation'])
    runtime = tasks.component.mapping
    assert runtime.direct_path_tracking_margin == .10
    runtime.map.occupancy.observed[:] = True
    assert runtime._map_plan((-2., -2., .3), (-1., -2., .3)) == ((-1., -2., .3),)
    assert runtime.direct_plan_count == 1


def test_failed_voronoi_still_falls_back_to_observed_only_astar():
    runtime = component().mapping
    occ = runtime.map.occupancy
    occ.mark_occupied((.05, .05), radius=.01, evidence=4.)
    start, goal = (-2., .05, .3), (2., .05, .3)
    with patch.object(runtime.voronoi_planner, 'plan', side_effect=PlanningError('disconnected skeleton')):
        path = runtime._map_plan(start, goal)
    assert runtime.direct_plan_count == 0 and runtime.voronoi_plan_count == 0
    assert runtime.voronoi_plan_fallbacks == 1
    points = [start] + list(path)
    blocked = occ.inflated_mask() | ~occ.observed
    for left, right in zip(points, points[1:]):
        assert runtime.map.planner.segment_clear(occ.world_to_cell(left[:2]), occ.world_to_cell(right[:2]), blocked)
