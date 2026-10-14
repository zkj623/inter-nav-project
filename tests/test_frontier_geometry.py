"""Adversarial geometry checks through main's semantic navigation runtime."""
from math import cos, sin
from unittest.mock import patch

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.exploration_frontiers import frontier_goals
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner, MappingConfig, OccupancyGridMap, PlanningError
from grutopia_extension.interactive_navigation.semantic_exploration_component import SemanticExplorationComponent, SemanticExplorationConfig


def component():
    return SemanticExplorationComponent(SemanticExplorationConfig(
        target_query='fire hydrant', semantic_detection_mode='isaac', require_visual_scan=False, enable_target_cues=False,
        mapping=MappingConfig(x_limits=(0, 6), y_limits=(0, 5), grid_resolution=.1,
                              robot_radius=.1, obstacle_inflation_radius=.1),
        frontier_selection_interval=1, topology_update_interval=1))


def observation(x=1., y=1., yaw=0.):
    return dict(position=(x, y, .4), orientation=(cos(yaw/2), 0, 0, sin(yaw/2)), sensors={})


def ready(c):
    c.mapping.map.occupancy.observed[5:40, 5:40] = True
    c.update(0, observation())
    c._frontier_phase = 'select'
    c.update(1, observation())


def test_exploration_uses_voronoi_and_keeps_fallback_inside_observed_space():
    c = component()
    occ = c.mapping.map.occupancy
    occ.observed[12:19, 3:38] = True
    c.mapping.semantic_voronoi.update()
    start, goal = (0.55, 1.55, 0.4), (3.45, 1.55, 0.4)
    paths = [c.mapping._map_plan(start, goal)]
    assert c.mapping.voronoi_plan_count == 1
    with patch.object(c.mapping.voronoi_planner, 'plan', side_effect=PlanningError('no skeleton')):
        paths.append(c.mapping._map_plan(start, goal))
        with pytest.raises(PlanningError):
            c.mapping._map_plan(start, (3.45, 2.55, 0.4))
    checker = AStarMapPlanner(occ, observed_only=True)
    blocked = occ.inflated_mask() | ~occ.observed
    for path in paths:
        cells = [occ.world_to_cell(p[:2]) for p in (start, *path)]
        assert all(checker.segment_clear(a, b, blocked) for a, b in zip(cells, cells[1:]))



def test_voronoi_rejects_unknown_endpoint_and_diagonal_corner():
    from types import SimpleNamespace
    from grutopia_extension.interactive_navigation.semantic_voronoi import VoronoiPathPlanner

    occ = component().mapping.map.occupancy
    occ.observed[15, 24] = occ.observed[16, 25] = True
    graph = SimpleNamespace(
        occupancy=occ, cached_snapshot=lambda: SimpleNamespace(skeleton_cells=((15, 24), (16, 25)))
    )
    planner = VoronoiPathPlanner(graph, observed_only=True)
    for goal in [(2.55, 1.65, 0.4), (2.55, 1.55, 0.4)]:
        with pytest.raises(PlanningError):
            planner.plan((2.45, 1.55, 0.4), goal)



def test_strict_routes_never_cross_unknown_or_snap_blocked_goal():
    occ = OccupancyGridMap(MappingConfig(x_limits=(0, 5), y_limits=(0, 5), robot_radius=0.01))
    occ.observed[:] = True
    occ.observed[:, 25] = False
    planner = AStarMapPlanner(occ, observed_only=True)
    with pytest.raises(PlanningError):
        planner.plan((1, 1, 0.4), (4, 1, 0.4))
    with pytest.raises(PlanningError):
        planner.plan((1, 1, 0.4), (2.55, 1, 0.4))
    # An observed doorway is a valid detour, including path simplification.
    occ.observed[35:40, 25] = True
    path = planner.plan((1, 1, 0.4), (4, 1, 0.4))
    assert max(p[1] for p in path) >= 3.5



def test_strict_path_cannot_squeeze_diagonally_between_unknown_cells():
    occ = OccupancyGridMap(MappingConfig(x_limits=(0, 2), y_limits=(0, 2), robot_radius=0.01))
    occ.observed[5, 5] = occ.observed[6, 6] = True
    with pytest.raises(PlanningError):
        AStarMapPlanner(occ, observed_only=True).plan((0.55, 0.55, 0.4), (0.65, 0.65, 0.4))



def test_candidate_endpoints_are_actual_reachable_frontier_cells():
    c = component()
    ready(c)
    occ = c.mapping.map.occupancy
    topology = c.mapping.semantic_voronoi.cached_snapshot()
    cells = {cell for frontier in topology.frontiers for cell in frontier.cells}
    goals = frontier_goals(occ, topology, (1, 1), set())
    assert goals
    assert all(occ.world_to_cell(g.position) in cells for g in goals)
    assert all(np.isfinite(g.distance) for g in goals)



def test_switching_navigation_goal_releases_old_route():
    c = component()
    c.mapping.map.occupancy.observed[:] = True
    c.mapping.map.lidar_frames = 1
    c.mapping.point_navigation_action((2, 1, 0.4), observation(), step=0)
    first = c.mapping.map.plan_count
    c.mapping.point_navigation_action((1, 3, 0.4), observation(), step=1)
    assert c.mapping.map.plan_count == first + 1
    assert c.mapping._planned_path[-1][:2] == (1, 3)



def test_strict_planner_prefers_clearance_in_wide_doorway():
    occ = OccupancyGridMap(MappingConfig(x_limits=(0, 6), y_limits=(0, 6), robot_radius=0.1))
    occ.observed[:] = True
    occ.mark_box_occupied((2.9, 0), (3.1, 2))
    occ.mark_box_occupied((2.9, 4), (3.1, 6))
    path = AStarMapPlanner(occ, observed_only=True).plan((1, 2.3, 0.4), (5, 2.3, 0.4))
    # The direct path is legal, but leaves less tracking clearance.
    assert max(point[1] for point in path) >= 2.7



def test_fusion_clears_moved_obstacle_only_after_repeated_fresh_free_rays():
    c = component()
    occ = c.mapping.map.occupancy
    cell = occ.world_to_cell((2, 1))
    occ.observed[cell] = True
    occ.log_odds[cell] = 4.0
    endpoint = np.array([[3.0, 1.0, 0.0]])
    occ.update_lidar((1, 1, 0.4), endpoint, max_range=8)
    assert occ.occupied_mask()[cell]
    for _ in range(60):
        occ.update_lidar((1, 1, 0.4), endpoint, max_range=8)
    assert not occ.occupied_mask()[cell]



def test_strict_velocity_keeps_translation_on_checked_segment_when_lateral_speed_saturates():
    c = component()
    c.mapping.map.occupancy.observed[:] = True
    c.mapping.map.lidar_frames = 1
    obs = observation(yaw=-0.65)
    command = c.mapping.point_navigation_action(
        (3, 1, 0.4),
        obs,
        step=0,
        velocity_control=True,
        max_forward_speed=0.5,
        max_lateral_speed=0.12,
    )['move_by_speed']
    body = np.asarray(command[:2])
    yaw = -0.65
    world = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]) @ body
    target = c.mapping._lookahead_target(np.array(obs['position'][:2]))
    direction = target - obs['position'][:2]
    assert abs(world[0] * direction[1] - world[1] * direction[0]) < 1e-9
    assert np.dot(world, direction) > 0
    assert abs(body[1]) <= 0.12 + 1e-9



@pytest.mark.parametrize('yaw', [np.pi, np.pi / 2])
def test_safe_path_translation_does_not_require_a_stationary_turn(yaw):
    c = component()
    c.mapping.map.occupancy.observed[:] = True
    c.mapping.map.lidar_frames = 1
    command = c.mapping.point_navigation_action(
        (3, 1, 0.4), observation(yaw=yaw), step=0, velocity_control=True,
        max_forward_speed=0.35, max_lateral_speed=0.12,
    )['move_by_speed']
    assert np.linalg.norm(command[:2]) > 0
    assert command[0] >= -0.12 - 1e-9
    assert abs(command[1]) <= 0.12 + 1e-9
    world = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]) @ command[:2]
    assert world[0] > 0



def test_buffer_exit_never_uses_unknown_space_or_decreases_clearance():
    from grutopia_extension.interactive_navigation.exploration_frontiers import buffer_exit_goal
    from scipy.ndimage import distance_transform_edt
    from grutopia_extension.interactive_navigation.mapping import _bresenham

    occ = OccupancyGridMap(MappingConfig(x_limits=(0, 3), y_limits=(0, 3),
                                       robot_radius=.1, obstacle_inflation_radius=.4))
    occ.observed[:] = True
    occ.mark_box_occupied((0., 0.), (.15, 3.))
    start = (.35, 1.55)
    goal = buffer_exit_goal(occ, start)
    assert goal is not None
    clearance = distance_transform_edt(~occ.occupied_mask()) * occ.config.grid_resolution
    start_cell = occ.world_to_cell(start)
    for cell in _bresenham(start_cell, occ.world_to_cell(goal)):
        assert occ.observed[cell]
        assert clearance[cell] >= clearance[start_cell]
    occ.observed[:, 4:] = False
    assert buffer_exit_goal(occ, start, preferred=goal) is None


def target_component():
    from dataclasses import replace
    from grutopia_extension.interactive_navigation.mapping import SemanticDetection
    c = SemanticExplorationComponent(replace(component().config, target_distance=1.1))
    c.mapping.map.occupancy.observed[:] = True
    c.mapping.map.scene_graph.update_detections([
        SemanticDetection('fire hydrant', (3., 2., .5), step=i) for i in range(8)
    ])
    return c


def test_target_approach_avoids_isolated_free_island():
    c = target_component()
    occ = c.mapping.map.occupancy
    occ.observed[:] = False
    occ.observed[:, :22] = True
    occ.observed[15:25, 25:35] = True
    obs = observation(1., 2.)
    c.update(0, obs)
    assert c.target_confirmed
    assert c.current_goal[0] < 2.2
    AStarMapPlanner(occ, observed_only=True).plan(obs['position'], c.current_goal)


def test_target_goal_reselected_when_inflated_and_then_kept_stable():
    c = target_component()
    obs = observation(1., 2.)
    c.update(0, obs)
    old = c.current_goal
    occ = c.mapping.map.occupancy
    occ.static_occupied[occ.world_to_cell(old[:2])] = True
    c.update(1, obs)
    assert c.current_goal is not None and c.current_goal != old
    AStarMapPlanner(occ, observed_only=True).plan(obs['position'], c.current_goal)
    goal = c.current_goal
    c.update(2, observation(1.1, 2.))
    assert c.current_goal == goal


def test_free_target_goal_reselected_after_route_is_disconnected():
    c = target_component()
    obs = observation(1., 2.)
    c.update(0, obs)
    old = c.current_goal
    occ = c.mapping.map.occupancy
    occ.observed[:, 22:25] = False
    assert occ.observed[occ.world_to_cell(old[:2])]
    c.update(1, obs)
    assert c.current_goal != old and c.current_goal[0] < 2.2


def test_target_planning_failure_reselects_without_discarding_target():
    c = target_component()
    obs = observation(1., 2.)
    c.update(0, obs)
    old, node = c.current_goal, c.target_node.node_id
    def fail(*args, **kwargs):
        c.mapping.planning_failures += 1
        return {'move_by_speed': [0., 0., 0.]}
    with patch.object(c.mapping, 'point_navigation_action', side_effect=fail):
        c.action(0, obs)
    assert c.current_goal is None
    c.update(1, obs)
    assert c.current_goal is not None and c.current_goal != old
    assert c.target_node.node_id == node and c.target_confirmed


def test_unreachable_target_has_no_fabricated_goal_or_false_arrival():
    c = target_component()
    c.mapping.map.occupancy.observed[:, 15:] = False
    obs = observation(1., 2.)
    c.update(0, obs)
    assert c.target_confirmed and c.current_goal is None
    assert c.action(0, obs) == c.warmup_action()
    assert not c.evaluate(0, obs).success


def test_target_arrival_requires_reaching_new_approach_point():
    c = target_component()
    obs = observation(1., 2.)
    c.update(0, obs)
    gx, gy, _ = c.current_goal
    assert not c.evaluate(1, observation(gx-.6, gy)).success
    result = c.evaluate(2, observation(gx-.2, gy))
    assert result.success and result.arrival_threshold == .3
