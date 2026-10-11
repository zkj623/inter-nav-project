"""Regression coverage for planning's frontier lifecycle inside main's component."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grutopia_extension.interactive_navigation.exploration_frontiers import frontier_goals, reachable_distances
from grutopia_extension.interactive_navigation.mapping import MappingConfig, PlanningError
from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent, PointNavigationConfig
from grutopia_extension.interactive_navigation.semantic_exploration_component import SemanticExplorationComponent, SemanticExplorationConfig
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiConfig


def observation(x=1.0, y=1.5):
    return dict(position=(x, y, .5), orientation=(1., 0., 0., 0.), sensors={})


from grutopia_extension.interactive_navigation.mapping import SemanticDetection


def component():
    result = SemanticExplorationComponent(SemanticExplorationConfig(
        target_query='fire hydrant', semantic_detection_mode='isaac',
        mapping=MappingConfig(x_limits=(0., 5.), y_limits=(0., 4.), grid_resolution=.1,
                              robot_radius=.1, safe_recovery_y_limits=(.1, 3.9)),
        voronoi=SemanticVoronoiConfig(spur_length=0), frontier_selection_interval=10,
        topology_update_interval=1, enable_target_cues=False))
    result.mapping.map.occupancy.observed[8:32, 5:30] = True
    obs = observation()
    obs['sensors']['lidar'] = dict(horizontal_fov=360, rotation_frequency=0)
    result.update(0, obs)
    assert result.current_goal is None
    result.mapping.map.lidar_frames += 8
    result.update(1, obs)
    assert result.current_goal is not None
    return result


def test_goal_stays_locked_past_selection_interval():
    c = component()
    goal, key = c.current_goal, c.current_frontier_id
    c.update(20, observation(1.2))
    assert c.current_goal == goal and c.current_frontier_id == key
    assert len(c.decision_history) == 1


def test_route_failure_stops_and_rescans_with_feedback():
    c = component()
    key = c.current_frontier_id
    def fail(*args, **kwargs):
        c.mapping.planning_failures += 1
        return c.warmup_action()
    with patch.object(c.mapping, 'point_navigation_action', side_effect=fail):
        assert c.action(2, observation()) == c.warmup_action()
    assert c.current_goal is None
    assert c.planner.failure_counts[key] == 1
    assert c._frontier_phase == 'scan'
    assert any(e.get('reason') == 'path_unreachable' for e in c.frontier_history)


def test_scan_needs_fresh_evidence_and_marks_visit():
    c = component()
    key, goal = c.current_frontier_id, c.current_goal
    obs = observation(*goal[:2])
    obs['sensors']['lidar'] = dict(horizontal_fov=360, rotation_frequency=0)
    with patch.object(c.mapping, 'update'):
        c.update(2, obs)
        c.update(3, obs)
        assert key not in c._observed_frontiers
        assert c._frontier_phase == 'scan'
        c.mapping.map.lidar_frames += 8
        c.update(4, obs)
        assert key in c._observed_frontiers


def test_target_preempts_scan_and_false_match_resumes_without_losing_map():
    c = component()
    obs = observation(*c.current_goal[:2])
    c.update(2, obs)
    assert c._frontier_phase == 'scan'
    world = c.mapping.map
    world.scene_graph.update_detections([
        SemanticDetection('fire hydrant', (2., 1.5, .7), confidence=.9)
        for _ in range(8)])
    with patch.object(c.mapping, 'update') as update:
        c.update(3, obs)
        update.assert_called_once_with(3, obs)
    assert c.target_confirmed
    assert c._frontier_paused
    assert c.state == 'navigate_to_semantic_target'
    c.target_node = None
    c.target_navigation_position = None
    with patch.object(c, '_best_target_node', return_value=None), patch.object(c.mapping, 'update') as update:
        c.update(4, obs)
        update.assert_called_once_with(4, obs)
    assert c.mapping.map is world
    assert not c._frontier_paused
    assert c._frontier_phase == 'scan'
    assert c.current_goal is None


def test_exhaustion_reports_search_failure_not_target_success():
    c = component()
    c._finish_frontier('reachable_observation_goals_exhausted')
    result = c.evaluate(3, observation())
    assert result.terminal and not result.success
    assert result.failure_reason == 'reachable_observation_goals_exhausted'


def test_missing_target_selection_uses_original_adaptive_planner():
    from grutopia_extension.interactive_navigation.semantic_exploration import AdaptiveExplorationPlanner

    c = component()
    assert isinstance(c.planner, AdaptiveExplorationPlanner)
    with patch.object(c.planner, 'select_goal', wraps=c.planner.select_goal) as select:
        c._select_frontier(2, observation()['position'])
    select.assert_called_once()
    assert select.call_args.kwargs['frontiers']
    assert c.last_decision.selected_frontier_id == c.current_frontier_id


def test_actual_reachable_cells_and_stable_spatial_keys():
    c = component()
    occupancy = c.mapping.map.occupancy
    cells = ((8, 8), (8, 9), (31, 8), (38, 38))
    snapshot = SimpleNamespace(frontiers=[SimpleNamespace(frontier_id='old', cells=cells)])
    first = frontier_goals(occupancy, snapshot, (1., 1.5), set())
    snapshot.frontiers[0].frontier_id = 'new'
    second = frontier_goals(occupancy, snapshot, (1., 1.5), set())
    assert first == second and first
    reach = reachable_distances(occupancy, (1., 1.5))
    for goal in first:
        cell = occupancy.world_to_cell(goal.position)
        assert cell in cells and cell in reach
        assert np.isclose(goal.distance, reach[cell])
    assert (38, 38) not in reach


def test_point_navigation_calls_voronoi_and_fallback_counts():
    c = PointNavigationComponent(PointNavigationConfig(goal=(2., 0., .5), prefer_voronoi_paths=True))
    runtime = c.mapping
    runtime.map.lidar_frames = 1
    runtime.map.occupancy.observed.fill(True)
    path = ((0., 0., .5), (1., 0., .5), (2., 0., .5))
    obs = observation(0., 0.)
    with patch.object(runtime.voronoi_planner, 'plan', return_value=path) as planner:
        c.action(0, obs)
        planner.assert_called_once()
    assert runtime.voronoi_plan_count == 1
    runtime.invalidate_path()
    with patch.object(runtime.voronoi_planner, 'plan', side_effect=PlanningError('disconnected')):
        c.action(1, obs)
    assert runtime.voronoi_plan_fallbacks == 1
    assert runtime.planning_failures == 0


def test_real_voronoi_route_is_used_by_point_entry():
    c = PointNavigationComponent(PointNavigationConfig(
        goal=(2., 0., .5), prefer_voronoi_paths=True,
        mapping=MappingConfig(x_limits=(-1., 4.), y_limits=(-2., 2.),
                              safe_recovery_y_limits=(-1.5, 1.5))))
    runtime = c.mapping
    runtime.map.lidar_frames = 1
    runtime.map.occupancy.observed.fill(True)
    c.action(0, observation(0., 0.))
    assert runtime.voronoi_plan_count == 1
    assert runtime.voronoi_plan_fallbacks == 0
    np.testing.assert_allclose(runtime._planned_path[-1], c.goal)


def test_scan_timeout_stops_without_fresh_lidar():
    from dataclasses import replace

    c = component()
    c.config = replace(c.config, frontier_scan_timeout=3)
    c._fail_frontier('path_unreachable')
    with patch.object(c.mapping, 'update'):
        for step in range(2, 7):
            c.update(step, observation())
    result = c.evaluate(7, observation())
    assert result.terminal and result.failure_reason == 'scan_timeout'
    assert c.action(7, observation()) == c.warmup_action()


def test_no_progress_failure_is_returned_to_original_planner():
    from dataclasses import replace

    c = component()
    key = c.current_frontier_id
    c.config = replace(c.config, frontier_progress_timeout=3)
    with patch.object(c.mapping, 'update'):
        for step in range(2, 5):
            c.update(step, observation())
    assert c.current_goal is None
    assert c.planner.failure_counts[key] == 1
    assert c.frontier_history[-1]['reason'] == 'no_motion_progress'
    assert c._frontier_phase == 'scan'


def test_failed_frontier_gives_other_candidates_priority_and_is_bounded():
    from grutopia_extension.interactive_navigation.exploration_frontiers import ObservationGoal

    c = component()
    goals = [ObservationGoal('blocked', (2., 1.5), 1., 10., 5.),
             ObservationGoal('other', (1., 2.5), 1., 2., 1.)]
    c.planner.record_failure('blocked')
    with patch('grutopia_extension.interactive_navigation.semantic_exploration_component.frontier_goals', return_value=goals):
        c._select_frontier(2, observation()['position'])
        assert c.current_frontier_id == 'other'
        c.planner.record_failure('blocked')
        c._select_frontier(3, observation()['position'])
    assert 'blocked' in c.planner.blacklist
    assert all(x.frontier_id != 'blocked' for x in c.last_decision.candidates)


def test_reachable_candidates_still_use_original_optional_semantic_scorer():
    from grutopia_extension.interactive_navigation.exploration_frontiers import ObservationGoal
    from grutopia_extension.interactive_navigation.semantic_exploration import Qwen3Scorer

    c = component()
    c.planner.scorer = Qwen3Scorer(generator=lambda *a, **k: '{"a":0.0,"b":1.0}')
    goals = [ObservationGoal('a', (2., 1.5), 1., 1., 1.),
             ObservationGoal('b', (1., 2.5), 1., 1., 1.)]
    with patch.object(c.mapping.semantic_voronoi, 'update', return_value={'semantic_evidence': 1.}), patch(
        'grutopia_extension.interactive_navigation.semantic_exploration_component.frontier_goals', return_value=goals
    ):
        c._select_frontier(2, observation()['position'])
    assert c.last_decision.used_model
    assert c.current_frontier_id == 'b'


def test_buffer_exit_resumes_scan_and_never_counts_as_arrival():
    from dataclasses import replace

    c = component()
    key = c.current_frontier_id
    occ = c.mapping.map.occupancy
    occ.config = replace(occ.config, robot_radius=.1, obstacle_inflation_radius=.4)
    occ.observed[:] = True
    occ.mark_box_occupied((0., 0.), (.15, 4.))
    obs = observation(.35, 2.)
    with patch.object(c.mapping, 'update'):
        c.update(2, obs)
        assert c.state == 'clear_start'
        assert c.planner.failure_counts[key] == 1
        assert c.action(2, obs)['move_by_speed'][0] > 0
        exit_xy = c.mapping._buffer_exit
        c.update(3, observation(exit_xy[0], exit_xy[1] + .06))
    assert c._frontier_phase == 'scan'
    assert key not in c.planner.visit_counts
    assert c.frontier_history[-1]['kind'] == 'start_cleared'


def test_blocked_physical_start_stops_and_reports_timeout():
    c = component()
    occ = c.mapping.map.occupancy
    occ.observed[:] = True
    occ.mark_box_occupied((.5, .5), (1.5, 2.))
    c.mapping.buffer_recovery_timeout = 3
    with patch.object(c.mapping, 'update'):
        for step in range(2, 6):
            c.update(step, observation())
            assert c.action(step, observation()) == c.warmup_action()
    result = c.evaluate(5, observation())
    assert result.terminal and not result.success
    assert result.failure_reason == 'buffer_exit_timeout'
