from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.exploration_frontiers import buffer_exit_goal
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner, MappingConfig, OccupancyGridMap, PlanningError, _bresenham
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime


def occupancy():
    occ = OccupancyGridMap(MappingConfig(x_limits=(0, 3), y_limits=(0, 3),
                                        robot_radius=.1, obstacle_inflation_radius=.4))
    occ.observed[:] = True
    occ.mark_box_occupied((0., 0.), (.15, 3.))
    return occ


def observation(x=.35, y=1.55, yaw=0):
    return dict(position=(x,y,.5), orientation=(np.cos(yaw/2),0,0,np.sin(yaw/2)), sensors={})


def test_buffer_exit_preserves_clearance_and_never_enters_unknown():
    occ = occupancy()
    start = (.35, 1.55)
    goal = buffer_exit_goal(occ, start)
    assert goal is not None and np.linalg.norm(np.array(goal)-start) <= .8
    clearance = distance_transform_edt(~occ.occupied_mask()) * occ.config.grid_resolution
    initial = clearance[occ.world_to_cell(start)]
    for cell in _bresenham(occ.world_to_cell(start),occ.world_to_cell(goal)):
        assert occ.observed[cell] and clearance[cell] >= initial
    occ.observed[:,4:] = False
    assert buffer_exit_goal(occ, start, preferred=goal) is None


def test_recovery_is_low_speed_revalidates_and_completes():
    occ = occupancy()
    rt = MapNavigationRuntime(mapping_config=occ.config, safe_path_tracking=True, safe_base_height=.3)
    rt.map.occupancy = occ
    assert rt.update_buffer_recovery(0, observation())
    action = rt.buffer_recovery_action(observation(), .5, .1)['move_by_speed']
    assert 0 < action[0] <= .15 and abs(action[1]) <= .1 and action[2] == 0
    goal = rt._buffer_exit
    assert not rt.update_buffer_recovery(1, observation(*goal))
    assert [event['event'] for event in rt.buffer_history] == ['buffer_exit_started','buffer_exit_completed']
    rt.update_buffer_recovery(2, observation())
    occ.observed[:,4:] = False
    assert rt.buffer_recovery_action(observation(), .5, .1)['move_by_speed'] == [0.,0.,0.]


def test_physical_obstacle_cannot_be_relaxed_and_recovery_times_out():
    occ = occupancy()
    rt = MapNavigationRuntime(mapping_config=occ.config, safe_path_tracking=True, buffer_recovery_timeout=3)
    rt.map.occupancy = occ
    obs = observation(x=.15)
    for step in range(4):
        assert rt.update_buffer_recovery(step, obs)
        assert rt.buffer_recovery_action(obs,.5,.2)['move_by_speed'] == [0.,0.,0.]
    assert rt.buffer_recovery_failed
    assert rt.buffer_history[-1]['event'] == 'buffer_exit_timeout'


def test_strict_astar_rejects_unknown_goal_and_diagonal_corner():
    occ=OccupancyGridMap(MappingConfig(x_limits=(0,3),y_limits=(0,3),robot_radius=.01))
    occ.observed[5,5]=occ.observed[6,6]=True
    planner=AStarMapPlanner(occ,observed_only=True)
    with pytest.raises(PlanningError):planner.plan((.55,.55,.5),(.65,.65,.5))
    with pytest.raises(PlanningError):planner.plan((.55,.55,.5),(2.,2.,.5))


def test_common_velocity_scaling_keeps_checked_world_direction():
    rt=MapNavigationRuntime(safe_path_tracking=True, safe_base_height=.3)
    rt.map.occupancy.observed[:]=True
    rt.map.lidar_frames=1
    obs=observation(x=1.,y=1.,yaw=-.65)
    cmd=rt.point_navigation_action((3.,1.,.5),obs,step=0,velocity_control=True,
                                   max_forward_speed=.5,max_lateral_speed=.12)['move_by_speed']
    world=np.array([[np.cos(-.65),-np.sin(-.65)],[np.sin(-.65),np.cos(-.65)]]) @ cmd[:2]
    assert world[0] > 0 and abs(world[1]) < 1e-9
    assert abs(cmd[1]) <= .12 + 1e-9


def test_lookahead_chord_falls_back_to_safe_corner_waypoint():
    rt=MapNavigationRuntime(mapping_config=MappingConfig(x_limits=(0,3),y_limits=(0,3),robot_radius=.01),
                            safe_path_tracking=True, safe_base_height=.3)
    occ=rt.map.occupancy;occ.observed[:]=True
    occ.mark_occupied((1.35,1.25),radius=.01,evidence=4)
    rt._planned_path=((1.05,1.55,.5),(1.55,1.55,.5))
    rt._waypoint_index=0
    rt.velocity_lookahead_distance=1.
    obs=observation(x=1.05,y=1.05)
    from grutopia_extension.interactive_navigation.state_machine import InteractionState
    from types import SimpleNamespace
    decision=SimpleNamespace(state=InteractionState.NAVIGATE_TO_GOAL)
    with patch.object(rt,'action_for',return_value={'move_along_path':[rt._planned_path]}):
        cmd=rt.velocity_action_for(decision,obs,step=0)['move_by_speed']
    assert rt.tracking_chord_fallbacks == 1
    assert abs(cmd[0]) < 1e-9 and cmd[1] > 0
    occ.mark_occupied((1.05,1.35),radius=.01,evidence=4)
    with patch.object(rt,'action_for',return_value={'move_along_path':[rt._planned_path]}):
        cmd=rt.velocity_action_for(decision,obs,step=1)['move_by_speed']
    assert cmd == [0.,0.,0.] and rt._planned_path is None


def test_static_buffer_exit_does_not_use_smaller_dynamic_margin():
    occ=occupancy()
    occ.config=replace(occ.config, obstacle_inflation_radius=.2, static_obstacle_inflation_radius=.5)
    start=(.35,1.55)
    # A preferred diagonal exit dips closer to the wall before reaching free space.
    with patch('scipy.ndimage.distance_transform_edt') as distance:
        clearance=np.ones(occ.observed.shape)
        clearance[:,:3]=0
        clearance[:,3]=.3
        clearance[:,4]=.25
        distance.return_value=clearance/occ.config.grid_resolution
        # Must reject every route through the lower-clearance band.
        assert buffer_exit_goal(occ,start,preferred=(.75,1.55)) is None


def test_point_success_waits_for_buffer_exit():
    from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent, PointNavigationConfig
    c=PointNavigationComponent(PointNavigationConfig(goal=(.35,1.55,.5),fall_height=.1))
    c.mapping.buffer_recovery_active=True
    assert not c.evaluate(0,observation()).success


@pytest.mark.parametrize('offset', [(-.002, 0.), (.002, 0.), (.02, .02)])
def test_reached_fallback_waypoint_stops_and_replans_instead_of_oscillating(offset):
    from types import SimpleNamespace
    from grutopia_extension.interactive_navigation.state_machine import InteractionState

    runtime = MapNavigationRuntime(
        mapping_config=MappingConfig(x_limits=(0, 3), y_limits=(0, 3), robot_radius=.01),
        safe_path_tracking=True, safe_base_height=.3,
    )
    occupancy = runtime.map.occupancy
    occupancy.observed[:] = True
    occupancy.mark_occupied((1.35, 1.55), radius=.01, evidence=4)
    runtime.map.lidar_frames = 1
    path = ((1.05, 1.55, .5), (1.55, 1.55, .5), (2.05, 1.55, .5))
    runtime._planned_path = path
    runtime._planned_goal = path[-1]
    runtime._planned_state = InteractionState.NAVIGATE_TO_GOAL
    obs = observation(x=1.05 + offset[0], y=1.55 + offset[1], yaw=1.6)
    decision = SimpleNamespace(state=InteractionState.NAVIGATE_TO_GOAL)
    assert runtime._tracking_segment_clear(obs['position'][:2], path[0][:2])
    assert not runtime._tracking_segment_clear(obs['position'][:2], runtime._lookahead_target(np.array(obs['position'][:2])))
    with patch.object(runtime, 'action_for', return_value={'move_along_path': [path]}):
        command = runtime.velocity_action_for(decision, obs, step=0)
    assert command['move_by_speed'] == [0., 0., 0.]
    assert runtime._planned_path is None
    assert runtime.last_replan['reason'] == 'reached_tracking_waypoint'

    # The next normal navigation call must find a checked detour, not remain
    # stopped or bypass the obstacle in order to consume the old waypoint.
    resumed = runtime.point_navigation_action(path[-1], obs, step=1, velocity_control=True)
    assert np.linalg.norm(resumed['move_by_speed'][:2]) > 0
    assert runtime.map.plan_count == 1
    route = (obs['position'], *runtime._planned_path)
    assert all(runtime._tracking_segment_clear(a[:2], b[:2]) for a, b in zip(route, route[1:]))


def test_near_waypoint_keeps_moving_when_lookahead_is_clear():
    from types import SimpleNamespace
    from grutopia_extension.interactive_navigation.state_machine import InteractionState

    runtime = MapNavigationRuntime(safe_path_tracking=True)
    runtime.map.occupancy.observed[:] = True
    path = ((1.05, 1.55, .5), (2.05, 1.55, .5))
    runtime._planned_path = path
    decision = SimpleNamespace(state=InteractionState.NAVIGATE_TO_GOAL)
    with patch.object(runtime, 'action_for', return_value={'move_along_path': [path]}):
        command = runtime.velocity_action_for(decision, observation(1.049, 1.55), step=0)
    assert command['move_by_speed'][0] > 0
    assert runtime.replan_count == 0


def test_corner_fallback_still_approaches_waypoint_before_precision_limit():
    from types import SimpleNamespace
    from grutopia_extension.interactive_navigation.state_machine import InteractionState

    runtime = MapNavigationRuntime(
        mapping_config=MappingConfig(x_limits=(0, 3), y_limits=(0, 3), robot_radius=.01),
        safe_path_tracking=True,
    )
    runtime.map.occupancy.observed[:] = True
    runtime.map.occupancy.mark_occupied((1.35, 1.45), radius=.01, evidence=4)
    path = ((1.05, 1.55, .5), (1.55, 1.55, .5))
    runtime._planned_path = path
    obs = observation(1.05, 1.35)
    decision = SimpleNamespace(state=InteractionState.NAVIGATE_TO_GOAL)
    assert runtime._tracking_segment_clear(obs['position'][:2], path[0][:2])
    assert not runtime._tracking_segment_clear(obs['position'][:2], runtime._lookahead_target(np.array(obs['position'][:2])))
    with patch.object(runtime, 'action_for', return_value={'move_along_path': [path]}):
        command = runtime.velocity_action_for(decision, obs, step=0)['move_by_speed']
    assert command[1] > 0 and abs(command[0]) < 1e-9
    assert runtime.replan_count == 0
