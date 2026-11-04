"""Reject unsafe endpoints and expose real planner failure without robot I/O."""
from dataclasses import replace

import numpy as np
import pytest

from deploy.app.config import load
from deploy.app.controller import NavigationController
from deploy.app.tasks import Tasks
from deploy.recording.replay import snapshots
from deploy.tests.make_fixture import make_fixture
from deploy.tests.test_controller import Events, Motion


class TrackedMotion(Motion):
    def __init__(self):
        super().__init__()
        self.arm_calls = 0

    def arm(self):
        self.arm_calls += 1
        return super().arm()


@pytest.fixture
def navigation(tmp_path):
    config = load()
    snapshot = next(snapshots(make_fixture(tmp_path)))
    tasks, motion, events = Tasks(config['navigation']), TrackedMotion(), Events()
    controller = NavigationController(config, tasks, motion, events)
    assert controller.step(snapshot, snapshot.captured_at) is None
    assert tasks.map_ready
    return controller, snapshot, events


def grid_for(controller):
    return controller.tasks.component.mapping.map.occupancy


def test_point_tracking_and_display_do_not_build_topology(navigation, monkeypatch):
    from deploy.visualization.scene import SceneCapture
    controller, snapshot, _ = navigation
    graph = controller.tasks.component.mapping.semantic_voronoi
    assert graph.peek_snapshot() is None
    def unexpected(*args, **kwargs):
        raise AssertionError('Unnecessary topology build on the control/display path')
    monkeypatch.setattr(graph, 'update', unexpected)
    grid_for(controller).observed[:] = True
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    for elapsed in (0., 1.01, 2.01):
        now = snapshot.captured_at+elapsed
        current = replace(snapshot, captured_at=now, pose_at=now, lidar_at=now)
        assert controller.step(current, now) is None
        scene = SceneCapture(include_cloud=False).capture(current, controller.tasks)
        assert scene['topology_revision'] is None and scene['skeleton'] == []
        stats = controller.tasks.component.mapping.statistics()
        assert stats['semantic_voronoi']['builds'] == 0
    assert controller.motion.commands and controller.tasks.state == 'active'


def test_deferred_topology_is_fresh_when_a_detour_is_needed(navigation, monkeypatch):
    controller, _, _ = navigation
    runtime = controller.tasks.component.mapping
    grid = grid_for(controller)
    grid.observed[:] = True
    grid.mark_box_occupied((.8, -.2), (1.1, .2))
    graph = runtime.semantic_voronoi
    plan = runtime.voronoi_planner.plan
    revisions = []
    def check_fresh(start, goal):
        assert graph.peek_snapshot() is not None
        assert graph.peek_snapshot().revision == grid.revision
        revisions.append(grid.revision)
        return plan(start, goal)
    monkeypatch.setattr(runtime.voronoi_planner, 'plan', check_fresh)
    runtime._map_plan((0., 0., .3), (2., 0., .3))
    grid.mark_box_occupied((.8, .6), (1.1, .8))
    runtime._map_plan((0., 0., .3), (2., 0., .3))
    assert len(revisions) == 2 and revisions[1] > revisions[0]


@pytest.mark.parametrize('axis,speeds', [(0, (.80, .60, .49, .31)), (1, (.30, .30, .30, .30))])
def test_navigation_speed_profile_reaches_success_then_stops(navigation, axis, speeds):
    controller, snapshot, events = navigation
    grid_for(controller).observed[:] = True
    controller.operate('navigate', '1,0' if axis == 0 else '0,1', snapshot, snapshot.captured_at)
    for index, (distance, speed) in enumerate(zip((1., .60, .49, .31), speeds)):
        now = snapshot.captured_at+.1*(index+1)
        position = [0., 0., .3]
        position[axis] = 1.-distance
        current = replace(snapshot, observation={**snapshot.observation, 'position': position},
                          captured_at=now, pose_at=now, lidar_at=now)
        assert controller.step(current, now) is None
        # The mock records planner requests; the independent motion guard
        # applies the configured yaw cap before SDK transmission.
        expected = [0., 0., 0. if axis == 0 else 1.2]
        expected[axis] = speed
        np.testing.assert_allclose(controller.motion.commands[-1][0], expected, atol=1e-9)
        assert controller.tasks.state == 'active'
    commands = len(controller.motion.commands)
    now += .1
    position[axis] = .71
    reached = replace(current, observation={**current.observation, 'position': position},
                      captured_at=now, pose_at=now, lidar_at=now)
    controller.step(reached, now)
    assert controller.tasks.result['success']
    assert len(controller.motion.commands) == commands
    assert controller.motion.stops
    # Original shared/simulation defaults must remain unchanged.
    from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent, PointNavigationConfig
    simulation = PointNavigationComponent(PointNavigationConfig(goal=(1., 0., .3)))
    assert simulation.mapping.final_goal_slow_radius == .5


@pytest.mark.parametrize('case,reason', [
    ('not_ready', '地图'), ('outside', '范围'), ('unknown', '未观测'),
    ('occupied', '障碍'), ('inflated', '膨胀'),
])
def test_invalid_endpoint_rejected_before_arm_and_reason_remains_visible(navigation, capsys, case, reason):
    controller, snapshot, events = navigation
    grid = grid_for(controller)
    goal = (1., 0.)
    if case == 'not_ready':
        controller.tasks.map_ready = False
    elif case == 'outside':
        goal = (20., 0.)
    elif case == 'unknown':
        grid.observed[grid.world_to_cell(goal)] = False
    elif case == 'occupied':
        grid.mark_box_occupied(goal, goal)
    else:
        grid.mark_box_occupied((1.05, -.25), (1.05, -.25))
        cell = grid.world_to_cell(goal)
        assert not grid.occupied_mask()[cell] and grid.inflated_mask()[cell]

    controller.operate('navigate', f'{goal[0]},{goal[1]}', snapshot, snapshot.captured_at)

    assert controller.motion.arm_calls == 0
    assert not controller.motion.commands
    assert controller.tasks.state != 'active'
    assert reason in controller.task_error
    assert reason in capsys.readouterr().out
    rejection = [e for e in events.items if e['event'] == 'task_rejected']
    assert len(rejection) == 1 and reason in rejection[0]['reason']
    # A healthy next input must not immediately erase the operator's rejection.
    assert controller.step(snapshot, snapshot.captured_at) == controller.task_error


def test_disconnected_free_goal_uses_original_planner_then_stops_once(navigation):
    controller, snapshot, events = navigation
    grid = grid_for(controller)
    grid.mark_box_occupied((.5, -10.), (.5, 10.))
    for xy in ((0., 0.), (1., 0.)):
        cell = grid.world_to_cell(xy)
        assert grid.observed[cell] and not grid.inflated_mask()[cell]
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.motion.arm_calls == 1 and controller.tasks.state == 'active'

    reason = controller.step(snapshot, snapshot.captured_at)

    runtime = controller.tasks.component.mapping
    assert runtime.planning_failures == 1  # Real A*/Voronoi failure, not a mocked zero command.
    assert reason and '规划' in reason
    assert controller.tasks.state == 'paused'
    assert not controller.motion.commands
    assert len(controller.motion.stops) == 1
    assert controller.tasks.result is None
    assert len([e for e in events.items if e['event'] == 'task_paused']) == 1
    controller.step(snapshot, snapshot.captured_at)
    assert len(controller.motion.stops) == 1 and runtime.planning_failures == 1
    assert controller.step(snapshot, snapshot.captured_at) == reason


def test_new_scan_blocks_goal_before_next_velocity_is_submitted(navigation):
    controller, snapshot, events = navigation
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    assert controller.motion.commands
    previous_commands = len(controller.motion.commands)
    # A new, connected obstacle cluster at the target; it must be fused before
    # checking whether this tick may request movement.
    points = np.array([[x, y, .35] for x in (1.01, 1.11, 1.21)
                       for y in (.01, .11, .21)], dtype=np.float32)
    lidar = dict(snapshot.observation['sensors']['lidar'], pointcloud=points, physics_step=1)
    updated = replace(snapshot, observation={**snapshot.observation, 'sensors': {'lidar': lidar}},
                      captured_at=snapshot.captured_at+.1, pose_at=snapshot.pose_at+.1,
                      lidar_at=snapshot.lidar_at+.1, sequence=1)

    error = controller.step(updated, updated.captured_at)

    assert grid_for(controller).occupied_mask()[grid_for(controller).world_to_cell((1., 0.))]
    assert controller.tasks.last_lidar == 1
    assert error and '障碍' in error
    assert controller.tasks.state == 'paused'
    assert len(controller.motion.commands) == previous_commands
    assert len(controller.motion.stops) == 1
    assert not any(e['event'] == 'task_result' and e['result'].get('success') for e in events.items)


def test_resume_rechecks_goal_and_requires_explicit_action_after_map_recovers(navigation):
    controller, snapshot, _ = navigation
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    grid = grid_for(controller)
    grid.observed[grid.world_to_cell((1., 0.))] = False
    reason = controller.step(snapshot, snapshot.captured_at)
    assert reason and controller.tasks.state == 'paused'
    commands = len(controller.motion.commands)

    controller.operate('r', None, snapshot, snapshot.captured_at)
    assert controller.motion.arm_calls == 1
    assert controller.tasks.state == 'paused' and controller.task_error
    grid.mark_free((1., 0.), radius=.1)
    controller.step(snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'paused' and len(controller.motion.commands) == commands

    controller.motion_events([{'state': 'stopped', 'detail': 'test stop feedback'}])
    controller.operate('r', None, snapshot, snapshot.captured_at)
    assert controller.motion.arm_calls == 2 and controller.tasks.state == 'active'
    assert controller.task_error is None
    assert controller.step(snapshot, snapshot.captured_at) is None
    assert len(controller.motion.commands) > commands


def test_historical_planning_error_does_not_reject_a_successful_new_plan(navigation):
    controller, snapshot, _ = navigation
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    runtime = controller.tasks.component.mapping
    runtime.planning_failures = 4
    runtime.last_planning_error = 'previous unrelated no map path to goal'

    assert controller.step(snapshot, snapshot.captured_at) is None
    assert runtime.planning_failures == 4
    assert controller.tasks.state == 'active' and controller.motion.commands
    assert not controller.motion.stops


def test_protective_zero_during_buffer_recovery_is_not_a_planning_failure(navigation):
    controller, snapshot, _ = navigation
    grid = grid_for(controller)
    grid.mark_box_occupied((0., 0.), (0., 0.))
    # The goal is free; rejecting the start here would bypass buffer recovery.
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.motion.arm_calls == 1

    assert controller.step(snapshot, snapshot.captured_at) is None

    runtime = controller.tasks.component.mapping
    assert runtime.buffer_recovery_active and runtime.planning_failures == 0
    assert controller.tasks.state == 'active'
    np.testing.assert_array_equal(controller.motion.commands[-1][0], np.zeros(3))
    assert not controller.motion.stops


@pytest.mark.parametrize('operation', ['cancel', 'new_goal'])
def test_explicit_cancel_or_valid_new_goal_clears_rejection(navigation, operation):
    controller, snapshot, _ = navigation
    grid = grid_for(controller)
    grid.observed[grid.world_to_cell((2., 0.))] = False
    controller.operate('navigate', '2,0', snapshot, snapshot.captured_at)
    assert controller.task_error and controller.motion.arm_calls == 0
    if operation == 'cancel':
        controller.operate('m', None, snapshot, snapshot.captured_at)
        assert controller.tasks.state == controller.tasks.mode == 'observe'
    else:
        controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
        assert controller.tasks.state == 'active' and controller.motion.arm_calls == 1
    assert controller.task_error is None
    assert controller.step(snapshot, snapshot.captured_at) is None
