"""Deployment lifecycle around observed-map planning; no robot communication."""
from dataclasses import replace
from types import SimpleNamespace
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
def navigation(tmp_path, monkeypatch):
    from deploy.app import controller as module
    config = load()
    snapshot = next(snapshots(make_fixture(tmp_path)))
    tasks, motion, events = Tasks(config['navigation']), TrackedMotion(), Events()
    controller = NavigationController(config, tasks, motion, events)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: snapshot.captured_at))
    assert controller.step(snapshot, snapshot.captured_at) is None
    assert tasks.map_ready
    return controller, snapshot, events


def occupy(controller, goal):
    engine = controller.tasks.engine
    cell = engine.planner.cell(goal)
    engine.layers.observed[cell] = True
    engine.layers.evidence.log_odds[cell] = 4.
    engine.layers._latest_static[cell] = True
    engine.layers.refresh_global(reason='test_measured_obstacle')
    engine.layers._bind_local()


@pytest.mark.parametrize('case,reason', [('not_ready', '地图'), ('outside', '范围'), ('occupied', '障碍')])
def test_invalid_endpoint_rejected_before_arm(navigation, capsys, case, reason):
    controller, snapshot, events = navigation
    goal = (1., 0.)
    if case == 'not_ready':
        controller.tasks.map_ready = False
    elif case == 'outside':
        goal = (20., 0.)
    else:
        occupy(controller, goal)
    controller.operate('navigate', f'{goal[0]},{goal[1]}', snapshot, snapshot.captured_at)
    assert controller.motion.arm_calls == 0 and not controller.motion.commands
    assert reason in controller.task_error and reason in capsys.readouterr().out
    assert any(e['event'] == 'task_rejected' for e in events.items)


@pytest.mark.parametrize('connected_frontier', [False, True])
def test_unknown_goal_is_accepted_without_revealing_unknown_map(navigation, connected_frontier):
    controller, snapshot, events = navigation
    if connected_frontier:
        # Floor returns reveal a free prefix with an unobserved continuation,
        # unlike the fixture's enclosing obstacle ring.
        snapshot.observation['sensors']['lidar']['pointcloud'][:, 2] = 0.
        controller.tasks = Tasks(load()['navigation'])
        controller.step(snapshot, snapshot.captured_at)
    engine = controller.tasks.engine
    goal = (8., 0.)
    cell = engine.planner.cell(goal)
    assert not engine.layers.observed[cell]
    controller.operate('navigate', '8,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    assert controller.tasks.state == ('active' if connected_frontier else 'paused')
    assert controller.motion.arm_calls == int(connected_frontier)
    assert engine.decision['kind'] == ('explore' if connected_frontier else 'wait')
    assert engine.goal[:2] == goal
    assert not any(e['event'] == 'task_rejected' for e in events.items)
    assert not engine.layers.observed[cell] and not engine.layers.global_observed[cell]
    assert not engine.planner.interactable_ids


def test_new_obstacle_at_goal_stops_before_next_command(navigation):
    controller, snapshot, events = navigation
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    sent = len(controller.motion.commands)
    # Actual new measured hits remain local obstacles even before slow global refresh.
    points = np.array([[x, y, .35] for x in (1.01, 1.11, 1.21) for y in (.01, .11, .21)])
    lidar = dict(snapshot.observation['sensors']['lidar'], pointcloud=points, physics_step=1)
    current = replace(snapshot, observation={**snapshot.observation, 'sensors': {'lidar': lidar}},
                      captured_at=snapshot.captured_at+.2, pose_at=snapshot.pose_at+.2,
                      lidar_at=snapshot.lidar_at+.2, sequence=1)
    error = controller.step(current, current.captured_at)
    assert controller.tasks.last_lidar == 1
    assert error and '障碍' in error
    assert controller.tasks.state == 'paused'
    assert len(controller.motion.commands) == sent and len(controller.motion.stops) == 1
    assert not any(e['event'] == 'task_result' and e['result'].get('success') for e in events.items)


def test_resume_rechecks_obstacle_and_keeps_map(navigation):
    controller, snapshot, _ = navigation
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    layers = controller.tasks.engine.layers
    controller.operate('p', None, snapshot, snapshot.captured_at)
    occupy(controller, (1., 0.))
    controller.operate('r', None, snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'paused' and controller.motion.arm_calls == 1
    assert '障碍' in controller.task_error
    assert controller.tasks.engine.layers is layers
    assert controller.tasks.engine.skill is None


@pytest.mark.parametrize('operation', ['cancel', 'new_goal'])
def test_explicit_cancel_or_valid_goal_clears_rejection(navigation, operation):
    controller, snapshot, _ = navigation
    controller.operate('navigate', '20,0', snapshot, snapshot.captured_at)
    assert controller.task_error and controller.motion.arm_calls == 0
    controller.operate('m' if operation == 'cancel' else 'navigate', None if operation == 'cancel' else '1,0',
                       snapshot, snapshot.captured_at)
    assert controller.task_error is None
