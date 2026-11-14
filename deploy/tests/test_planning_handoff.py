"""Real navigation/task transitions with only clock and robot transport faked."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from deploy.app.config import load
from deploy.app.controller import NavigationController
from deploy.app.tasks import Tasks
from deploy.recording.replay import snapshots
from deploy.tests.make_fixture import make_fixture


class Motion:
    def __init__(self, clock):
        self.clock = clock
        self.state = 'manual'
        self.arm_count = 0
        self.commands, self.stops, self.pending = [], [], []

    def arm(self):
        self.arm_count += 1
        self.state = 'active'
        self.pending.append({'state': 'active', 'detail': '', 'worker_at': self.clock[0]})
        return True

    def stop(self, reason):
        self.state = 'stopping'
        self.stops.append(reason)

    def events(self):
        result, self.pending = self.pending, []
        return result

    def submit(self, velocity, generated_at, expires_at):
        self.commands.append((velocity, generated_at, expires_at))
        return len(self.commands)


class Session:
    def __init__(self):
        self.items = []

    def event(self, kind, **details):
        self.items.append({'event': kind, **details})


@pytest.fixture
def run(tmp_path, monkeypatch):
    import deploy.app.controller as module
    config = load()
    original = next(snapshots(make_fixture(tmp_path)))
    clock = [original.captured_at]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    tasks, session, motion = Tasks(config['navigation']), Session(), Motion(clock)
    controller = NavigationController(config, tasks, motion, session)

    def sample(advance=.01, *, pose_at=None, lidar_at=None):
        clock[0] += advance
        return replace(original, captured_at=clock[0],
                       pose_at=clock[0] if pose_at is None else pose_at,
                       lidar_at=clock[0] if lidar_at is None else lidar_at)

    def feedback(state, reason):
        motion.state = state
        controller.motion_events([{'state': state, 'detail': reason, 'worker_at': clock[0]}])

    def hold():
        # A changed route or completed recovery legitimately returns here.
        tasks.engine.phase = 'decide'
        tasks.engine.tracker = None
        current = sample()
        controller.step(current, clock[0])
        assert controller.planning_hold is not None
        return controller.planning_hold['token']

    controller.step(original, clock[0])
    controller.operate('navigate', '1,0', original, clock[0])
    assert tasks.state == 'active' and motion.arm_count == 1
    return SimpleNamespace(controller=controller, tasks=tasks, motion=motion,
                           session=session, clock=clock, sample=sample,
                           feedback=feedback, hold=hold)


def count_calls(monkeypatch, owner, name, clock=None, delay=0.):
    original = getattr(owner, name)
    calls = []

    def call(*args, **kwargs):
        calls.append((args, kwargs))
        if clock is not None:
            clock[0] += delay
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, call)
    return calls


def prepare_hold(run):
    token = run.hold()
    run.sample(.1)
    run.feedback('stopped', token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.controller.planning_hold['prepared_at'] is not None
    return token


def test_active_global_decision_yields_before_selector_or_route(run, monkeypatch):
    import deploy.app.navigation as navigation
    buffer_queries = count_calls(monkeypatch, navigation, 'decision_buffer_recovery')
    decisions = count_calls(monkeypatch, run.tasks.engine.selector, 'decide')
    routes = count_calls(monkeypatch, run.tasks.engine.routes, 'plan')
    token = run.hold()
    assert token.endswith(':decision_buffer_recovery')
    assert not buffer_queries and not decisions and not routes
    assert run.motion.stops == [token] and run.motion.state == 'stopping'
    assert run.tasks.state == 'active' and not run.motion.commands


def test_only_matching_stop_and_new_pose_and_lidar_allow_planning(run, monkeypatch):
    decisions = count_calls(monkeypatch, run.tasks.engine.selector, 'decide')
    token = run.hold()
    run.feedback('stopped', 'planning_hold:old:global_decision')
    run.controller.step(run.sample(), run.clock[0])
    assert run.controller.planning_hold['stopped_at'] is None and not decisions

    run.feedback('stopped', token)
    stopped_at = run.clock[0]
    for stamps in ({'pose_at': stopped_at}, {'lidar_at': stopped_at}):
        current = run.sample(**stamps)
        run.controller.step(current, run.clock[0])
        assert not decisions and run.motion.arm_count == 1
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert len(decisions) == 1
    assert run.controller.planning_hold['prepared_at'] is not None
    assert run.tasks.state == 'paused' and not run.motion.commands


def test_slow_plan_is_discarded_until_fresh_revalidation_then_same_goal_resumes(run, monkeypatch):
    decisions = count_calls(monkeypatch, run.tasks.engine.selector, 'decide', run.clock, .7)
    goal = run.tasks.engine.goal
    token = prepare_hold(run)
    prepared_at = run.controller.planning_hold['prepared_at']
    assert len(decisions) == 1 and run.motion.arm_count == 1 and not run.motion.commands
    for stamps in ({'pose_at': prepared_at}, {'lidar_at': prepared_at}):
        current = run.sample(**stamps)
        run.controller.step(current, run.clock[0])
        assert run.motion.arm_count == 1 and not run.motion.commands

    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 2 and run.controller.planning_hold is None
    assert run.tasks.state == 'active' and run.tasks.engine.goal == goal
    assert not run.motion.commands
    assert any(e['event'] == 'planning_hold_resumed' and e['hold_id'] == token for e in run.session.items)
    # The arm acknowledgement cannot reuse the revalidation observation.
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert len(run.motion.commands) == 1
    assert run.motion.commands[0][1] == current.captured_at


def test_slow_revalidation_also_stays_disarmed_and_requires_another_frame(run, monkeypatch):
    prepare_hold(run)
    prepared_at = run.controller.planning_hold['prepared_at']
    with monkeypatch.context() as patch:
        count_calls(patch, run.tasks.engine, 'update', run.clock, .2)
        current = run.sample()
        run.controller.step(current, run.clock[0])
    assert run.controller.planning_hold['prepared_at'] > prepared_at
    assert run.motion.arm_count == 1 and not run.motion.commands
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 2 and not run.motion.commands


@pytest.mark.parametrize('reason', ['command_timeout', 'remote_takeover', 'robot_state_stale_or_unsafe'])
@pytest.mark.parametrize('prepared', [False, True])
def test_real_stop_invalidates_hold_and_never_autoarms(run, reason, prepared):
    token = prepare_hold(run) if prepared else run.hold()
    run.feedback('stopping', reason)
    run.feedback('stopped', reason)
    # A delayed confirmation from the abandoned hold must not revive it.
    run.feedback('stopped', token)
    for _ in range(2):
        current = run.sample()
        run.controller.step(current, run.clock[0])
    assert run.controller.planning_hold is None
    assert run.tasks.state == 'paused' and run.motion.arm_count == 1
    assert not run.motion.commands


@pytest.mark.parametrize('operation', ['p', 'stop'])
@pytest.mark.parametrize('prepared', [False, True])
def test_user_pause_or_stop_cancels_resume_permission(run, operation, prepared):
    token = prepare_hold(run) if prepared else run.hold()
    current = run.sample()
    run.controller.operate(operation, None, current, run.clock[0])
    run.feedback('stopped', token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.controller.planning_hold is None and run.motion.arm_count == 1
    assert run.tasks.state == ('paused' if operation == 'p' else 'observe')
    assert not run.motion.commands


def test_new_goal_during_hold_replaces_permission_and_requires_its_own_stop(run):
    old_token = run.hold()
    current = run.sample()
    run.controller.operate('navigate', '.8,0', current, run.clock[0])
    new_token = run.controller.planning_hold['token']
    assert new_token != old_token and run.tasks.engine.goal[:2] == (1., 0.)
    run.feedback('stopped', old_token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 1 and run.tasks.engine.goal[:2] == (1., 0.)
    run.feedback('stopped', new_token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 2 and run.tasks.engine.goal[:2] == (.8, 0.)
    assert run.controller.planning_hold is None and not run.motion.commands


def test_replacing_active_goal_waits_for_stop_before_new_plan(run, monkeypatch):
    starts = count_calls(monkeypatch, run.tasks, 'start')
    current = run.sample()
    run.controller.operate('navigate', '.8,0', current, run.clock[0])
    token = run.controller.planning_hold['token']
    assert token.endswith(':operator_goal_change') and not starts
    assert run.tasks.state == 'paused' and run.motion.arm_count == 1
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert not starts
    run.feedback('stopped', token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert len(starts) == 1 and run.tasks.engine.goal[:2] == (.8, 0.)
    assert run.motion.arm_count == 2 and not run.motion.commands
