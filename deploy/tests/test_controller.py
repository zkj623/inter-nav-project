"""Task/control transitions without ROS, SDK, or physical motion."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from deploy.app.config import load, check_control_configuration, check_motion_configuration
from deploy.app.controller import NavigationController
from deploy.app.tasks import Tasks
from deploy.recording.replay import snapshots
from deploy.tests.make_fixture import make_fixture


class Motion:
    def __init__(self, accepted=True):
        self.state = 'manual'
        self.accepted = accepted
        self.commands, self.stops = [], []
        self.pending = []

    def arm(self):
        self.state = 'active' if self.accepted else 'rejected'
        self.pending.append({'state': self.state, 'detail': '', 'accepted': self.accepted})
        return self.accepted

    def events(self):
        events, self.pending = self.pending, []
        return events

    def submit(self, velocity, generated, expires):
        self.commands.append((velocity, generated, expires))

    def stop(self, reason):
        self.state = 'stopping'
        self.stops.append(reason)


class Events:
    def __init__(self):
        self.items = []

    def event(self, kind, **data):
        self.items.append({'event': kind, **data})


@pytest.fixture
def control(tmp_path, monkeypatch):
    from deploy.app import controller as module
    config = load()
    snapshot = next(snapshots(make_fixture(tmp_path)))
    tasks, motion, events = Tasks(config['navigation']), Motion(), Events()
    controller = NavigationController(config, tasks, motion, events)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: snapshot.captured_at))
    controller.step(snapshot, snapshot.captured_at)  # Observe/map before entering a goal, as in the console.
    return controller, snapshot, events


def test_observe_never_arms_and_cancel_clears_task_without_losing_map(control):
    controller, snapshot, events = control
    controller.step(snapshot, snapshot.captured_at)
    retained = controller.tasks.engine.layers
    assert controller.motion.state == 'manual' and not controller.motion.commands
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'active'
    assert any(item.get('state') == 'active' for item in events.items)
    controller.operate('m', None, snapshot, snapshot.captured_at)
    assert controller.tasks.state == controller.tasks.mode == 'observe'
    assert controller.tasks.result is None
    assert controller.tasks.engine.layers is retained
    assert controller.motion.stops == ['m']


def test_arm_rejection_keeps_task_paused_and_never_submits(control):
    controller, snapshot, events = control
    controller.motion.accepted = False
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'paused'
    assert not controller.motion.commands
    assert any(item['event'] == 'task_rejected' for item in events.items)


@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_stale_observation_stops_once_and_requires_explicit_resume(control, missing):
    controller, snapshot, events = control
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(None if missing else snapshot, snapshot.captured_at+1.)
    assert controller.tasks.state == 'paused'
    assert len(controller.motion.stops) == 1
    controller.step(snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'paused' and not controller.motion.commands
    assert len(controller.motion.stops) == 1


def test_completion_is_recorded_once_even_if_motion_state_has_advanced(control):
    controller, snapshot, events = control
    controller.operate('navigate', '0,0', snapshot, snapshot.captured_at)
    controller.motion.state = 'stopping'
    controller.step(snapshot, snapshot.captured_at)
    stopped = replace(snapshot, captured_at=snapshot.captured_at+.5, pose_at=snapshot.pose_at+.5, lidar_at=snapshot.lidar_at+.5)
    controller.step(stopped, stopped.captured_at)
    controller.step(stopped, stopped.captured_at)
    assert controller.tasks.state == 'completed'
    assert controller.tasks.result['success']
    assert controller.motion.stops == ['task_completed']
    assert len([e for e in events.items if e['event'] == 'task_result']) == 1
    assert all(not command[0].any() for command in controller.motion.commands)


def test_worker_stop_feedback_pauses_task_and_does_not_auto_resume(control):
    controller, snapshot, events = control
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.motion_events([{'state': 'stopping', 'detail': 'remote_takeover'},
                              {'state': 'stopped', 'detail': 'fresh stationary feedback'}])
    assert controller.tasks.state == 'paused' and controller.stop_confirmed is True
    controller.step(snapshot, snapshot.captured_at)
    assert not controller.motion.commands


def test_command_expires_with_oldest_required_observation(control):
    controller, snapshot, _ = control
    snapshot = replace(snapshot, pose_at=snapshot.captured_at-.2, lidar_at=snapshot.captured_at-.3)
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    assert controller.motion.commands
    _, generated, expires = controller.motion.commands[-1]
    assert generated == snapshot.captured_at
    assert expires == pytest.approx(snapshot.lidar_at+controller.robot['max_data_age_s'])


@pytest.mark.parametrize('decision_duration,sensor_age', [(.25, 0.), (.35, 0.), (.45, 0.), (.15, .30)])
def test_decision_timing_does_not_extend_command_or_sensor_expiry(
        control, monkeypatch, decision_duration, sensor_age):
    from deploy.app import controller as module
    controller, snapshot, events = control
    snapshot = replace(snapshot, pose_at=snapshot.captured_at-sensor_age,
                       lidar_at=snapshot.captured_at-sensor_age)
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    submitted_at = snapshot.captured_at + decision_duration + .01
    ticks = iter([snapshot.captured_at, snapshot.captured_at + decision_duration,
                  submitted_at, submitted_at])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    controller.step(snapshot, snapshot.captured_at)
    expired = submitted_at >= min(snapshot.captured_at+controller.control['command_ttl_s'],
                                  snapshot.lidar_at+controller.robot['max_data_age_s'])
    kind = 'command_discarded' if expired else 'command_requested'
    command = next(e for e in events.items if e['event'] == kind)
    assert command['decision_duration_s'] == pytest.approx(decision_duration)
    stages = command['decision_stages_s']
    assert stages['navigation_s'] >= 0
    assert stages['total_s'] >= sum(v for k, v in stages.items() if k != 'total_s')
    assert command['generated_at'] == snapshot.captured_at
    assert command['expires_at'] <= snapshot.captured_at + controller.control['command_ttl_s']
    assert command['expires_at'] <= snapshot.lidar_at + controller.robot['max_data_age_s']
    assert command['remaining_valid_s'] == pytest.approx(command['expires_at']-submitted_at)
    if expired:
        assert command['discarded_at'] == submitted_at and command['remaining_valid_s'] < 0
        assert command['pose_age_at_discard'] == pytest.approx(submitted_at-snapshot.pose_at)
        assert command['lidar_age_at_discard'] == pytest.approx(submitted_at-snapshot.lidar_at)
        assert command['delivery'] == 'not_queued'
        assert not controller.motion.commands and controller.tasks.state == 'paused'
        assert controller.motion.stops == ['decision_expired']
    else:
        assert command['submitted_at'] == submitted_at
        assert command['pose_age_at_submit'] == pytest.approx(submitted_at-snapshot.pose_at)
        assert command['lidar_age_at_submit'] == pytest.approx(submitted_at-snapshot.lidar_at)
        assert command['delivery'] == 'queued_not_execution_ack'


def test_cancel_navigation_discards_pending_recognition_and_goal():
    resets = []
    perception = SimpleNamespace(reset=lambda: resets.append(True))
    tasks = Tasks(load()['navigation'], perception)
    tasks.start('navigate', (1., 0., .3), 10.)
    tasks.cancel()
    assert len(resets) == 2
    assert tasks.engine.goal is None
    assert tasks.mode == tasks.state == 'observe'


def test_mapping_failure_does_not_report_ready(control, monkeypatch):
    controller, snapshot, _ = control
    controller.tasks = Tasks(load()['navigation'])
    def fail(*args, **kwargs):
        raise ValueError('bad input')
    update = controller.tasks.engine.observe
    monkeypatch.setattr(controller.tasks.engine, 'observe', fail)
    assert controller.step(snapshot, snapshot.captured_at) == 'bad input'
    assert not controller.tasks.map_ready
    assert controller.tasks.last_lidar is None
    assert controller.tasks.last_timings['mapping_s'] >= 0
    assert controller.tasks.last_timings['total_s'] >= controller.tasks.last_timings['mapping_s']
    monkeypatch.setattr(controller.tasks.engine, 'observe', update)
    assert controller.step(snapshot, snapshot.captured_at) is None
    assert controller.tasks.map_ready
    assert controller.tasks.last_mapping_at == snapshot.lidar_at


def test_sdk_configuration_and_full_navigation_acceptance_are_distinct():
    config = load()
    config['robot']['deskew_and_self_filter_verified'] = False
    config['robot']['takeover_button_mask'] = 0
    check_control_configuration(config, require_takeover=False)
    with pytest.raises(ValueError, match='takeover_button_mask'):
        check_control_configuration(config)
    config['robot']['takeover_button_mask'] = 2
    check_control_configuration(config)
    with pytest.raises(ValueError, match='deskew_and_self_filter_verified'):
        check_motion_configuration(config)


def test_worker_stop_reason_is_visible_in_console(control, capsys):
    controller, _, events = control
    controller.motion_events([{'state': 'stopping', 'detail': 'command_timeout'},
                              {'state': 'stopped', 'detail': 'command_timeout'}])
    output = capsys.readouterr().out
    assert '运动状态=stopping 原因=command_timeout' in output
    assert '运动状态=stopped 原因=command_timeout' in output
    assert events.items[-1]['detail'] == 'command_timeout'


def test_arm_rejection_includes_worker_reason(control, capsys):
    controller, snapshot, events = control
    def reject():
        controller.motion.state = 'rejected'
        controller.motion.pending.append({'state': 'rejected',
                                         'detail': 'Need fresh telemetry and a stationary robot'})
        return False
    controller.motion.arm = reject
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'paused'
    assert not controller.motion.commands
    assert '原因=Need fresh telemetry and a stationary robot' in capsys.readouterr().out
    assert 'Need fresh telemetry' in events.items[-1]['reason']


def test_live_cleanup_restores_other_resources_even_if_motion_close_raises(monkeypatch):
    import signal
    from deploy.app import runtime, session, keyboard
    from deploy.robot import sensors, motion
    closed = []
    events = Events()
    events.info = {'session_id': 'fake'}
    events.url = 'fake-session'
    events.frames = SimpleNamespace(error=lambda *args: None)
    events.rgb = lambda *args: None
    def finish(summary):
        closed.append('session')
        events.summary = summary
    events.close = finish
    def fail_close():
        closed.append('motion')
        raise RuntimeError('simulated worker failure')
    fake_motion = Motion()
    fake_motion.events = lambda: []
    fake_motion.close = fail_close
    monkeypatch.setattr(session, 'Session', lambda *a, **k: events)
    monkeypatch.setattr(motion, 'MotionProcess', lambda *a: fake_motion)
    monkeypatch.setattr(sensors, 'RosSensors', lambda *a: SimpleNamespace(
        latest=lambda: None, close=lambda: closed.append('sensors')))
    monkeypatch.setattr(keyboard, 'Keyboard', lambda: SimpleNamespace(
        poll=lambda: ('q', None), close=lambda: closed.append('keyboard')))
    config = load()
    for key in ('calibration_verified', 'deskew_and_self_filter_verified',
                'communication_verified', 'remote_arbitration_verified'):
        config['robot'][key] = True  # Only this in-memory fake-SDK test.
    config['robot']['takeover_button_mask'] = 2
    args = SimpleNamespace(allow_motion=True, demo=False, session_port=0, camera='none', rgb_topic=None)
    handler = signal.getsignal(signal.SIGINT)
    assert runtime.run_live(args, config) == 2
    assert closed == ['motion', 'keyboard', 'sensors', 'session']
    assert signal.getsignal(signal.SIGINT) == handler
    assert 'physical stop NOT confirmed' in events.summary['error']
    assert events.summary['last_autonomous_stop_confirmed'] is None


def test_old_stop_received_during_arm_cannot_pause_the_new_task(control):
    controller, snapshot, events = control
    controller.motion.pending = [{'state': 'stopping', 'detail': 'earlier stop'},
                                 {'state': 'stopped', 'detail': 'earlier stop'}]
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.motion_events(controller.motion.events())  # Next live-loop iteration.
    controller.step(snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'active'
    assert controller.motion.commands
    assert controller.stop_confirmed is False
    assert [e['state'] for e in events.items if e['event'] == 'motion_state'] == ['stopping', 'stopped', 'active']


def test_immediate_worker_fault_after_arm_does_not_start_task(control):
    controller, snapshot, _ = control
    arm = controller.motion.arm
    def fails_after_ack():
        arm()
        controller.motion.state = 'fault'
        controller.motion.pending.append({'state': 'fault', 'detail': 'telemetry lost'})
        return True
    controller.motion.arm = fails_after_ack
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'paused'
    assert not controller.motion.commands
    assert controller.motion_error == 'telemetry lost'


@pytest.mark.parametrize('fresh_after_arm', [True, False])
def test_live_start_reacquires_sensors_after_slow_arm(tmp_path, monkeypatch, fresh_after_arm):
    """A slow acknowledgement must neither reuse nor re-date pre-arm sensor data."""
    from deploy.app import controller as controller_module, runtime, session, keyboard
    from deploy.robot import sensors, motion

    snapshot = next(snapshots(make_fixture(tmp_path)))
    clock = [snapshot.captured_at]
    fake_clock = SimpleNamespace(monotonic=lambda: clock[0],
                                 sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(runtime, 'time', fake_clock)
    monkeypatch.setattr(controller_module, 'time', fake_clock)
    config = load()
    for key in ('calibration_verified', 'deskew_and_self_filter_verified',
                'communication_verified', 'remote_arbitration_verified'):
        config['robot'][key] = True  # In-memory fake motion only.
    config['robot']['takeover_button_mask'] = 2
    events = Events()
    events.info, events.url = {'session_id': 'fake'}, 'fake-session'
    events.frames = SimpleNamespace(error=lambda *args: None)
    events.rgb = lambda *args: None
    events.update = lambda *args, **kwargs: None
    events.close = lambda summary: None
    fake_motion = Motion()
    fake_motion.close = lambda: []
    original_arm = fake_motion.arm
    pre_arm_stamp = []
    def slow_arm():
        pre_arm_stamp.append(clock[0]-.05)
        clock[0] += .5
        return original_arm()
    fake_motion.arm = slow_arm
    reads = []
    def latest():
        stamp = (pre_arm_stamp[0] if pre_arm_stamp and not fresh_after_arm else clock[0]-.05)
        reads.append(stamp)
        return replace(snapshot, captured_at=clock[0], pose_at=stamp, lidar_at=stamp,
                       planning_frame='odom')
    fake_sensors = SimpleNamespace(latest=latest, close=lambda: None, last_error=None)
    operations = iter([None, ('navigate', '1,0'), None, ('q', None)])
    monkeypatch.setattr(session, 'Session', lambda *args, **kwargs: events)
    monkeypatch.setattr(motion, 'MotionProcess', lambda *args: fake_motion)
    monkeypatch.setattr(sensors, 'RosSensors', lambda *args: fake_sensors)
    monkeypatch.setattr(keyboard, 'Keyboard', lambda: SimpleNamespace(
        poll=lambda: next(operations), editing=None, close=lambda: None))
    args = SimpleNamespace(allow_motion=True, demo=False, session_port=0, camera='none', rgb_topic=None)

    assert runtime.run_live(args, config) == 0

    assert len(reads) == 4 and len(pre_arm_stamp) == 1
    if fresh_after_arm:
        assert len(fake_motion.commands) == 1
        _, generated, expires = fake_motion.commands[0]
        assert generated > pre_arm_stamp[0]+.5
        assert generated < expires <= reads[2]+config['robot']['max_data_age_s']
        assert not fake_motion.stops
    else:
        assert not fake_motion.commands
        assert len(fake_motion.stops) == 1 and 'stale' in fake_motion.stops[0]


def test_online_decisions_are_recorded_once_with_their_kind(control):
    controller, snapshot, events = control
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    controller.step(snapshot, snapshot.captured_at)
    decisions = [e for e in events.items if e['event'] == 'decision']
    assert len(decisions) == 1 and decisions[0]['decision_kind'] == 'navigate'
    assert not controller.tasks.engine.events


@pytest.mark.parametrize('operation', ['navigate', 'r'])
def test_cold_plan_longer_than_ttl_precedes_arm_and_is_not_rebuilt(control, monkeypatch, operation):
    import numpy as np
    from deploy.app import controller as module
    from deploy.robot.safety import CommandGuard
    controller, snapshot, events = control
    clock = [snapshot.captured_at]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    guard = CommandGuard(controller.control)
    original_arm = controller.motion.arm
    original_submit = controller.motion.submit
    armed_at = []
    def arm():
        armed_at.append(clock[0])
        guard.arm(clock[0])
        return original_arm()
    def submit(command, generated, expires):
        assert guard.submit(command, generated, expires, clock[0])
        original_submit(command, generated, expires)
    controller.motion.arm, controller.motion.submit = arm, submit
    decide = controller.tasks.engine._decide
    decisions = []
    def slow_decide(recognition):
        assert not guard.armed and not controller.motion.commands
        decisions.append(clock[0])
        clock[0] += .7  # Two command TTLs; the initial route is not a live command.
        return decide(recognition)
    monkeypatch.setattr(controller.tasks.engine, '_decide', slow_decide)
    if operation == 'r':
        controller.tasks.start('navigate', (1., 0., .3), clock[0])
        controller.tasks.pause()
    controller.operate(operation, '1,0' if operation == 'navigate' else None, snapshot, clock[0])
    assert armed_at == [snapshot.captured_at+.7]
    assert len(decisions) == 1 and controller.tasks.engine.phase == 'navigate'
    assert not controller.motion.commands
    np.testing.assert_array_equal(controller.tasks.engine.limiter.command, np.zeros(3))
    # This is a new sensor observation, not a renewed timestamp on the old command.
    clock[0] += .02
    fresh = replace(snapshot, captured_at=clock[0], pose_at=clock[0], lidar_at=clock[0])
    controller.step(fresh, clock[0])
    assert len(decisions) == 1 and len(controller.motion.commands) == 1
    assert guard.tick(clock[0]) is not None and guard.armed
    prepared = next(e for e in events.items if e['event'] == 'navigation_prepared')
    assert prepared['duration_s'] == pytest.approx(.7) and prepared['sent_to_robot'] is False
    # After activation, the original watchdog still latches off on a real stall.
    assert guard.tick(clock[0]+controller.control['command_ttl_s']+.01) is None
    assert guard.reason == 'command_timeout' and not guard.armed


def test_preparation_failure_never_arms(control, monkeypatch):
    controller, snapshot, events = control
    def failed(recognition):
        return controller.tasks.engine._finish(False, 'no_safe_route')
    monkeypatch.setattr(controller.tasks.engine, '_decide', failed)
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.motion.state == 'manual' and not controller.motion.commands
    assert controller.tasks.state == 'completed'
    assert events.items[-1]['event'] == 'task_result'
    assert events.items[-1]['result']['reason'] == 'no_safe_route'
