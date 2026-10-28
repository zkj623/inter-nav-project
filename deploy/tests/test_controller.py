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
def control(tmp_path):
    config = load()
    snapshot = next(snapshots(make_fixture(tmp_path)))
    tasks, motion, events = Tasks(config['navigation']), Motion(), Events()
    controller = NavigationController(config, tasks, motion, events)
    return controller, snapshot, events


def test_observe_never_arms_and_cancel_clears_task_without_losing_map(control):
    controller, snapshot, events = control
    controller.step(snapshot, snapshot.captured_at)
    retained = controller.tasks.component.mapping.map
    assert controller.motion.state == 'manual' and not controller.motion.commands
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'active'
    assert any(item.get('state') == 'active' for item in events.items)
    controller.operate('m', None, snapshot, snapshot.captured_at)
    assert controller.tasks.state == controller.tasks.mode == 'observe'
    assert controller.tasks.result is None
    assert controller.tasks.component.mapping.map is retained
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
    controller.step(snapshot, snapshot.captured_at)
    assert controller.tasks.state == 'completed'
    assert controller.tasks.result['success']
    assert controller.motion.stops == ['task_completed']
    assert len([e for e in events.items if e['event'] == 'task_result']) == 1
    assert not controller.motion.commands


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


def test_decision_timing_does_not_extend_command_or_sensor_expiry(control, monkeypatch):
    from deploy.app import controller as module
    controller, snapshot, events = control
    controller.operate('navigate', '1,0', snapshot, snapshot.captured_at)
    ticks = iter([snapshot.captured_at, snapshot.captured_at + .25])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    controller.step(snapshot, snapshot.captured_at)
    command = next(e for e in events.items if e['event'] == 'command_requested')
    assert command['decision_duration_s'] == pytest.approx(.25)
    assert command['generated_at'] == snapshot.captured_at
    assert command['expires_at'] <= snapshot.captured_at + controller.control['command_ttl_s']
    assert command['expires_at'] <= snapshot.lidar_at + controller.robot['max_data_age_s']


def test_cancel_exploration_discards_pending_perception_and_old_target():
    resets = []
    perception = SimpleNamespace(reset=lambda: resets.append(True))
    tasks = Tasks(load()['navigation'], perception)
    tasks.start('explore', 'chair', 10.)
    tasks.component.target_node = object()
    tasks.cancel()
    assert len(resets) == 2
    assert tasks.component.target_node is None
    assert tasks.component.mapping.semantic_target == ''
    assert tasks.mode == tasks.state == 'observe'


def test_mapping_failure_does_not_report_ready(control, monkeypatch):
    controller, snapshot, _ = control
    def fail(*args, **kwargs):
        raise ValueError('bad input')
    update = controller.tasks.component.mapping.update
    monkeypatch.setattr(controller.tasks.component.mapping, 'update', fail)
    assert controller.step(snapshot, snapshot.captured_at) == 'bad input'
    assert not controller.tasks.map_ready
    assert controller.tasks.last_lidar is None
    monkeypatch.setattr(controller.tasks.component.mapping, 'update', update)
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
