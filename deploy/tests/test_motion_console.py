"""Bounded-trial operator flow with no hardware imports or outputs."""
from types import SimpleNamespace
import pytest
from deploy.app.config import load
from deploy.app.motion_test import MotionTrial
from deploy.tests.test_controller import Events


class FakeMotion:
    def __init__(self):
        self.state = 'manual'
        self.arms, self.commands, self.stops = [], [], []
        self.pending = []
        self.status = dict(healthy=True, stationary=True, takeover_held=False)
    def telemetry(self):
        return self.status
    def arm(self, **kwargs):
        self.arms.append(kwargs)
        self.state = 'active'
        self.pending.append({'state': 'active', 'detail': '', 'accepted': True})
        return True
    def events(self):
        events, self.pending = self.pending, []
        return events
    def submit(self, *args):
        self.commands.append(args)
    def stop(self, reason):
        self.stops.append(reason)
        self.state = 'stopping'


@pytest.fixture
def console(monkeypatch):
    config = load()
    config['robot']['takeover_button_mask'] = 2
    motion, events = FakeMotion(), Events()
    trial = MotionTrial(config, motion, events)
    monkeypatch.setattr('deploy.app.motion_test.time.monotonic', lambda: 100.)
    return trial, motion, events


def test_startup_and_prepare_never_send_motion(console):
    trial, motion, _ = console
    trial.tick(100.)
    trial.operate('a', 100.)
    trial.tick(100.)
    assert trial.state == 'ready'
    assert not motion.arms and not motion.commands


def test_button_inspection_allowed_but_unset_mask_prevents_trial(console):
    trial, motion, _ = console
    trial.config['robot']['takeover_button_mask'] = 0
    with pytest.raises(ValueError, match='takeover_button_mask'):
        trial.operate('a', 100.)
    assert not motion.arms


@pytest.mark.parametrize('key', ['f', 'j', 't'])
def test_every_action_needs_fresh_prepare_and_cannot_extend_trial(console, key):
    trial, motion, _ = console
    with pytest.raises(ValueError):
        trial.operate(key, 100.)
    trial.operate('a', 100.)
    trial.operate(key, 100.)
    trial.tick(100.)
    assert motion.arms == [{'duration_s': 3.}]
    assert len(motion.commands) == 1
    assert motion.commands[0][0] == {'f': (.10, 0., 0.), 'j': (0., 0., .25),
                                    't': (.05, 0., 0.)}[key]
    with pytest.raises(ValueError):
        trial.operate(key, 100.1)
    assert len(motion.arms) == 1
    with pytest.raises(ValueError, match='等待'):
        trial.operate('a', 100.2)


def test_watchdog_trial_stops_feeding_but_does_not_fake_a_stop_request(console):
    trial, motion, _ = console
    trial.operate('a', 100.)
    trial.operate('t', 100.)
    trial.tick(100.)
    trial.tick(100.05)
    trial.tick(100.2)
    trial.tick(100.3)
    assert len(motion.commands) == 2 and not motion.stops
    trial.events([{'state': 'stopping', 'detail': 'command_timeout'}])
    assert trial.last_stop_confirmed is False
    trial.events([{'state': 'stopped', 'detail': 'command_timeout', 'peak_speed': .02}])
    assert trial.last_stop_confirmed is True and trial.state == 'idle'


def test_prepare_expires_and_space_cancels_it(console):
    trial, motion, _ = console
    trial.operate('a', 100.)
    trial.tick(106.)
    assert trial.state == 'idle'
    with pytest.raises(ValueError):
        trial.operate('f', 106.)
    trial.operate('a', 110.)
    trial.operate('stop', 111.)
    assert trial.ready_until is None
    assert motion.stops == ['operator_stop']


@pytest.mark.parametrize('field', ['healthy', 'stationary', 'takeover_held'])
def test_unsafe_feedback_cannot_prepare(console, field):
    trial, motion, _ = console
    motion.status[field] = field == 'takeover_held'
    with pytest.raises(ValueError, match='准备失败'):
        trial.operate('a', 100.)
    assert trial.ready_until is None and not motion.arms


def test_old_stop_received_during_arm_does_not_clear_new_trial(console):
    trial, motion, events = console
    motion.pending = [{'state': 'stopping', 'detail': 'previous'},
                      {'state': 'stopped', 'detail': 'previous'}]
    trial.operate('a', 100.)
    trial.operate('f', 100.)
    trial.events(motion.events())  # Events are no longer deferred until the next loop.
    trial.tick(100.)
    assert trial.number == 1 and trial.started == 100.
    assert trial.last_stop_confirmed is False and motion.commands
    assert [e['state'] for e in events.items if e['event'] == 'motion_state'] == ['stopping', 'stopped', 'active']


def test_fault_consumed_in_loop_is_retained_in_final_summary(monkeypatch):
    import signal
    from deploy.app import motion_test
    from deploy.tests.test_controller import Events
    calls = []
    session = Events()
    session.info = {'session_id': 'fake'}
    session.url = 'fake-session'
    session.set_status = lambda *a, **k: None
    def finish(summary):
        calls.append('session')
        session.summary = summary
    session.close = finish
    motion = FakeMotion()
    motion.pending = [{'state': 'fault', 'detail': 'Stop not confirmed'}]
    def close_motion():
        calls.append('motion')
        return []  # The fault was already delivered once in the live loop.
    motion.close = close_motion
    monkeypatch.setattr(motion_test, 'Session', lambda *a, **k: session)
    monkeypatch.setattr(motion_test, 'MotionProcess', lambda *a, **k: motion)
    monkeypatch.setattr(motion_test, 'Keyboard', lambda **k: SimpleNamespace(
        poll=lambda: ('q', None), close=lambda: calls.append('keyboard')))
    handler = signal.getsignal(signal.SIGINT)
    assert motion_test.run(SimpleNamespace(session_port=0), load()) == 2
    assert calls == ['motion', 'keyboard', 'session']
    assert signal.getsignal(signal.SIGINT) == handler
    assert session.summary['error'] == 'Stop not confirmed'
    assert not session.summary['completed']
