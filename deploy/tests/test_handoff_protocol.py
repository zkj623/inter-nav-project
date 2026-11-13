"""Exercise the real motion worker and guard without ROS or a robot SDK."""
from types import SimpleNamespace

import pytest

from deploy.app.config import load
from deploy.robot.motion import _worker
from deploy.tests.test_deploy import FakeSport
from deploy.tests.test_planning_handoff import run


@pytest.mark.parametrize('delay,unsafe,expected', [(.10, None, 'planning_hold:1:global_decision'),
                                                 (.30, None, 'command_timeout'), (.50, None, 'command_timeout'),
                                                 (.10, 'takeover', 'remote_takeover'),
                                                 (.10, 'health', 'robot_state_stale_or_unsafe')])
def test_planning_stop_cannot_relabel_real_fault_before_worker_tick(monkeypatch, delay, unsafe, expected):
    import deploy.robot.motion as module
    clock = [10.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
                                                      sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))

    class Pipe:
        def __init__(self):
            self.packets = [dict(op='arm'), dict(op='velocity', velocity=[.1, 0., 0.],
                                              generated_at=10., expires_at=10.3, command_id=1),
                            dict(op='stop', reason='planning_hold:1:global_decision')]
            self.reports = []
            self.closed = False

        def poll(self):
            return bool(self.packets) or clock[0] > 11.

        def recv(self):
            if not self.packets:
                return {'op': 'close'}
            packet = self.packets.pop(0)
            if packet['op'] == 'stop':
                clock[0] += delay  # No intervening tick: reproduce the pipe-drain race.
            return packet

        def send(self, event):
            self.reports.append(event)

        def close(self):
            self.closed = True

    class Sport(FakeSport):
        @property
        def takeover(self):
            return unsafe == 'takeover' and clock[0] >= 10.1

        @takeover.setter
        def takeover(self, value):
            pass

        def healthy(self, now):
            return not (unsafe == 'health' and now >= 10.1)

    pipe = Pipe()
    _worker(pipe, {}, load()['runtime']['control'], Sport)
    assert pipe.closed
    ack = next(e for e in pipe.reports if e.get('kind') == 'command_ack')
    assert ack['accepted'] and ack['command_id'] == 1 and ack['expires_at'] == 10.3
    stops = [e for e in pipe.reports if e['state'] in ('stopping', 'stopped')]
    assert stops and all(e['detail'] == expected for e in stops)
    assert any(e['state'] == 'stopped' for e in stops)
    assert (any(e['state'] == 'rejected' for e in pipe.reports)) == (unsafe is not None or expected == 'command_timeout')


def test_command_ack_is_recorded_without_changing_task_state():
    from deploy.tests.test_controller import Motion, Events
    from deploy.app.controller import NavigationController
    from deploy.app.tasks import Tasks
    config = load()
    controller = NavigationController(config, Tasks(config['navigation']), Motion(), Events())
    controller.motion_events([{'kind': 'command_ack', 'state': 'active', 'command_id': 7,
                              'accepted': True, 'worker_at': 10., 'expires_at': 10.3, 'detail': 'active'}])
    assert controller.tasks.state == 'observe'
    assert controller.session.items[-1]['event'] == 'command_accepted'
    assert controller.session.items[-1]['command_id'] == 7


def test_short_previous_lease_yields_before_starting_next_control_cycle(run, monkeypatch):
    current = run.sample()
    run.controller.last_expires_at = run.clock[0]+.12408  # P4 actual remaining budget.
    run.controller.last_decision_cost = .09358
    calls = []
    original = run.tasks.update
    def update(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(run.tasks, 'update', update)
    run.controller.step(current, run.clock[0])
    assert not calls and not run.motion.commands
    assert run.controller.planning_hold['reason'] == 'control_budget'


def test_rejected_command_ack_pauses_before_following_state_event(run):
    run.controller.motion_events([{'kind': 'command_ack', 'state': 'command_rejected', 'command_id': 1,
                                  'accepted': False, 'detail': 'command_timeout', 'worker_at': run.clock[0]}])
    assert run.tasks.state == 'paused' and run.controller.planning_hold is None
    assert run.controller.task_error == 'motion_command_rejected:command_timeout'


def test_already_expired_lease_remains_a_fault_not_a_planning_hold(run):
    current = run.sample()
    run.controller.last_expires_at = run.clock[0]-.001
    run.controller.step(current, run.clock[0])
    assert run.controller.planning_hold is None and run.tasks.state == 'paused'
    assert run.controller.task_error == 'control_deadline_missed'
    assert run.motion.arm_count == 1 and not run.motion.commands


def test_new_command_still_valid_cannot_bridge_expired_previous_command(run, monkeypatch):
    current = run.sample()
    run.controller.last_expires_at = run.clock[0]+.18
    original = run.tasks.update
    def update(*args):
        result = original(*args)
        run.clock[0] += .20
        return result
    monkeypatch.setattr(run.tasks, 'update', update)
    run.controller.step(current, current.captured_at)
    assert run.tasks.state == 'paused' and not run.motion.commands
    discarded = [e for e in run.session.items if e['event'] == 'command_discarded'][-1]
    assert discarded['reason'] == 'previous_command_expired'
    assert run.controller.planning_hold is None
