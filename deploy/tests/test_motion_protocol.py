"""Protocol regressions run only with fake SDKs; no robot connection is possible."""
from collections import deque
import multiprocessing as mp
import time
from types import SimpleNamespace

import pytest

from deploy.app.config import load
from deploy.robot.motion import MotionProcess, UnitreeSport, _worker
from deploy.robot.safety import CommandGuard
from deploy.tests.test_deploy import FakeSport
from deploy.tests.test_motion_acceptance import telemetry_only_sport


class TelemetrySport(FakeSport):
    def __init__(self, robot, control):
        super().__init__(robot, control)
        self.velocity = [0., 0., 0.]
        self.held = robot.get('held')
        self.release_latch = robot.get('release_latch')
        self.takeover = bool(robot.get('initial_latch'))

    @property
    def takeover_held(self):
        return self.held is not None and self.held.is_set()

    def move(self, velocity):
        self.velocity = list(map(float, velocity))
        self.is_stopped = False

    def stop(self):
        self.velocity = [0., 0., 0.]
        self.is_stopped = True

    def telemetry(self, now):
        return {'keys': 2 if self.takeover_held else 0, 'state_code': 100,
                'healthy': self.healthy(now), 'stationary': self.stopped(now),
                'state_age_s': 0., 'remote_age_s': 0.,
                'velocity': self.velocity[:2]+[0.], 'yaw_speed': self.velocity[2],
                'takeover': self.takeover, 'takeover_held': self.takeover_held,
                'received_at': now}


def fake_motion(robot=None, control=None, test_mode=True):
    context = mp.get_context('spawn')
    motion = MotionProcess.__new__(MotionProcess)
    motion.pipe, child = context.Pipe()
    robot = {'takeover_button_mask':2, **(robot or {})}
    motion.process = context.Process(target=_worker,
                                    args=(child, robot, control or load()['runtime']['control'], TelemetrySport),
                                    kwargs={'test_mode': test_mode})
    motion.state, motion.closing = 'starting', False
    motion._pending, motion._responses, motion._request_id = [], {}, 0
    motion.process.start()
    child.close()
    assert motion.pipe.poll(5), 'fake SDK did not initialize'
    return motion


class ReplyPipe:
    def __init__(self, queued=()):
        self.queued = deque(queued)
        self.sent = []

    def send(self, packet):
        self.sent.append(packet)
        if packet['op'] == 'arm':
            self.queued.append({'state':'active', 'detail':'', 'accepted':True,
                                'request_id':packet['request_id']})
        if packet['op'] == 'status':
            self.queued.append({'kind':'reply', 'request_id':packet['request_id'],
                                'telemetry':{'keys':2, 'healthy':True}})

    def poll(self, timeout=0):
        return bool(self.queued)

    def recv(self):
        return self.queued.popleft()


def protocol_client(pipe):
    motion = MotionProcess.__new__(MotionProcess)
    motion.pipe, motion.process = pipe, SimpleNamespace(is_alive=lambda: True)
    motion.state, motion.closing = 'manual', False
    motion._pending, motion._responses, motion._request_id = [], {}, 0
    return motion


def test_arm_waits_for_its_reply_and_preserves_prior_stop_event():
    motion = protocol_client(ReplyPipe([{'state':'stopped', 'detail':'earlier stop'}]))
    assert motion.arm()
    assert [event['state'] for event in motion.events()] == ['stopped', 'active']
    assert motion.events() == []


def test_status_does_not_change_motion_state_or_consume_events():
    motion = protocol_client(ReplyPipe([{'state':'stopped', 'detail':'complete'}]))
    assert motion.telemetry(timeout_s=.05)['keys'] == 2
    assert motion.state == 'stopped'
    assert motion.events() == [{'state':'stopped', 'detail':'complete'}]
    assert motion.telemetry()['healthy']
    assert motion.events() == []


def test_arm_timeout_requests_stop(monkeypatch):
    motion = protocol_client(ReplyPipe())
    def timeout(*args, **kwargs):
        raise RuntimeError('acknowledgement timed out')
    monkeypatch.setattr(motion, '_request', timeout)
    assert not motion.arm()
    assert motion.pipe.sent == [{'op':'stop', 'reason':'arm_ack_timeout'}]


@pytest.mark.parametrize('timeout', [0., -.1, .31, float('nan'), float('inf')])
def test_status_timeout_must_be_bounded(timeout):
    motion = protocol_client(ReplyPipe())
    with pytest.raises(ValueError, match='timeout_s'):
        motion.telemetry(timeout_s=timeout)


def test_sdk_status_exposes_current_buttons_and_keeps_latch_after_release():
    sdk = telemetry_only_sport()
    sdk.takeover = False
    sdk.robot['takeover_button_mask'] = 2
    sdk._remote(SimpleNamespace(wireless_remote=b'\0\0\2\0'))
    now = time.monotonic()
    sdk.state_at = now
    held = sdk.telemetry(now)
    assert held['keys'] == 2 and held['takeover_held'] and held['takeover']
    sdk._remote(SimpleNamespace(wireless_remote=b'\0\0\0\0'))
    released = sdk.telemetry(time.monotonic())
    assert released['keys'] == 0 and not released['takeover_held'] and released['takeover']
    assert released['state_code'] == 100 and released['stationary']
    sdk.state.velocity[0] = float('nan')
    assert sdk.telemetry(time.monotonic())['velocity'][0] is None


def test_nonrenewable_deadline_survives_fresh_commands():
    guard = CommandGuard(load()['runtime']['control'])
    guard.arm(10., duration_s=.5)
    for now in (10., 10.2, 10.4):
        assert guard.submit([.1,0,0], now, now+.3, now)
        assert guard.tick(now) is not None
    assert guard.tick(10.5) is None
    assert guard.reason == 'test_duration_elapsed'
    assert not guard.submit([.1,0,0], 10.5, 10.8, 10.5)


@pytest.mark.parametrize('duration,expected_duration', [(.18,.18), (8.,1.)])
def test_test_worker_enforces_duration_and_speed_even_with_fresh_commands(duration, expected_duration):
    motion = fake_motion()
    try:
        assert motion.arm(duration_s=duration)
        events = motion.events()
        assert events[-1]['duration_s'] == expected_duration
        start = time.monotonic()
        while time.monotonic()-start < expected_duration+1.:
            now = time.monotonic()
            motion.submit([1.,1.,1.], now, now+.3)
            events.extend(motion.events())
            if any(e['state']=='stopped' for e in events):
                break
            time.sleep(.025)
        stopped = next(e for e in events if e['state']=='stopped')
        assert stopped['detail'] == 'test_duration_elapsed'
        assert stopped['telemetry']['stationary']
        assert stopped['peak_speed'] > 0
        assert stopped['peak_speed'] <= (.05**2+.03**2)**.5+1e-6
        assert stopped['peak_yaw_speed'] <= .15+1e-6
        accepted = [e for e in events if e.get('event')=='move_accepted']
        assert len(accepted) == 1
        assert all(abs(v)<=limit+1e-6 for v,limit in zip(accepted[0]['velocity'], (.05,.03,.15)))
        assert len([e for e in events if e.get('event')=='stop_requested']) == 1
        now = time.monotonic()
        motion.submit([1.,0.,0.], now, now+.3)
        assert motion.telemetry()['stationary']
    finally:
        motion.close()


def test_test_worker_can_observe_unconfigured_remote_but_cannot_arm():
    motion = fake_motion({'takeover_button_mask':0})
    try:
        assert motion.telemetry()['state_code'] == 100
        assert not motion.arm()
        assert 'takeover_button_mask' in motion.events()[-1]['detail']
    finally:
        motion.close()


def test_held_button_cannot_be_cleared_by_arm_and_release_requires_ack():
    held = mp.get_context('spawn').Event()
    held.set()
    motion = fake_motion({'held':held, 'initial_latch':True})
    try:
        assert not motion.arm()
        assert motion.telemetry()['takeover']
        assert 'Release' in motion.events()[-1]['detail']
        held.clear()
        assert not motion.arm()
        assert not motion.telemetry()['takeover']
        assert 'latch cleared' in motion.events()[-1]['detail']
        assert motion.arm(duration_s=.1)
    finally:
        motion.close()


class LostLinkSport(TelemetrySport):
    def __init__(self, robot, control):
        super().__init__(robot, control)
        self.failed = False

    def healthy(self, now):
        return not self.failed

    def move(self, velocity):
        super().move(velocity)
        if any(abs(v)>1e-6 for v in velocity):
            self.failed = True

    def stop(self):
        raise RuntimeError('fake SDK communication unavailable')


def test_sdk_link_loss_is_not_reported_as_a_confirmed_stop():
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    control = dict(load()['runtime']['control'], stop_timeout_s=.15, stop_settle_s=.03)
    process = context.Process(target=_worker, args=(child, {'takeover_button_mask':2}, control, LostLinkSport),
                              kwargs={'test_mode':True})
    process.start()
    child.close()
    try:
        assert parent.poll(5) and parent.recv()['state']=='manual'
        parent.send({'op':'arm'})
        assert parent.poll(1) and parent.recv()['state']=='active'
        now = time.monotonic()
        parent.send({'op':'velocity', 'velocity':[.05,0,0], 'generated_at':now, 'expires_at':now+.3})
        events=[]
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            if parent.poll(.05):
                events.append(parent.recv())
                if events[-1]['state']=='fault':
                    break
        assert any(e['state']=='stop_error' for e in events)
        assert events[-1]['state']=='fault'
        assert not any(e['state']=='stopped' for e in events)
    finally:
        if process.is_alive():
            parent.send({'op':'close'})
            process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(1)
        parent.close()


def test_command_flood_cannot_postpone_worker_deadline(monkeypatch):
    import deploy.robot.motion as module
    ticks=[10.]
    def clock():
        ticks[0]+=.002
        return ticks[0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=clock, sleep=lambda seconds:None))
    class FloodPipe:
        count=0
        reports=[]
        def poll(self):
            return True
        def recv(self):
            self.count+=1
            if self.count==1:
                return {'op':'arm', 'duration_s':.01}
            if self.count>=96:
                return {'op':'close'}
            now=clock()
            return {'op':'velocity', 'velocity':[.05,0,0], 'generated_at':now, 'expires_at':now+.3}
        def send(self, event):
            self.reports.append((self.count,event))
        def close(self):
            pass
    pipe=FloodPipe()
    control=dict(load()['runtime']['control'], stop_settle_s=.001)
    _worker(pipe, {'takeover_button_mask':2}, control, FakeSport, test_mode=True)
    count,event=next((n,e) for n,e in pipe.reports if e['state']=='stopping')
    assert count==32
    assert event['detail']=='test_duration_elapsed'
