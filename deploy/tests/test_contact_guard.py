"""Contact envelopes are independently enforced without ROS or an SDK connection."""
from tempfile import TemporaryFile
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import ROOT, load
from deploy.robot.motion import _worker
from deploy.robot.safety import CommandGuard
from deploy.tests.test_deploy import FakeSport
from deploy.tests.test_motion_protocol import ReplyPipe, protocol_client


def control():
    return dict(load()['runtime']['control'], contact_max_duration_s=.5,
                contact_limits=dict(max_vx=.15, max_vy=.04, max_wz=.2,
                                    acceleration=.2, angular_acceleration=.4))


def envelope(now=10., **changes):
    packet = dict(id='skill-1/segment-1', source_expires_at=now+.25, deadline=10.5,
                  phase='engage', limits=control()['contact_limits'].copy())
    packet.update(changes)
    return packet


def submit(guard, now=10., packet=None, velocity=(.4, .1, .5)):
    return guard.submit(velocity, now, now+.3, now,
                        interaction=envelope(now) if packet is None else packet)


def test_motion_packet_carries_envelope_without_changing_command_id_contract():
    motion = protocol_client(ReplyPipe())
    packet = envelope()
    assert motion.submit([.1, 0., 0.], 10., 10.3, interaction=packet) == 1
    assert motion.pipe.sent[-1]['interaction'] == packet
    assert motion.submit([0., 0., 0.], 10.1, 10.4) == 2
    assert motion.pipe.sent[-1]['interaction'] is None


@pytest.mark.parametrize('missing', ['contact_limits', 'contact_max_duration_s'])
def test_unconfigured_writer_rejects_interaction_but_allows_navigation(missing):
    cfg = control()
    del cfg[missing]
    guard = CommandGuard(cfg)
    guard.arm(10.)
    assert guard.submit([.1, 0., 0.], 10., 10.3, 10.)
    assert not submit(guard)
    assert guard.reason == 'interaction_profile_unconfigured' and not guard.armed


@pytest.mark.parametrize('change', [dict(id=''), dict(object_id=[]), dict(phase=''), dict(deadline=float('nan')),
                                   dict(source_expires_at=True), dict(limits={}),
                                   dict(limits=dict(control()['contact_limits'], max_vx=-1.))])
def test_invalid_envelope_latches_off(change):
    guard = CommandGuard(control())
    guard.arm(10.)
    assert not submit(guard, packet=envelope(**change))
    assert guard.reason == 'invalid_interaction_metadata'
    np.testing.assert_array_equal(guard.output, np.zeros(3))


def test_fixed_budget_and_limits_cannot_be_raised_by_new_phase_or_fresh_sample():
    guard = CommandGuard(control())
    guard.arm(10.)
    assert submit(guard, packet=envelope(deadline=30.))
    assert guard.contact.active['deadline'] == 10.5  # independent profile cap
    for now, phase in [(10.1, 'push'), (10.2, 'release'), (10.3, 'verify'), (10.4, 'retry')]:
        packet = envelope(now, phase=phase, deadline=now+20.,
                          limits={key: 100. for key in control()['contact_limits']})
        assert submit(guard, now, packet)
        assert guard.contact.active['deadline'] == 10.5
        assert guard.contact.active['limits'] == control()['contact_limits']
    assert not submit(guard, 10.5, envelope(10.5, deadline=30.))
    assert guard.reason == 'interaction_duration_elapsed'


def test_payload_can_only_shrink_existing_limits_and_deadline():
    guard = CommandGuard(control())
    guard.arm(10.)
    assert submit(guard)
    smaller = {key: value/2. for key, value in control()['contact_limits'].items()}
    assert submit(guard, 10.1, envelope(10.1, deadline=10.4, limits=smaller))
    assert submit(guard, 10.2, envelope(10.2, deadline=11.))
    assert guard.contact.active['deadline'] == 10.4
    assert guard.contact.active['limits'] == smaller
    assert guard.tick(10.4) is None
    assert guard.reason == 'interaction_duration_elapsed'


@pytest.mark.parametrize('via_tick', [False, True])
def test_stale_sensor_lease_cannot_be_bridged_with_fresh_packet(via_tick):
    guard = CommandGuard(control())
    guard.arm(10.)
    assert submit(guard, packet=envelope(source_expires_at=10.1))
    assert guard.expires == 10.1
    if via_tick:
        assert guard.tick(10.1) is None
    else:
        assert not submit(guard, 10.1, envelope(10.1))
    assert guard.reason == 'interaction_observation_expired'
    assert not submit(guard, 10.11, envelope(10.11, id='a-new-id'))


def test_new_source_renews_only_observation_lease_not_skill_duration():
    guard = CommandGuard(control())
    guard.arm(10.)
    assert submit(guard, packet=envelope(source_expires_at=10.1))
    assert submit(guard, 10.09, envelope(10.09, source_expires_at=10.3))
    assert guard.expires == 10.3
    assert guard.contact.active['deadline'] == 10.5
    assert guard.tick(10.15) is not None


def test_contact_cannot_hide_a_shorter_command_ttl():
    guard = CommandGuard(control())
    guard.arm(10.)
    assert guard.submit([.1, 0., 0.], 10., 10.08, 10., interaction=envelope())
    assert guard.tick(10.08) is None
    assert guard.reason == 'command_timeout'


@pytest.mark.parametrize('packet,reason', [(None, 'interaction_metadata_missing'),
                                         (envelope(object_id='other'), 'interaction_object_changed_without_stop'),
                                         (envelope(id='skill-2'), 'interaction_id_changed_without_stop')])
def test_metadata_cannot_disappear_or_change_id_while_armed(packet, reason):
    guard = CommandGuard(control())
    guard.arm(10.)
    assert submit(guard)
    assert not guard.submit([.1, 0., 0.], 10.05, 10.35, 10.05, interaction=packet)
    assert guard.reason == reason and not guard.armed
    assert not submit(guard, 10.06)


def test_explicit_stop_and_arm_restore_navigation_but_do_not_reuse_id():
    guard = CommandGuard(control())
    guard.arm(10.)
    assert submit(guard)
    guard.stop('interaction_completed')
    guard.arm(11.)  # worker separately requires confirmed stationary feedback
    assert guard.submit([.6, 0., .6], 11., 11.3, 11.)
    np.testing.assert_array_equal(guard.velocity, [.6, 0., .6])
    assert not submit(guard, 11.05, envelope(11.05, deadline=12.))
    assert guard.reason == 'interaction_id_reused'
    guard.arm(12.)
    assert submit(guard, 12., envelope(12., id='skill-2', deadline=13.))


def test_contact_clamps_existing_navigation_output_and_all_later_phase_outputs():
    guard = CommandGuard(control())
    guard.arm(9.)
    for now in np.arange(9., 10., .05):
        assert guard.submit([.8, .3, .9], now, now+.3, now)
        guard.tick(now)
    assert guard.output[0] > .15 and guard.output[2] > .2
    assert submit(guard)
    caps = np.array([.15, .04, .2])
    assert np.all(np.abs(guard.output) <= caps)
    guard.tick(10.)
    for now, phase in [(10.05, 'align'), (10.1, 'push'), (10.15, 'release'), (10.2, 'verify')]:
        assert submit(guard, now, envelope(now, phase=phase), velocity=(-9., -9., -9.))
        before = guard.output.copy()
        output = guard.tick(now)
        assert np.all(np.abs(output) <= caps+1e-12)
        assert np.all(np.abs(output-before) <= np.array([.2, .2, .4])*.05+1e-12)


def run_worker(monkeypatch, packets, *, unsafe=None, worker_control=None):
    """Scheduled pipe and clock exercise the actual writer loop deterministically."""
    import deploy.robot.motion as module
    clock = [packets[0][0]]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
                                                       sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    moves = []
    class Sport(FakeSport):
        state = SimpleNamespace(gait_type=1, foot_force=[22, 0, 0, 21],
                                foot_position_body=[0., 0., -.29, 0., 0., -.21, 0., 0., -.22, 0., 0., -.29])

        @property
        def state_at(self):
            return clock[0]

        @property
        def takeover(self):
            return unsafe == 'takeover' and clock[0] >= 10.08

        @takeover.setter
        def takeover(self, value):
            pass

        def healthy(self, now):
            return not (unsafe == 'health' and now >= 10.08)

        def move(self, velocity):
            moves.append((clock[0], np.array(velocity)))
            super().move(velocity)

    class Pipe:
        def __init__(self):
            self.packets = list(packets)+[(packets[-1][0]+1.5, dict(op='close'))]
            self.reports = []

        def poll(self):
            return bool(self.packets) and clock[0] >= self.packets[0][0]-1e-9

        def recv(self):
            _, packet = self.packets.pop(0)
            return packet(clock[0]) if callable(packet) else packet

        def send(self, event):
            self.reports.append(event)

        def close(self):
            pass

    pipe = Pipe()
    # A simulated Sport endpoint has its own writer. Keep its real flock on
    # a private file even if a live run is open on the same host.
    with monkeypatch.context() as offline:
        offline.setattr(module, 'open', lambda path, mode: TemporaryFile(), raising=False)
        _worker(pipe, {}, dict(worker_control or control(), stop_settle_s=.02), Sport)
    return pipe.reports, moves


def velocity_packet(now, **changes):
    return dict(op='velocity', velocity=[.8, .3, .9], generated_at=now, expires_at=now+.3,
                command_id=int((now-9.)*1000), interaction=envelope(now, **changes))


def test_worker_reports_first_forward_push_speed_after_sdk_move(monkeypatch):
    reports, moves = run_worker(monkeypatch, [(10., dict(op='arm')),
        (10., lambda now: velocity_packet(now, phase='align', object_id='box-1')),
        (10.08, lambda now: velocity_packet(now, phase='engage', object_id='box-1')),
        (10.16, lambda now: velocity_packet(now, phase='push', object_id='box-1'))])
    sent = [event for event in reports if event.get('event') == 'push_speed_sent']
    assert len(sent) == 1 and sent[0]['phase'] == 'engage'
    assert sent[0]['object_id'] == 'box-1'
    assert sent[0]['velocity'][0] > 0
    assert any(t == sent[0]['worker_at'] and np.array_equal(v, sent[0]['velocity']) for t, v in moves)


@pytest.mark.parametrize('phase,requested', [('engage', .20), ('push', .16), ('engage', .12)])
def test_field_push_launch_sends_selected_forward_speed_on_first_sdk_tick(monkeypatch, phase, requested):
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['runtime']['control']
    def packet(t, velocity):
        return dict(velocity_packet(t, phase=phase, object_id='box-1', limits=cfg['contact_limits']),
                    velocity=velocity)

    reports, moves = run_worker(monkeypatch, [(10., dict(op='arm')),
        (10., packet(10., [0., 0., 0.])), (10.04, packet(10.04, [requested, .08, .2]))],
        worker_control=cfg)
    sent, = [event for event in reports if event.get('event') == 'push_speed_sent']
    t, velocity = next((t, v) for t, v in moves if v[0] > 0.)
    assert t == pytest.approx(10.04) and t == sent['worker_at']
    np.testing.assert_array_equal(velocity, sent['velocity'])
    assert velocity[0] == pytest.approx(requested)
    assert 0. < velocity[1] < .08 and 0. < velocity[2] < .2


@pytest.mark.parametrize('box,phase,requested,previous,expected', [
    ('other-box', 'engage', .15, 0., .004), ('box-1', 'align', .15, 0., .004),
    ('box-1', 'push', -.15, 0., -.004), ('box-1', 'engage', .15, .05, .054),
    ('box-1', 'push', .15, 0., .12),
])
def test_field_push_launch_is_scoped_to_first_forward_tick_from_rest(box, phase, requested, previous, expected):
    cfg = dict(control(), field_push_launch_speeds={'box-1': .12})
    guard = CommandGuard(cfg)
    guard.arm(10.)
    guard.output[0] = previous
    assert submit(guard, packet=envelope(phase=phase, object_id=box), velocity=(requested, 0., 0.))
    assert guard.tick(10.02)[0] == pytest.approx(expected)


def test_field_push_launch_cannot_repeat_after_braking_in_same_segment():
    guard = CommandGuard(dict(control(), field_push_launch_speeds={'box-1': .12}))
    guard.arm(10.)
    packet = envelope(object_id='box-1')
    assert submit(guard, packet=packet, velocity=(.015, 0., 0.))
    assert guard.tick(10.02)[0] == pytest.approx(.015)
    assert submit(guard, 10.04, packet, velocity=(0., 0., 0.))
    assert guard.tick(10.12)[0] == 0.
    assert submit(guard, 10.13, packet, velocity=(.12, 0., 0.))
    assert guard.tick(10.14)[0] == pytest.approx(.004)


def test_timed_push_worker_cuts_forward_output_even_if_planner_keeps_sending(monkeypatch):
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['runtime']['control']
    def packet(t, duration=.11):
        return dict(velocity_packet(t, phase='timed_push', object_id='box-1', limits=cfg['contact_limits'],
                                    forward_duration_s=duration), velocity=[.2, 0., 0.])

    reports, moves = run_worker(monkeypatch, [(10., dict(op='arm')), (10.04, packet(10.04)),
        (10.14, packet(10.14)), (10.22, packet(10.22)), (10.28, packet(10.28, .3))], worker_control=cfg)
    sent, = [r for r in reports if r.get('event') == 'push_speed_sent']
    assert sent['move_at'] == pytest.approx(10.04)
    assert all(v[0] == pytest.approx(.2) for t, v in moves if 10.04-1e-9 <= t < 10.15)
    assert any(t >= 10.16-1e-9 for t, _ in moves)
    assert all(not v.any() for t, v in moves if t >= 10.15)
    assert any(r['detail'] == 'timed_push_changed_without_stop' for r in reports if r['state'] == 'stopping')


def test_field_writer_ramps_to_stroke_speed_and_can_only_shorten_its_timer(monkeypatch):
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['runtime']['control']
    packets = [(10., dict(op='arm'))]
    for i in range(13):
        t = 10.04 + i*.10
        duration = 3. if i < 5 else (.95 if i < 12 else 3.)
        packet = velocity_packet(t, phase='timed_push', object_id='box-1',
                                 limits={**cfg['contact_limits'], 'max_vx': .8}, deadline=15., forward_duration_s=duration)
        packets.append((t, dict(packet, velocity=[.8, 0., 0.])))
    reports, moves = run_worker(monkeypatch, packets, worker_control=cfg)
    sent, = [r for r in reports if r.get('event') == 'push_speed_sent']
    assert sent['velocity'] == pytest.approx([.20, 0., 0.])
    at_speed, = [r for r in reports if r.get('event') == 'timed_push_speed_reached']
    assert at_speed['velocity'] == pytest.approx([.50, 0., 0.])
    assert any(t == at_speed['move_at'] and v[0] == pytest.approx(.50) for t, v in moves)
    active = [v[0] for t, v in moves if 10.04-1e-9 <= t < 10.99]
    assert max(active) == pytest.approx(.50)
    assert max(np.diff(active)) <= cfg['contact_limits']['acceleration'] / cfg['poll_hz'] + 1e-8
    assert all(not v.any() for t, v in moves if t >= 10.99)
    assert any(r.get('detail') == 'timed_push_changed_without_stop' for r in reports)


@pytest.mark.parametrize('unsafe,reason', [(None, 'interaction_observation_expired'),
                                         ('takeover', 'remote_takeover'),
                                         ('health', 'robot_state_stale_or_unsafe')])
def test_worker_stops_on_stale_source_takeover_or_unsafe_state(monkeypatch, unsafe, reason):
    reports, moves = run_worker(monkeypatch, [(10., dict(op='arm')),
        (10., lambda now: velocity_packet(now, source_expires_at=10.15))], unsafe=unsafe)
    assert any(e['state'] == 'stopped' and e['detail'] == reason for e in reports)
    assert moves and all(np.all(np.abs(v) <= [.15, .04, .2]) for _, v in moves)
    ack = next(e for e in reports if e.get('kind') == 'command_ack')
    assert ack['accepted'] and ack['expires_at'] == 10.15
    assert ack['interaction']['deadline'] == 10.5


def test_worker_requires_confirmed_stop_before_arm_after_contact_fault(monkeypatch):
    reports, _ = run_worker(monkeypatch, [(10., dict(op='arm', request_id=1)),
        (10., lambda now: velocity_packet(now)),
        (10.04, lambda now: velocity_packet(now, id='new-id')),
        (10.04, dict(op='arm', request_id=2)),
        (10.4, dict(op='arm', request_id=3)),
        (10.4, lambda now: velocity_packet(now, id='new-id', deadline=11.))])
    assert not next(e for e in reports if e.get('request_id') == 2)['accepted']
    assert next(e for e in reports if e.get('request_id') == 3)['accepted']
    assert any(e['state'] == 'stopped' and e['detail'] == 'interaction_id_changed_without_stop'
               for e in reports)


def test_worker_fresh_packets_and_phase_changes_cannot_extend_duration(monkeypatch):
    packets = [(10., dict(op='arm'))]
    packets.extend((float(now), lambda now: velocity_packet(now, phase='retry', deadline=now+20.))
                   for now in np.arange(10., 10.7, .08))
    reports, moves = run_worker(monkeypatch, packets)
    assert any(e['state'] == 'stopped' and e['detail'] == 'interaction_duration_elapsed' for e in reports)
    assert moves and all(at < 10.5 for at, _ in moves)
    accepted = [e for e in reports if e.get('kind') == 'command_ack' and e['accepted']]
    assert len(accepted) > 3
    assert all(e['interaction']['deadline'] == 10.5 for e in accepted)


def test_expired_interaction_cannot_be_relabelled_as_planning_hold(monkeypatch):
    reports, _ = run_worker(monkeypatch, [(10., dict(op='arm')),
        (10., lambda now: velocity_packet(now, deadline=10.1)),
        (10.12, dict(op='stop', reason='planning_hold:1:interaction_completed'))])
    assert any(e['state'] == 'rejected' and 'interaction_duration_elapsed' in e['detail'] for e in reports)
    assert any(e['state'] == 'stopped' and e['detail'] == 'interaction_duration_elapsed' for e in reports)


def test_worker_safe_stop_then_new_arm_allows_normal_navigation(monkeypatch):
    reports, moves = run_worker(monkeypatch, [(10., dict(op='arm')),
        (10., lambda now: velocity_packet(now)),
        (10.08, dict(op='stop', reason='planning_hold:1:interaction_completed')),
        (10.3, dict(op='arm', request_id=5)),
        (10.3, lambda now: dict(op='velocity', velocity=[.8, 0., 0.],
                               generated_at=now, expires_at=now+.3, command_id=5)),
        (10.5, lambda now: dict(op='velocity', velocity=[.8, 0., 0.],
                               generated_at=now, expires_at=now+.3, command_id=6))])
    assert next(e for e in reports if e.get('request_id') == 5)['accepted']
    assert any(e['state'] == 'stopped' and e['detail'] == 'planning_hold:1:interaction_completed'
               for e in reports)
    assert max(v[0] for at, v in moves if at >= 10.3) > .15


def test_zero_engage_packet_does_not_count_residual_navigation_motion(monkeypatch):
    packets = [(10., dict(op='arm')),
               (10., dict(op='velocity', velocity=[.12, 0., 0.], generated_at=10.,
                          expires_at=10.30, command_id=1)),
               (10.14, dict(velocity_packet(10.14), velocity=[0., 0., 0.])),
               (10.22, dict(op='stop', reason='stop'))]
    reports, moves = run_worker(monkeypatch, packets)
    assert any(10.14 <= t < 10.22 and v[0] > 0 for t, v in moves)
    assert not any(e.get('event') == 'push_speed_sent' for e in reports)


def test_scan_hold_arriving_after_expiry_keeps_the_actual_worker_stop(monkeypatch):
    token = 'planning_hold:1:first_push_wait_scan'
    packets = [(10., dict(op='arm')),
               (10., lambda now: velocity_packet(now, phase='engage', source_expires_at=10.07)),
               (10.09, dict(op='stop', reason=token))]
    reports, _ = run_worker(monkeypatch, packets)
    assert any(e.get('state') == 'stopped' and e.get('detail') == 'interaction_observation_expired' for e in reports)
    assert not any(e.get('state') == 'rejected' or e.get('detail') == token for e in reports)
