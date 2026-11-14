"""Execution identity, clock budgets and controller/writer handoff contracts."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.interaction.execution import InteractionExecution
from deploy.tests.test_planning_handoff import run
from deploy.tests.contact_fixtures import sdk_profile


def profile():
    # Synthetic scheduling fixture; it is not hardware calibration evidence.
    return replace(sdk_profile(), backend='sdk', calibration_reference='synthetic-execution-test',
                   max_forward_mps=.25, max_reverse_mps=.15, max_lateral_mps=.12, max_yaw_rps=.3,
                   forward_mps=.15, reverse_mps=.12, lateral_mps=.1, push_start_mps=.15, push_max_mps=.2,
                   min_translation_mps=.05, min_yaw_rps=.08, linear_acceleration_mps2=.3,
                   yaw_acceleration_rps2=.6, max_contact_seconds=10., stall_seconds=2.)


def test_contact_deadline_starts_once_and_survives_retry_phases_and_stop():
    execution = InteractionExecution(profile(), 100.)
    execution.start('B1', 10.)
    preparing = execution.metadata('prepare', 11., 11.3)
    assert preparing['deadline'] == 110.
    engaging = execution.metadata('engage', 20., 20.3)
    assert engaging['deadline'] == 30.
    execution.stopped()
    assert execution.segment == 1
    for phase in ('prepare', 'align_lateral', 'push', 'retreat', 'verify_passage'):
        resumed = execution.metadata(phase, 29., 29.3)
        assert resumed['id'] != engaging['id'] and resumed['deadline'] == 30.
        assert resumed['limits'] == engaging['limits']
    with pytest.raises(ValueError, match='interaction_duration_elapsed'):
        execution.metadata('push', 30., 30.3)


def test_short_whole_skill_budget_is_never_extended_by_contact_entry():
    execution = InteractionExecution(profile(), 5.)
    execution.start('B1', 10.)
    assert execution.metadata('engage', 13., 13.3)['deadline'] == 15.


@pytest.mark.parametrize('expiry', [None, 10., 9., float('nan'), float('inf')])
def test_execution_rejects_missing_stale_and_nonfinite_object_evidence(expiry):
    execution = InteractionExecution(profile(), 100.)
    execution.start('B1', 10.)
    with pytest.raises(ValueError, match='interaction_observation_expired'):
        execution.metadata('prepare', 10., expiry)


def test_completion_waits_for_physical_stop_before_resetting_scope():
    execution = InteractionExecution(profile(), 100.)
    execution.start('B1', 10.)
    first = execution.metadata('push', 10., 10.3)
    execution.completed()
    assert execution.active and execution.awaiting_stop
    assert execution.deadline == first['deadline']
    execution.stopped()
    assert not execution.active and not execution.awaiting_stop
    assert execution.metadata('prepare', 11., 11.3) is None
    execution.start('B2', 12.)
    assert execution.metadata('prepare', 12., 12.3)['id'] != first['id']


def install_interaction(run, monkeypatch, *, phase='prepare', source_age=.25):
    engine = run.tasks.engine
    engine.contact_profile = p = profile()
    engine.execution = InteractionExecution(p, 100.)
    engine.execution.start('B1', run.clock[0])
    engine.velocity_limits = dict(max_vx=p.max_forward_mps, max_vy=p.max_lateral_mps, max_wz=p.max_yaw_rps)
    engine.limiter.limits = engine.velocity_limits
    engine.limiter.linear_acceleration = p.linear_acceleration_mps2
    engine.limiter.yaw_acceleration = p.yaw_acceleration_rps2
    command_packets = []

    def update(snapshot, now):
        if engine.execution.active:
            engine.interaction_command = engine.execution.metadata(phase, now, snapshot.lidar_at+source_age)
        else:
            engine.interaction_command = None
        return np.array([.1, 0., 0.])

    def submit(velocity, generated_at, expires_at, *, interaction=None):
        command_packets.append(dict(velocity=velocity, generated_at=generated_at,
                                    expires_at=expires_at, interaction=interaction))
        return len(command_packets)

    monkeypatch.setattr(run.tasks, 'update', update)
    monkeypatch.setattr(run.motion, 'submit', submit)
    return SimpleNamespace(engine=engine, packets=command_packets)


def test_controller_submits_original_source_expiry_and_execution_identity(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, source_age=.20)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    packet = fixture.packets[-1]
    assert packet['interaction']['id'] == fixture.engine.execution.identity+':0'
    assert packet['expires_at'] == pytest.approx(current.lidar_at+.2)
    assert packet['interaction']['source_expires_at'] == current.lidar_at+.2
    assert any(e['event'] == 'interaction_command' for e in run.session.items)


def test_controller_drops_contact_command_if_computation_crosses_source_expiry(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, source_age=.20)
    original = run.tasks.update

    def delayed(snapshot, now):
        velocity = original(snapshot, now)
        run.clock[0] += .21
        return velocity

    monkeypatch.setattr(run.tasks, 'update', delayed)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert not fixture.packets and run.tasks.state == 'paused'
    assert run.controller.task_error == 'decision_expired'


def test_skill_completion_requires_matching_stop_before_scope_and_limits_restore(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch)
    execution = fixture.engine.execution
    execution.completed()
    previous_limits = dict(fixture.engine.velocity_limits)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    token = run.controller.planning_hold['token']
    assert 'interaction_completed' in token
    assert not fixture.packets and execution.active
    assert fixture.engine.velocity_limits == previous_limits
    run.feedback('stopping', token)
    assert execution.active and fixture.engine.velocity_limits == previous_limits
    run.feedback('stopped', 'planning_hold:old:interaction_completed')
    assert execution.active and run.controller.planning_hold['stopped_at'] is None
    run.feedback('stopped', token)
    assert not execution.active and fixture.engine.interaction_command is None
    assert fixture.engine.velocity_limits == dict(max_vx=run.tasks.config['max_forward_speed'],
                                                  max_vy=run.tasks.config['max_lateral_speed'],
                                                  max_wz=run.tasks.config['max_yaw_speed'])
    assert fixture.engine.limiter.linear_acceleration == run.tasks.config['linear_acceleration']
    assert fixture.engine.limiter.yaw_acceleration == run.tasks.config['yaw_acceleration']
    np.testing.assert_array_equal(fixture.engine.limiter.command, np.zeros(3))
    # Fresh observations after physical stop and after preparation are both required.
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 1
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 2 and not fixture.packets
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert fixture.packets[-1]['interaction'] is None


def test_mid_skill_planning_hold_uses_new_segment_with_same_contact_deadline(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, phase='engage')
    current = run.sample()
    run.controller.step(current, run.clock[0])
    first = fixture.packets[-1]['interaction']
    run.controller._hold('control_budget')
    token = run.controller.planning_hold['token']
    run.sample(.10)
    run.feedback('stopped', token)
    assert fixture.engine.execution.segment == 1
    assert fixture.engine.execution.deadline == first['deadline']
    assert fixture.engine.velocity_limits['max_vx'] == profile().max_forward_mps
    for _ in range(3):
        current = run.sample()
        run.controller.step(current, run.clock[0])
    resumed = fixture.packets[-1]['interaction']
    assert run.motion.arm_count == 2
    assert resumed['id'] != first['id']
    assert resumed['deadline'] == first['deadline']
    assert resumed['source_expires_at'] > first['source_expires_at']


def test_deadline_expired_while_planning_stopped_never_rearms(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, phase='engage')
    current = run.sample()
    run.controller.step(current, run.clock[0])
    run.controller._hold('control_budget')
    token = run.controller.planning_hold['token']
    run.sample(11.)
    run.feedback('stopped', token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 1 and run.tasks.state == 'paused'
    assert run.controller.task_error == 'interaction_duration_elapsed'


def test_passage_route_yields_before_global_search_or_guidance_mutation(run, monkeypatch):
    import deploy.app.navigation as navigation
    engine = run.tasks.engine
    searched = []
    passage = SimpleNamespace(entry_reached=True, target=(1., 0.),
                              path=lambda *args: searched.append(args))
    engine.passage = passage
    engine.phase = 'decide'
    monkeypatch.setattr(navigation, 'decision_recovery_required', lambda *args: False)
    monkeypatch.setattr(navigation, 'decision_buffer_recovery', lambda *args, **kwargs: None)
    seen = []

    def boundary(reason):
        seen.append(reason)
        if reason == 'passage_route':
            raise navigation.PlanningRequired(reason)

    # Permit the independently guarded buffer query to reach the passage
    # branch, then verify that its own A* never starts with motion enabled.
    engine.before_planning = boundary
    with pytest.raises(navigation.PlanningRequired, match='passage_route'):
        engine._decide(None)
    assert 'decision_buffer_recovery' in seen
    assert not searched and engine.passage is passage and engine.phase == 'decide'


def test_pending_contact_stop_cannot_advance_navigation_or_choose_a_new_box(run, monkeypatch):
    engine = run.tasks.engine
    engine.execution = InteractionExecution(profile(), 100.)
    engine.execution.start('B1', run.clock[0])
    identity = engine.execution.identity
    engine.execution.completed()
    engine.phase = 'decide'
    monkeypatch.setattr(engine, '_decide', lambda *_: pytest.fail('Decision advanced before interaction stop'))
    current = run.sample()
    command = engine.update(current.observation, run.clock[0])
    np.testing.assert_array_equal(command, np.zeros(3))
    assert engine.execution.identity == identity and engine.execution.awaiting_stop


def test_contact_completed_while_disarmed_reuses_the_confirmed_stop(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch)
    engine = fixture.engine
    ordinary_update = run.tasks.update
    completed = False

    def finish_during_planning(snapshot, now):
        nonlocal completed
        if not completed:
            completed = True
            engine.execution.completed()
            engine.skill = None
            engine.phase = 'decide'
            return np.zeros(3)
        return ordinary_update(snapshot, now)

    monkeypatch.setattr(run.tasks, 'update', finish_during_planning)
    run.controller._hold('skill_verify_passage')
    token = run.controller.planning_hold['token']
    run.sample(.1)
    run.feedback('stopped', token)
    current = run.sample()
    run.controller.step(current, run.clock[0])
    assert completed and not engine.execution.active
    assert run.motion.arm_count == 1 and not fixture.packets
    for _ in range(2):
        current = run.sample()
        run.controller.step(current, run.clock[0])
    assert run.motion.arm_count == 2 and run.motion.stops == [token]
    assert fixture.packets[-1]['interaction'] is None


def test_ten_hz_async_geometry_refresh_does_not_force_a_stop_each_tick(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, source_age=.25)
    update = run.tasks.update

    def bounded_compute(snapshot, now):
        velocity = update(snapshot, now)
        run.clock[0] += .005
        return velocity

    monkeypatch.setattr(run.tasks, 'update', bounded_compute)
    for _ in range(12):
        current = run.sample(.1)
        # One navigation tick of fitting/queue latency; capture time is retained.
        current = replace(current, lidar_at=current.captured_at-.1)
        run.controller.step(current, run.clock[0])
        assert run.controller.planning_hold is None
        assert fixture.packets[-1]['expires_at'] == pytest.approx(current.lidar_at+.25)
    assert len(fixture.packets) == 12 and run.tasks.state == 'active'
    assert not run.motion.stops


def test_async_refresh_that_overruns_previous_lease_is_discarded(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, source_age=.25)
    update = run.tasks.update
    cost = [.005]

    def compute(snapshot, now):
        velocity = update(snapshot, now)
        run.clock[0] += cost[0]
        return velocity

    monkeypatch.setattr(run.tasks, 'update', compute)
    current = run.sample(.1)
    current = replace(current, lidar_at=current.captured_at-.1)
    run.controller.step(current, run.clock[0])
    assert len(fixture.packets) == 1
    previous_expiry = fixture.packets[-1]['expires_at']
    current = run.sample(.1)
    current = replace(current, lidar_at=current.captured_at-.1)
    cost[0] = .06  # Unexpected computation spike; new source remains fresh.
    run.controller.step(current, run.clock[0])
    assert previous_expiry < run.clock[0] < current.lidar_at+.25
    assert len(fixture.packets) == 1 and run.tasks.state == 'paused'
    assert run.controller.task_error == 'control_deadline_missed'


def test_async_refresh_still_stops_before_a_known_slow_control_cycle(run, monkeypatch):
    fixture = install_interaction(run, monkeypatch, source_age=.25)
    current = run.sample(.1)
    current = replace(current, lidar_at=current.captured_at-.1)
    run.controller.step(current, run.clock[0])
    run.controller.last_decision_cost = .04
    current = run.sample(.1)
    current = replace(current, lidar_at=current.captured_at-.1)
    run.controller.step(current, run.clock[0])
    assert len(fixture.packets) == 1
    assert run.controller.planning_hold['reason'] == 'control_budget'
