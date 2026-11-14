"""Recorded stop handoff tests; synthetic task completion is not robot evidence."""
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import load
from deploy.interaction.replay import RecordedInteractionHandoff
from deploy.recording.replay import snapshots
from deploy.tests.make_fixture import make_fixture


class Perception:
    def __init__(self):
        self.calls = []
        self.last_delivery = None
        self.closed = False

    def update(self, snapshot, now):
        self.calls.append(snapshot.sequence)
        return (snapshot.sequence,), self

    def close(self):
        self.closed = True


class Engine:
    def __init__(self):
        self.execution = SimpleNamespace(awaiting_stop=False, identity='synthetic-skill')
        self.layers = SimpleNamespace(scan_count=1, last_scan=-1,
                                     statistics=lambda: {'synthetic': True}, save=lambda path: None)
        self.events = deque()
        self.observations = []
        self.handoffs = 0

    def observe(self, observation, now, boxes=()):
        self.observations.append((now, boxes))
        self.layers.last_scan = observation['sensors']['lidar']['physics_step']

    def interaction_stopped(self):
        self.handoffs += 1
        self.execution.awaiting_stop = False


class Tasks:
    def __init__(self, config=None, perception=None, interaction=None):
        self.engine = Engine()
        self.perception = perception
        self.calls = []
        self.map_ready = False
        self.last_lidar = self.last_mapping_at = None
        self.result = None
        self.state = 'observe'

    def start(self, mode, goal, now):
        self.state = 'active'

    def observe_only(self, snapshot, now):
        boxes = () if self.perception is None else self.perception.update(snapshot, now)[0]
        observation = {**snapshot.observation, 'velocity': tuple(snapshot.velocity),
                       'lidar_captured_at': snapshot.lidar_at}
        self.engine.observe(observation, now, boxes)
        self.map_ready = True
        self.last_lidar = self.engine.layers.last_scan
        self.last_mapping_at = snapshot.lidar_at

    def update(self, snapshot, now):
        self.calls.append(now)
        if len(self.calls) == 1:
            # Only terminal skill orchestration is synthetic. All stop samples
            # below are actual inputs to the replay gate under test.
            self.engine.execution.awaiting_stop = True
        return np.array([.1, 0., 0.])


@pytest.fixture
def source(tmp_path):
    original = next(snapshots(make_fixture(tmp_path/'recording')))

    def sample(delta, *, linear=(0., 0., 0.), angular=(0., 0., 0.), complete=True,
               pose_delta=None, lidar_delta=None):
        now = original.captured_at+delta
        observation = dict(original.observation)
        if complete:
            observation['body_linear_velocity'] = np.array(linear)
            observation['body_angular_velocity'] = np.array(angular)
        else:
            observation.pop('body_linear_velocity', None)
            observation.pop('body_angular_velocity', None)
        observation['sensors'] = dict(observation['sensors'])
        observation['sensors']['lidar'] = dict(observation['sensors']['lidar'], physics_step=int(delta*1000))
        return replace(original, observation=observation, captured_at=now, sequence=int(delta*1000),
                       pose_at=now if pose_delta is None else original.captured_at+pose_delta,
                       lidar_at=now if lidar_delta is None else original.captured_at+lidar_delta)
    return sample


def test_recorded_motion_keeps_task_waiting_but_updates_original_provider_and_map(source):
    cfg = load()
    gate = RecordedInteractionHandoff(cfg['runtime']['control'], .4)
    perception = Perception()
    tasks = Tasks(perception=perception)
    first = source(0.)
    velocity, event = gate.update(tasks, first, first.captured_at)
    assert velocity is None and event['event'] == 'interaction_stop_handoff_waiting'
    for delta in (.1, .2, .3):
        packet = source(delta, linear=(.1, 0., 0.))
        assert gate.update(tasks, packet, packet.captured_at) == (None, None)
    assert len(tasks.calls) == 1 and tasks.engine.handoffs == 0
    assert perception.calls == [100, 200, 300]
    assert [boxes for _, boxes in tasks.engine.observations] == [(100,), (200,), (300,)]
    assert tasks.map_ready and tasks.last_lidar == 300
    assert gate.reason == 'recorded_robot_still_moving'


def test_distinct_recorded_stationary_feedback_restores_handoff_before_next_task_step(source):
    gate = RecordedInteractionHandoff(load()['runtime']['control'], .4)
    tasks = Tasks()
    for delta in (0., .1, .2):
        packet = source(delta)
        _, event = gate.update(tasks, packet, packet.captured_at)
    assert gate.pending and tasks.engine.handoffs == 0
    packet = source(.41)
    velocity, event = gate.update(tasks, packet, packet.captured_at)
    assert velocity is None and event['event'] == 'recorded_stop_handoff'
    assert event['evidence_source'] == 'recorded_body_motion' and event['sent_to_robot'] is False
    assert not gate.pending and tasks.engine.handoffs == 1 and len(tasks.calls) == 1
    packet = source(.5)
    velocity, event = gate.update(tasks, packet, packet.captured_at)
    assert np.any(velocity) and event is None and len(tasks.calls) == 2


@pytest.mark.parametrize('options,reason', [({'complete': False}, 'complete_recorded_body_motion_missing'),
    ({'angular': (.06, 0., 0.)}, 'recorded_robot_still_moving'),
    ({'linear': (0., 0., .04)}, 'recorded_robot_still_moving'),
    ({'lidar_delta': 0.}, 'awaiting_fresh_recorded_motion')])
def test_missing_full_feedback_motion_on_any_axis_or_pre_handoff_scan_never_continues(source, options, reason):
    gate = RecordedInteractionHandoff(load()['runtime']['control'], .4)
    tasks = Tasks()
    packet = source(0.)
    gate.update(tasks, packet, packet.captured_at)
    for delta in (.1, .2, .4, .6):
        packet = source(delta, **options)
        gate.update(tasks, packet, packet.captured_at)
    assert gate.pending and gate.reason == reason
    assert tasks.engine.handoffs == 0 and len(tasks.calls) == 1


def test_repeated_pose_and_gap_do_not_manufacture_continuous_recorded_stop(source):
    gate = RecordedInteractionHandoff(load()['runtime']['control'], .4)
    tasks = Tasks()
    for delta, pose in [(0., 0.), (.1, .1), (.2, .1), (.3, .1), (.4, .1)]:
        packet = source(delta, pose_delta=pose)
        gate.update(tasks, packet, packet.captured_at)
    assert gate.pending and tasks.engine.handoffs == 0
    # A fresh sample after a long data gap starts a new settling interval.
    packet = source(.7)
    gate.update(tasks, packet, packet.captured_at)
    assert gate.stationary_since == packet.pose_at and tasks.engine.handoffs == 0


@pytest.mark.parametrize('mode,expected_code,expected_reason,expected_task_calls', [
    ('moving', 2, 'interaction_stop_handoff_required', 1),
    ('missing', 2, 'interaction_stop_handoff_required', 1),
    ('stopped', 0, 'recording_exhausted', 2)])
def test_runtime_replay_loop_reports_unresolved_handoff_instead_of_false_completion(
        source, tmp_path, monkeypatch, mode, expected_code, expected_reason, expected_task_calls):
    from deploy.app import runtime
    from deploy.recording import replay
    config = load()
    records = []
    for delta in (0., .1, .2, .41, .5):
        records.append(source(delta, complete=mode != 'missing',
                              linear=(.1, 0., 0.) if mode == 'moving' else (0., 0., 0.)))
    tasks = Tasks(perception=Perception())

    class Recorder:
        directory = tmp_path
        def __init__(self):
            self.events, self.summary = [], None
        def event(self, kind, **facts):
            self.events.append({'event': kind, **facts})
        def close(self, summary):
            self.summary = summary

    recorder = Recorder()
    monkeypatch.setattr(runtime, 'Tasks', lambda *a, **kw: tasks)
    monkeypatch.setattr(runtime, 'create_perception', lambda *a, **kw: tasks.perception)
    monkeypatch.setattr(runtime, '_recorder', lambda *a, **kw: recorder)
    monkeypatch.setattr(replay, 'snapshots', lambda path: iter(records))
    args = SimpleNamespace(replay=tmp_path, task='navigate', goal=(1., 0.))
    assert runtime.run_replay(args, config) == expected_code
    assert recorder.summary['reason'] == expected_reason
    assert recorder.summary['completed'] is (mode == 'stopped')
    assert recorder.summary['sent_to_robot'] is False
    assert recorder.summary['hardware_validated'] is False
    assert len(tasks.calls) == expected_task_calls and tasks.perception.closed
    assert len([e for e in recorder.events if e['event'] == 'recorded_stop_handoff']) == (mode == 'stopped')
    assert all(e.get('sent_to_robot', False) is False for e in recorder.events)
