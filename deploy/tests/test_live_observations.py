"""Acquisition/tracking must keep running while the planner is stopped/busy."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import pytest

from deploy.app.observations import LiveObservations
from deploy.perception.registered_boxes import RegisteredBoxPerception
from deploy.tests.test_registered_boxes import config, packet, cloud


class Source:
    last_error = None

    def __init__(self):
        self.count = 0
        self.points = cloud()
        self.stalled = False
        self.previous = None

    def latest(self):
        if self.stalled:
            return self.previous
        self.count += 1
        now = time.monotonic()
        self.previous = replace(packet(self.count, now - .08, self.points), captured_at=now)
        return self.previous


class Sink:
    def __init__(self):
        self.observations, self.events = [], []

    def record_observation(self, snapshot):
        self.observations.append(snapshot)

    def event(self, name, **details):
        self.events.append((name, details))


def wait_for(callback, timeout=3.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = callback()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError('No matching observation before deadline')


def read(feed):
    snapshot = feed.latest()
    if snapshot is None:
        return None
    boxes, evidence = feed.update(snapshot, time.monotonic())
    return (snapshot, boxes, evidence) if boxes else None


@pytest.fixture
def live():
    source, sink = Source(), Sink()
    provider = RegisteredBoxPerception(config(), synchronous=True)
    # Fixtures already carry an experiment transform; actual conversion is
    # covered by the real run/view/record entrypoint test.
    reference = SimpleNamespace(metadata={}, apply=lambda snapshot, *_: snapshot)
    feed = LiveObservations(source, reference, sink, hz=10., max_age=.4, perception=provider)
    try:
        yield feed, source, sink, provider
    finally:
        feed.close()


def test_slow_planner_keeps_identity_and_full_rate_recording(live):
    feed, source, sink, _ = live
    first, _, frozen = wait_for(lambda: read(feed))
    stamp = frozen.snapshot(first.captured_at)['B1']['latest_at']
    count = len(sink.observations)
    # No main-thread polling for longer than the 250 ms geometry lease.
    time.sleep(.55)
    current, boxes, evidence = read(feed)
    assert len(sink.observations) >= count + 4
    assert current.lidar_at > first.lidar_at + .3
    assert boxes[0].id == 'B1'
    assert evidence.permission('B1', time.monotonic()) is not None
    assert frozen.snapshot(first.captured_at)['B1']['latest_at'] == stamp
    assert frozen.permission('B1', time.monotonic()) is None  # no renewal of old evidence
    assert evidence.snapshot(time.monotonic())['B1']['invalid_reason'] is None


def test_real_sensor_gap_still_loses_identity_and_needs_reset(live):
    feed, source, _, _ = live
    wait_for(lambda: read(feed))
    source.stalled = True
    time.sleep(.4)
    source.stalled = False
    wait_for(lambda: source.count > 4)
    assert read(feed) is None
    feed.reset()
    assert read(feed) is None
    assert wait_for(lambda: read(feed))[2].permission('B1', time.monotonic()) is not None


def test_reset_during_fit_drops_old_generation(live, monkeypatch):
    feed, _, _, provider = live
    wait_for(lambda: read(feed))
    entered, release = threading.Event(), threading.Event()
    original = provider._fit

    def held(frame):
        entered.set()
        assert release.wait(2)
        return original(frame)

    monkeypatch.setattr(provider, '_fit', held)
    try:
        assert entered.wait(1)
        feed.reset()
        assert feed.latest() is None
        release.set()
        # Reset must establish independent post-reset scans, not reuse the fit.
        current, _, evidence = wait_for(lambda: read(feed))
        assert evidence.permission('B1', current.captured_at).evidence_frames[0].startswith('lidar:2:')
    finally:
        release.set()


def test_navigation_reads_one_pair_even_when_worker_publishes_again(live):
    feed, source, _, _ = live
    snapshot, boxes, evidence = wait_for(lambda: read(feed))
    delivery = feed.last_delivery
    assert delivery['scan_id'] == snapshot.observation['sensors']['lidar']['physics_step']
    wait_for(lambda: source.count > snapshot.sequence + 1)
    other_boxes, other_evidence = feed.update(snapshot, time.monotonic())
    assert other_boxes is boxes and other_evidence is evidence
    assert feed.last_delivery is delivery
    newer = feed.latest()
    assert newer.sequence > snapshot.sequence
    assert feed.update(snapshot, time.monotonic()) == ((), None)


def test_wait_wakes_on_new_evidence_and_invalid_transform_revokes_identity(live):
    feed, _, _, _ = live
    snapshot, _, _ = wait_for(lambda: read(feed))
    feed.wait_next(snapshot, 1.)
    assert feed.latest().sequence > snapshot.sequence
    original = feed.reference.apply

    def invalid(*args):
        raise ValueError('Source coordinate frame changed')

    feed.reference.apply = invalid
    wait_for(lambda: feed.last_error == 'Source coordinate frame changed')
    feed.reference.apply = original
    wait_for(lambda: feed.last_error is None)
    assert read(feed) is None
    feed.reset()
    assert wait_for(lambda: read(feed))[1]


def test_actual_skill_preparation_keeps_live_tracking_and_execution_budget(monkeypatch):
    from deploy.app.navigation import NavigationEngine
    from deploy.tests.test_registered_interaction_integration import settings, scan, observed_layout
    navigation, interaction, tracking = settings()

    class IntegrationSource:
        count = 0
        last_error = None

        def latest(self):
            self.count += 1
            return scan(self.count, time.monotonic())

    provider = RegisteredBoxPerception(tracking, synchronous=True)
    sink = Sink()
    feed = LiveObservations(IntegrationSource(), SimpleNamespace(metadata={}, apply=lambda s, *_: s),
                            sink, hz=10., max_age=.4, perception=provider)
    engine = NavigationEngine(navigation, interaction=interaction)
    engine.set_goal((0., 2., .4), time.monotonic())
    try:
        snapshot, boxes, evidence = wait_for(lambda: read(feed))
        engine.observe(snapshot.observation, snapshot.captured_at, boxes)
        observed_layout(engine)
        engine.update(snapshot.observation, snapshot.captured_at, boxes, evidence)
        assert engine.phase == 'skill'
        deadline = engine.execution.deadline
        select = engine.skill._select_stage

        def slow_stage(context):
            time.sleep(.3)  # Ensure the computation exceeds a 250 ms lease on any CPU.
            return select(context)

        monkeypatch.setattr(engine.skill, '_select_stage', slow_stage)
        feed.wait_next(snapshot, 1.)
        snapshot, boxes, evidence = read(feed)
        engine.update(snapshot.observation, snapshot.captured_at, boxes, evidence)
        assert engine.skill.state == 'navigate_to_stage'
        assert evidence.permission('B1', time.monotonic()) is None
        # A new measured fit survives preparation; neither old evidence nor the
        # absolute skill budget is extended to disguise the slow computation.
        snapshot, boxes, evidence = read(feed)
        assert evidence.permission('B1', time.monotonic()) is not None
        engine.update(snapshot.observation, snapshot.captured_at, boxes, evidence)
        assert engine.status == 'running' and engine.execution.deadline == deadline
    finally:
        feed.close()


def test_worker_failure_clears_publication_and_is_reported_on_close():
    def fail():
        raise OSError('synthetic sensor reader failure')

    feed = LiveObservations(SimpleNamespace(latest=fail), None, Sink(), hz=10., max_age=.4)
    try:
        wait_for(lambda: feed.failure)
        assert feed.latest() is None
    finally:
        with pytest.raises(RuntimeError, match='synthetic sensor reader failure'):
            feed.close()
