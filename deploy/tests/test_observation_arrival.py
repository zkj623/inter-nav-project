"""Acquisition runs on LiDAR arrival; the consumer paces decisions itself.

Commands expire at lidar_at + max_data_age_s. A fixed 10 Hz poll against the
15.4 Hz Go2 cloud aged every command by up to one cloud period before planning.
"""
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import pytest

from deploy.app.observations import LiveObservations
from deploy.tests.test_live_observations import Sink, Source, wait_for
from deploy.tests.test_registered_boxes import packet


class ArrivingSource(Source):
    """Driver-like source: its own thread stores a frame every 65 ms (Go2 15.4 Hz)."""

    def __init__(self):
        super().__init__()
        self.ready = threading.Condition()
        self.frames = 0
        self.arrivals = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._drive, daemon=True)
        self.thread.start()

    def _drive(self):
        while not self.stop.wait(.065):
            with self.ready:
                self.frames += 1
                self.arrivals[self.frames] = time.monotonic()
                self.ready.notify_all()

    def wait_frame(self, seen, timeout):
        with self.ready:
            self.ready.wait_for(lambda: self.frames != seen, timeout=max(0., timeout))
            return self.frames

    def latest(self):
        with self.ready:
            index = self.frames
            arrived = self.arrivals.get(index)
        if not index:
            return None
        # Source capture precedes storage; the snapshot is assembled now.
        return replace(packet(index, arrived - .02, self.points), captured_at=time.monotonic())


@pytest.fixture
def arriving():
    source, sink = ArrivingSource(), Sink()
    reference = SimpleNamespace(metadata={}, apply=lambda snapshot, *_: snapshot)
    feed = LiveObservations(source, reference, sink, hz=10., max_age=.4)
    try:
        yield feed, source, sink
    finally:
        feed.close()
        source.stop.set()
        source.thread.join(1.)


def test_frames_are_published_on_arrival_not_on_a_fixed_poll(arriving):
    """A 10 Hz poll against 15.4 Hz scans aged every command by up to 65 ms."""
    feed, source, sink = arriving
    snapshot = wait_for(feed.latest)
    delays, seen = [], set()
    end = time.monotonic() + 1.5
    while time.monotonic() < end:
        feed.wait_next(snapshot, .5)
        woke = time.monotonic()
        snapshot = feed.latest()
        if snapshot.sequence not in seen:
            seen.add(snapshot.sequence)
            delays.append(woke - source.arrivals[snapshot.sequence])
    delays.sort()
    assert delays[len(delays)//2] < .015 and delays[int(.9*len(delays))] < .03
    # Every stored frame reaches navigation...
    assert len(seen) >= .8 * 1.5/.065
    # ...while the raw record keeps its average 10 Hz cadence.
    recorded = [item.captured_at for item in sink.observations[1:]]
    rate = (len(recorded)-1) / (recorded[-1]-recorded[0])
    assert 8.5 < rate < 11.5


def test_consumer_pacing_starts_on_a_scan_published_after_its_due_time(arriving):
    feed, _, _ = arriving
    snapshot = wait_for(feed.latest)
    due = time.monotonic() + .2
    assert feed.wait_next(snapshot, 1., not_before=due)
    woke = time.monotonic()
    assert feed.packet[3] >= due
    # The next scan after the due time, not a packet already waiting before it.
    assert woke - due < .065 + .03


def test_without_new_frames_the_worker_still_publishes_staleness(arriving):
    feed, source, _ = arriving
    snapshot = wait_for(feed.latest)
    source.stop.set()
    source.thread.join(1.)
    last = source.frames
    assert feed.wait_next(feed.latest(), .5)
    assert feed.latest().sequence == last  # republished, not renewed or invented
