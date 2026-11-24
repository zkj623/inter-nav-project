"""Continuous read-only acquisition; planning never owns the sensor cadence.

This worker owns coordinate initialization and perception state. Navigation reads
one detached observation/evidence pair. Reset is a generation request, so an
in-flight fit can never restore evidence from the preceding task. No map, skill
or motion writer is touched here.

Each LiDAR frame is processed when the driver stores it (``wait_frame``), not
on a fixed poll. Commands expire at ``lidar_at + max_data_age_s``; a 10 Hz poll
against the 15.4 Hz Go2 cloud added up to one cloud period of age before
planning even began. The raw record keeps an average ``hz`` cadence, and the
consumer paces its own decisions with ``wait_next(not_before=...)``.
"""
from copy import deepcopy
import json
import threading
import time


class LiveObservations:
    def __init__(self, sensors, reference, session, *, hz, max_age, perception=None, window_observer=None,
                 point_ceiling=None, world_boundary=None):
        self.sensors, self.reference, self.session = sensors, reference, session
        self.period, self.max_age = 1. / hz, max_age
        self.perception = perception
        self.point_ceiling = point_ceiling
        self.world_boundary = world_boundary
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.closed = threading.Event()
        self.generation = 0
        self.packet = self.delivered = None
        self.last_error = None
        self.failure = None
        self.window_worker = None
        if window_observer is not None:
            from deploy.perception.window_stream import WindowWorker
            event = 'box_window_result' if perception is not None else 'perception_result'
            self.window_worker = WindowWorker(window_observer, session, event=event)
        self.thread = threading.Thread(target=self._work, name='go2-observations', daemon=True)
        self.thread.start()

    def latest(self):
        with self.lock:
            self.delivered = self.packet
            return None if self.delivered is None else self.delivered[1]

    def update(self, snapshot, now):
        """Tasks' perception interface, pinned to the snapshot read this cycle."""
        with self.lock:
            packet = self.delivered
            if packet is None or packet[0] != self.generation or packet[1] is not snapshot:
                return (), None
            return packet[2]

    @property
    def last_delivery(self):
        """Diagnostics paired with the snapshot last handed to the main loop."""
        with self.lock:
            return self.delivered[4] if self.delivered is not None else None

    @property
    def perception_error(self):
        with self.lock:
            return self.delivered[5] if self.delivered is not None else None

    def reset(self):
        with self.lock:
            self.generation += 1
            self.packet = self.delivered = None

    def wait_next(self, snapshot, timeout, not_before=None):
        """Wake the consumer on fresh evidence, rather than drift one tick behind.

        With ``not_before``, only a packet published at or after that monotonic
        time qualifies: the consumer keeps its decision cadence yet starts each
        decision on a just-arrived scan instead of one that waited for it.
        """
        def ready():
            packet = self.packet
            return (self.closed.is_set() or self.failure is not None
                    or (packet is not None and packet[1] is not snapshot
                        and (not_before is None or packet[3] >= not_before)))
        with self.changed:
            return self.changed.wait_for(ready, timeout=max(0., timeout))

    def _work(self):
        applied_generation = -1
        last_delivery = None
        wait_frame = getattr(self.sensors, 'wait_frame', None)
        seen = None
        record_due = float('-inf')
        try:
            while not self.closed.is_set():
                if wait_frame is not None:
                    # Without a new frame for a whole period, still publish so
                    # staleness and adapter errors reach the consumer.
                    seen = wait_frame(seen, self.period)
                    if self.closed.is_set():
                        break
                started = time.monotonic()
                with self.lock:
                    generation = self.generation
                if generation != applied_generation:
                    if self.perception is not None:
                        self.perception.reset()
                    applied_generation, last_delivery = generation, None
                snapshot = self.sensors.latest()
                delivery = None
                result, error = ((), None), getattr(self.sensors, 'last_error', None)
                if snapshot is not None:
                    try:
                        initializing = self.reference.metadata is None
                        snapshot = self.reference.apply(snapshot, time.monotonic(), self.max_age)
                        if initializing and self.reference.metadata is not None:
                            self.session.event('coordinate_frame_initialized', **self.reference.metadata)
                            print('实验坐标系已固定：起点 XY=(0,0)，X=初始前方，Y=初始左侧，Z=距地面高度。', flush=True)
                            print('点导航输入本轮 experiment 坐标；暂停、换任务不重设原点。', flush=True)
                            print(json.dumps(self.reference.metadata, ensure_ascii=False), flush=True)
                        # The raw record keeps its average cadence even when
                        # the diagnostic window receives scans more frequently.
                        if snapshot.captured_at >= record_due:
                            record_due = max(record_due, snapshot.captured_at - .5*self.period) + self.period
                            self.session.record_observation(snapshot)
                        snapshot = snapshot.navigation_view(self.point_ceiling, self.world_boundary)
                        if self.window_worker is not None:
                            self.window_worker.submit(snapshot)
                        if self.perception is not None:
                            boxes, evidence = self.perception.update(snapshot, time.monotonic())
                            frozen = evidence.freeze() if hasattr(evidence, 'freeze') else deepcopy(evidence)
                            result = (boxes, frozen)
                            delivery = self.perception.last_delivery
                            if delivery is not None and delivery is not last_delivery:
                                self.session.event('perception_result', **delivery)
                                last_delivery = delivery
                    except (ValueError, RuntimeError) as exc:
                        if hasattr(self.perception, 'invalidate'):
                            self.perception.invalidate(str(exc))
                        snapshot, error = None, str(exc)
                if snapshot is None and self.window_worker is not None:
                    self.window_worker.submit(None, error)
                with self.lock:
                    if generation == self.generation:
                        fit_error = getattr(self.perception, 'last_error', None)
                        self.packet = (generation, snapshot, result, time.monotonic(), delivery, fit_error)
                        self.last_error = error
                        self.changed.notify_all()
                if wait_frame is None:
                    self.closed.wait(max(0., self.period - (time.monotonic() - started)))
        except Exception as exc:
            with self.changed:
                self.packet = None
                self.last_error = f'Observation worker failed: {type(exc).__name__}: {exc}'
                self.failure = self.last_error
                self.changed.notify_all()
            self.session.event('observation_error', reason=self.last_error)
        finally:
            if self.perception is not None:
                self.perception.close()

    def close(self):
        with self.changed:
            self.closed.set()
            self.changed.notify_all()
        self.thread.join(timeout=2.)
        if self.window_worker is not None:
            self.window_worker.close()
        if self.thread.is_alive():
            raise RuntimeError('Observation worker did not finish')
        if self.failure:
            raise RuntimeError(self.failure)
