"""Continuous read-only acquisition; planning never owns the sensor cadence.

This worker owns coordinate initialization and perception state. Navigation reads
one detached observation/evidence pair. Reset is a generation request, so an
in-flight fit can never restore evidence from the preceding task. No map, skill
or motion writer is touched here.
"""
from copy import deepcopy
import json
import threading
import time


class LiveObservations:
    def __init__(self, sensors, reference, session, *, hz, max_age, perception=None):
        self.sensors, self.reference, self.session = sensors, reference, session
        self.period, self.max_age = 1. / hz, max_age
        self.perception = perception
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.closed = threading.Event()
        self.generation = 0
        self.packet = self.delivered = None
        self.last_error = None
        self.failure = None
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

    def reset(self):
        with self.lock:
            self.generation += 1
            self.packet = self.delivered = None

    def wait_next(self, snapshot, timeout):
        """Wake the consumer on fresh evidence, rather than drift one tick behind."""
        with self.changed:
            self.changed.wait_for(lambda: self.closed.is_set() or self.failure is not None
                                  or (self.packet is not None and self.packet[1] is not snapshot),
                                  timeout=max(0., timeout))

    def _work(self):
        applied_generation = -1
        last_delivery = None
        try:
            while not self.closed.is_set():
                started = time.monotonic()
                with self.lock:
                    generation = self.generation
                if generation != applied_generation:
                    if self.perception is not None:
                        self.perception.reset()
                    applied_generation, last_delivery = generation, None
                snapshot = self.sensors.latest()
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
                        # Record before fitting/planning, at the acquisition cadence.
                        self.session.record_observation(snapshot)
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
                with self.lock:
                    if generation == self.generation:
                        self.packet = (generation, snapshot, result)
                        self.last_error = error
                        self.changed.notify_all()
                self.closed.wait(max(0., self.period - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.packet = None
                self.last_error = f'Observation worker failed: {type(exc).__name__}: {exc}'
                self.failure = self.last_error
                self.changed.notify_all()
            self.session.event('observation_error', reason=self.last_error)
        finally:
            if self.perception is not None:
                self.perception.close()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError('Observation worker did not finish')
        if self.failure:
            raise RuntimeError(self.failure)
