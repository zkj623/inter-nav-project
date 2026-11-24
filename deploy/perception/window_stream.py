"""Capture a source-timed LiDAR window while a separate worker fits its newest contents."""
from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Optional

import numpy as np

from .registered_boxes import RegisteredScanReader


WINDOW = 1.


@dataclass(frozen=True)
class ScanWindow:
    epoch: int
    frames: tuple
    ready: bool
    error: Optional[str] = None


class WindowScanBuffer:
    def __init__(self, config):
        self.reader = RegisteredScanReader(config)
        self.max_gap = config['max_age_s']
        self.frames = deque()
        self.epoch = 0
        self.started = self.last_now = self.error = None

    def clear(self, reason):
        self.epoch += 1
        self.frames.clear()
        self.started = None
        self.error = reason

    def append(self, snapshot, now, error=None):
        try:
            if snapshot is None:
                raise ValueError(error or 'no_valid_observation')
            if not math.isfinite(now) or self.last_now is not None and now < self.last_now:
                raise ValueError('Window observation clock moved backwards or is invalid')
            self.last_now = now
            frame = self.reader.capture(snapshot, now, self.epoch)
            if frame is None:
                return None
            q = np.asarray(snapshot.observation['orientation'], float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
                raise ValueError('Window observation requires a valid robot orientation')
            w, x, y, z = q / np.linalg.norm(q)
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        except (KeyError, TypeError, ValueError) as exc:
            if self.error == str(exc):
                return None
            self.clear(str(exc))
            return ScanWindow(self.epoch, (), False, self.error)
        stamp = frame.captured_at
        if self.frames and stamp-self.frames[-1][0].captured_at >= self.max_gap:
            self.clear('scan_gap')
        self.error = None
        if self.started is None:
            self.started = stamp
        while self.frames and self.frames[0][0].captured_at <= stamp-WINDOW:
            self.frames.popleft()
        self.frames.append((frame, yaw))
        return ScanWindow(self.epoch, tuple(self.frames), stamp-self.started >= WINDOW)


class WindowWorker:
    """One pending immutable window. Input scans and fitting have separate clocks of progress."""
    def __init__(self, observer, sink, *, clock=time.monotonic, event='box_window_result'):
        self.observer, self.sink, self.clock, self.event = observer, sink, clock, event
        self.changed = threading.Condition()
        self.pending = self.failure = None
        self.closing = self.discard = False
        self.thread = threading.Thread(target=self._work, name='go2-box-window', daemon=True)
        self.thread.start()

    def submit(self, snapshot, error=None):
        with self.changed:
            if self.closing or self.failure is not None:
                return
            window = self.observer.buffer.append(snapshot, self.clock(), error)
            if window is not None:
                self.pending = window
                self.changed.notify()

    def _work(self):
        try:
            while True:
                with self.changed:
                    self.changed.wait_for(lambda: self.pending is not None or self.closing)
                    if self.pending is None:
                        return
                    window, self.pending = self.pending, None
                self.observer.evaluate(window, self.clock())
                with self.changed:
                    if self.discard or window.epoch != self.observer.buffer.epoch:
                        continue
                    result = self.observer.last_delivery
                    if result is None:
                        result = dict(geometry_source='window_two_faces', detections=[], error=self.observer.last_error)
                    self.sink.event(self.event, published_at=self.clock(), **result)
        except Exception as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            self.sink.event(self.event, published_at=self.clock(), geometry_source='window_two_faces',
                            detections=[], error=self.failure)
        finally:
            self.observer.close()

    def close(self, *, drain=False):
        with self.changed:
            self.closing, self.discard = True, not drain
            if self.discard:
                self.pending = None
            self.changed.notify()
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            raise RuntimeError('Box window worker did not finish')
        if self.failure is not None:
            raise RuntimeError(self.failure)
