"""Read recorded observations; this module has no robot communication path."""
import json
from pathlib import Path
from queue import Full, Queue
import threading
import time
import numpy as np
from deploy.robot.observation import Snapshot
from .recorder import unpack


class ReplayClock:
    """Replay original arrival intervals; keep scans and feedback on their original time axis."""
    def __init__(self):
        self.first = self.last = self.wall_start = None
        self.max_lateness = 0.
        self.packets = 0

    def now(self):
        return self.first + (time.monotonic()-self.wall_start)

    def wait(self, arrival, stop):
        if self.first is None:
            self.first, self.wall_start = arrival, time.monotonic()
        remaining = arrival-self.now()
        if stop.wait(max(0., remaining)):
            return False
        self.last = arrival
        self.max_lateness = max(self.max_lateness, self.now()-arrival)
        self.packets += 1
        return True

    def play(self, packets, worker=None, *, on_start=None):
        """Pace mapping packets; share the first arrival's clock with an independent reader."""
        queue = Queue(maxsize=8)
        stop, end = threading.Event(), object()

        def enqueue(item):
            while not stop.is_set():
                try:
                    queue.put(item, timeout=.1)
                    return
                except Full:
                    continue

        def receive():
            try:
                for snapshot in packets:
                    if not self.wait(snapshot.captured_at, stop):
                        break
                    if self.packets == 1 and on_start is not None:
                        on_start((self.first, self.wall_start))
                    if worker is not None:
                        worker.submit(snapshot)
                    enqueue(snapshot)
            except Exception as exc:
                enqueue(exc)
            finally:
                enqueue(end)

        reader = threading.Thread(target=receive, name='replay-sensors', daemon=True)
        reader.start()
        try:
            while True:
                item = queue.get()
                if item is end:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            reader.join()


def snapshots(directory):
    root = Path(directory).resolve()
    previous = None
    with (root / 'observations.jsonl').open() as stream:
        for line in stream:
            entry = json.loads(line)
            path = (root / entry['file']).resolve()
            if root not in path.parents:
                raise ValueError('Recording array path escapes run directory')
            with np.load(path, allow_pickle=False) as arrays:
                snapshot = Snapshot(**unpack(entry['snapshot'], arrays))
            if previous is not None and snapshot.captured_at <= previous:
                raise ValueError('Recording timestamps must increase')
            previous = snapshot.captured_at
            yield snapshot
