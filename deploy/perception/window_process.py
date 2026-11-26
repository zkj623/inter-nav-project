"""Run the read-only window tracker beside navigation, on the same coordinates and clock."""
from dataclasses import replace
import multiprocessing as mp
from queue import Empty, Full
import signal
import threading
import time

from .window_observer import WindowBoxObserver
from .window_stream import WindowWorker


class _ResultPipe:
    def __init__(self, channel):
        self.channel = channel

    def event(self, name, **result):
        self.channel.send(('result', result))


def _start_message(channel, stop):
    channel.send(('ready', None))
    while not stop.is_set():
        if channel.poll(.1):
            return channel.recv()
    return None


def _ros(config, channel, stop, robot, timebase, point_height, boundary=None):
    from deploy.robot.sensors import RosSensors
    sensors = RosSensors(dict(robot, camera_mode='unavailable'), timebase=timebase, node_name='inter_nav_box_window')
    worker = None
    try:
        # The parent fixes the experiment origin once. Never initialize a second map here.
        reference = _start_message(channel, stop)
        if reference is None:
            return
        if reference.metadata is None:
            raise ValueError('Box tracking requires the initialized navigation frame')
        worker = WindowWorker(WindowBoxObserver(config, live=True, world_boundary=boundary), _ResultPipe(channel))
        seen = None
        while not stop.is_set():
            seen = sensors.wait_frame(seen, .1)
            if worker.failure:
                raise RuntimeError(worker.failure)
            snapshot = sensors.latest()
            if snapshot is None:
                worker.submit(None, sensors.last_error or 'no_valid_observation')
                continue
            try:
                snapshot = reference.apply(snapshot, time.monotonic(), robot['max_data_age_s'])
                worker.submit(snapshot.navigation_view(point_height, boundary))
            except ValueError as exc:
                worker.submit(None, str(exc))
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            sensors.close()


def _recording(config, channel, stop, directory, point_height, boundary=None):
    from deploy.recording.replay import ReplayClock, snapshots
    clock = ReplayClock()
    anchor = _start_message(channel, stop)
    if anchor is None:
        return
    clock.first, clock.wall_start = anchor
    worker = WindowWorker(WindowBoxObserver(config, live=True, world_boundary=boundary), _ResultPipe(channel), clock=clock.now)
    try:
        for packet in snapshots(directory):
            if not clock.wait(packet.captured_at, stop):
                break
            if worker.failure:
                raise RuntimeError(worker.failure)
            worker.submit(packet.navigation_view(point_height, boundary))
    finally:
        worker.close(drain=not stop.is_set())


def _navigation_scans(config, channel, stop, scans, boundary):
    worker = WindowWorker(WindowBoxObserver(config, live=True, world_boundary=boundary), _ResultPipe(channel))
    try:
        channel.send(('ready', None))
        while not stop.is_set():
            if worker.failure:
                raise RuntimeError(worker.failure)
            try:
                error, snapshot = scans.get(timeout=.1)
            except Empty:
                continue
            if error is not None:
                worker.submit(None, error)
            if snapshot is not None:
                worker.submit(snapshot)
    finally:
        worker.close()


def _run(source, config, channel, stop, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        source(config, channel, stop, *args)
        channel.send(('done', None))
    except Exception as exc:
        channel.send(('failed', f'{type(exc).__name__}: {exc}'))
    finally:
        channel.close()


class WindowProcess:
    """Fit in a separate process. Only small measured results cross the result pipe."""
    @classmethod
    def from_navigation_scans(cls, config, sink, event, boundary):
        scans = mp.get_context('spawn').Queue(maxsize=2)
        try:
            process = cls(_navigation_scans, (scans, boundary), config, sink, event)
        except BaseException:
            scans.close()
            raise
        process.scans = scans
        process.started = True  # Input packets already have the navigation frame and clock.
        return process

    @classmethod
    def live(cls, config, sink, event, robot, timebase, point_height, boundary=None):
        if timebase is None:
            raise ValueError('Box tracking requires the navigation time reference')
        return cls(_ros, (robot, timebase, point_height, boundary), config, sink, event)

    @classmethod
    def replay(cls, config, sink, directory, point_height, boundary=None, *, box_memory=None):
        return cls(_recording, (directory, point_height, boundary), config, sink, 'box_window_result', box_memory)

    def __init__(self, source, args, config, sink, event, box_memory=None):
        self.sink, self.event = sink, event
        self.box_memory = box_memory
        self.failure = None
        self.started = False
        self.scans = None
        self.pending_scan_error = None
        ctx = mp.get_context('spawn')
        self.channel, child = ctx.Pipe()
        self.stop = ctx.Event()
        self.process = ctx.Process(target=_run, args=(source, config, child, self.stop, args),
                                   name='go2-box-window', daemon=True)
        try:
            self.process.start()
            child.close()
            if not self.channel.poll(12.):
                raise RuntimeError('Box tracker startup timed out')
            status, detail = self.channel.recv()
            if status != 'ready':
                raise RuntimeError(f'Box tracker startup failed: {detail}')
        except BaseException:
            child.close()
            if self.process.pid is not None:
                self._stop_process()
            self.channel.close()
            raise
        self.reader = threading.Thread(target=self._collect, name='box-window-results', daemon=True)
        self.reader.start()

    def begin(self, reference):
        if self.started:
            return
        self.channel.send(reference)
        self.started = True

    def submit(self, snapshot, error=None):
        """Never wait for fitting or pipe serialization on the navigation input thread."""
        if self.stop.is_set():
            return
        if snapshot is None:
            self.pending_scan_error = error or 'no_valid_observation'
        elif 'camera' in snapshot.observation['sensors']:
            # Only LiDAR is used here. Keep RGB-D arrays off the box worker's pipe.
            sensors = {'lidar': snapshot.observation['sensors']['lidar']}
            snapshot = replace(snapshot, observation={**snapshot.observation, 'sensors': sensors}, camera_at=None)
        try:
            self.scans.put_nowait((self.pending_scan_error, snapshot))
        except Full:
            return  # Next accepted packet retains its actual timestamp and any invalidation.
        self.pending_scan_error = None

    def _collect(self):
        try:
            while True:
                kind, result = self.channel.recv()
                if kind == 'done':
                    return
                if kind == 'failed':
                    raise RuntimeError(result)
                if self.box_memory is not None:
                    self.box_memory.observe(result, result['published_at'])
                self.sink.event(self.event, **result)
        except Exception as exc:
            self.failure = f'Box tracker stopped: {type(exc).__name__}: {exc}'
            self.stop.set()
            try:
                self.sink.event(self.event, geometry_source='window_two_faces', detections=[], error=self.failure)
            except Exception as delivery_error:
                self.failure += f'; reporting the failure also failed: {delivery_error}'

    def _stop_process(self):
        self.stop.set()
        self.process.join(timeout=2.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.)

    def close(self, *, drain=False):
        try:
            if drain and self.started:
                self.process.join(timeout=7.)
                if self.process.is_alive():
                    self.failure = 'Box tracker did not finish the recording'
            self._stop_process()
            self.reader.join(timeout=2.)
        finally:
            self.channel.close()
            if self.scans is not None:
                self.scans.cancel_join_thread()
                self.scans.close()
        if self.failure or self.reader.is_alive():
            raise RuntimeError(self.failure or 'Box result reader did not finish')
