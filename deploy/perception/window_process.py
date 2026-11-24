"""Run the read-only window tracker beside navigation, on the same coordinates and clock."""
import multiprocessing as mp
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


def _ros(config, channel, stop, robot, timebase, ceiling):
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
        worker = WindowWorker(WindowBoxObserver(config, live=True), _ResultPipe(channel))
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
                worker.submit(snapshot.below_ceiling(ceiling))
            except ValueError as exc:
                worker.submit(None, str(exc))
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            sensors.close()


def _recording(config, channel, stop, directory, ceiling):
    from deploy.recording.replay import ReplayClock, snapshots
    clock = ReplayClock()
    anchor = _start_message(channel, stop)
    if anchor is None:
        return
    clock.first, clock.wall_start = anchor
    worker = WindowWorker(WindowBoxObserver(config, live=True), _ResultPipe(channel), clock=clock.now)
    try:
        for packet in snapshots(directory):
            if not clock.wait(packet.captured_at, stop):
                break
            if worker.failure:
                raise RuntimeError(worker.failure)
            worker.submit(packet.below_ceiling(ceiling))
    finally:
        worker.close(drain=not stop.is_set())


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
    """Only small measured results cross the pipe. The child owns its input stream."""
    @classmethod
    def live(cls, config, sink, event, robot, timebase, ceiling):
        if timebase is None:
            raise ValueError('Box tracking requires the navigation time reference')
        return cls(_ros, (robot, timebase, ceiling), config, sink, event)

    @classmethod
    def replay(cls, config, sink, directory, ceiling):
        return cls(_recording, (directory, ceiling), config, sink, 'box_window_result')

    def __init__(self, source, args, config, sink, event):
        self.sink, self.event = sink, event
        self.failure = None
        self.started = False
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

    def _collect(self):
        try:
            while True:
                kind, result = self.channel.recv()
                if kind == 'done':
                    return
                if kind == 'failed':
                    raise RuntimeError(result)
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
        if self.failure or self.reader.is_alive():
            raise RuntimeError(self.failure or 'Box result reader did not finish')
