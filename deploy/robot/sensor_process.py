"""Read high-rate ROS inputs off the navigation interpreter's control thread."""
import multiprocessing as mp
import signal


def _read_sensors(channel, stop, config):
    from .sensors import RosSensors

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    source = None
    try:
        source = RosSensors(config)
        channel.send(('ready', (source.config, source.timebase)))
        seen = None
        while not stop.is_set():
            seen = source.wait_frame(seen, .1)
            if not stop.is_set():
                snapshot = source.latest()
                channel.send(('scan', (snapshot, source.last_error)))
    except Exception as exc:
        if not stop.is_set():
            channel.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        try:
            if source is not None:
                source.close()
        finally:
            channel.close()


class RosSensorProcess:
    """The single observation reader consumes original timestamps and frame errors."""
    def __init__(self, config):
        ctx = mp.get_context('spawn')
        self.channel, writer = ctx.Pipe(duplex=False)
        self.stop = ctx.Event()
        self.process = ctx.Process(target=_read_sensors, args=(writer, self.stop, config),
                                   name='go2-navigation-sensors', daemon=True)
        self.sample, self.last_error = None, None
        self.revision = 0
        try:
            self.process.start()
            writer.close()
            if not self.channel.poll(20.):
                raise RuntimeError('Navigation sensor startup timed out')
            self.config, self.timebase = self._receive('ready')
        except BaseException:
            writer.close()
            self.close()
            raise

    def _receive(self, expected):
        try:
            kind, value = self.channel.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('Navigation sensor process disconnected') from exc
        if kind == 'error':
            raise RuntimeError(f'Navigation sensor process: {value}')
        if kind != expected:
            raise RuntimeError(f'Unexpected navigation sensor message {kind!r}, expected {expected!r}')
        return value

    def wait_frame(self, seen, timeout):
        if self.channel.poll(max(0., timeout)):
            # A busy consumer may have missed arrivals. Prefer the newest
            # packet already waiting, without projecting an old scan again.
            while True:
                self.sample, self.last_error = self._receive('scan')
                self.revision += 1
                if self.sample is None or not self.channel.poll(0.):
                    break
        return self.revision

    def latest(self):
        return self.sample

    def close(self):
        self.stop.set()
        self.channel.close()
        if self.process.pid is not None:
            self.process.join(timeout=2.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1.)
