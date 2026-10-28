"""Independent high-level SportClient writer. Never releases MCF or publishes LowCmd."""
import fcntl
import math
import multiprocessing as mp
import os
import time
from .safety import CommandGuard


class UnitreeSport:
    def __init__(self, robot, control):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        self.control, self.robot = control, robot
        self.state = None
        self.state_at = self.remote_at = 0.
        self.takeover = False
        self.remote_keys = 0
        ChannelFactoryInitialize(0, robot['network_interface'])
        self.client = SportClient()
        self.client.SetTimeout(.15)
        self.client.Init()
        self.state_sub = ChannelSubscriber(robot['sport_state_topic'], SportModeState_)
        self.state_sub.Init(self._state, 1)
        self.remote_sub = ChannelSubscriber(robot['low_state_topic'], LowState_)
        self.remote_sub.Init(self._remote, 1)

    def _state(self, state):
        self.state = state
        self.state_at = time.monotonic()

    def _remote(self, state):
        data = bytes(state.wireless_remote)
        if len(data) >= 4:
            self.remote_at = time.monotonic()
            self.remote_keys = int.from_bytes(data[2:4], 'little')
            if self.takeover_held:
                self.takeover = True

    @property
    def takeover_held(self):
        return bool(getattr(self, 'remote_keys', 0) & int(self.robot['takeover_button_mask']))

    def telemetry(self, now):
        state = self.state
        return {
            'keys': getattr(self, 'remote_keys', 0),
            'state_code': None if state is None else int(state.error_code),
            'state_age_s': None if not self.state_at else max(0., now-self.state_at),
            'remote_age_s': None if not self.remote_at else max(0., now-self.remote_at),
            'state_at': self.state_at, 'remote_at': self.remote_at, 'received_at': now,
            'velocity': None if state is None else [_finite(v) for v in state.velocity],
            'yaw_speed': None if state is None else _finite(state.yaw_speed),
            'healthy': self.healthy(now), 'stationary': self.stopped(now),
            'takeover': bool(getattr(self, 'takeover', False)), 'takeover_held': self.takeover_held,
        }

    def healthy(self, now):
        if self.state is None or now-self.state_at > .5 or now-self.remote_at > .5:
            return False
        q = self.state.imu_state.quaternion
        # This telemetry field can carry an FSM state (100 on this Go2),
        # unlike the RPC return code. Missing/unknown states remain blocked.
        return (self.state.error_code in self.robot.get('sport_state_motion_codes', ())
                and len(q) == 4 and all(math.isfinite(x) for x in q)
                and abs(sum(float(x)**2 for x in q)-1.) < .1
                and all(math.isfinite(float(x)) for x in self.state.velocity)
                and math.isfinite(float(self.state.yaw_speed))
                and 1-2*(q[1]**2+q[2]**2) >= math.cos(math.radians(self.control['max_tilt_degrees'])))

    def stopped(self, now):
        if not self.healthy(now):
            return False
        speed = sum(float(v)**2 for v in self.state.velocity)**.5
        return speed <= self.control['stop_speed'] and abs(self.state.yaw_speed) <= self.control['stop_yaw_speed']

    def stationary_feedback_time(self, now):
        return self.state_at if self.stopped(now) else None

    def move(self, velocity):
        code = self.client.Move(*map(float, velocity))
        if code != 0:
            raise RuntimeError(f'SportClient.Move failed: {code}')

    def stop(self):
        code = self.client.StopMove()
        if code != 0:
            raise RuntimeError(f'SportClient.StopMove failed: {code}')


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _telemetry(sdk, now):
    # Offline fake SDKs need only the original motion contract.
    if hasattr(sdk, 'telemetry'):
        return sdk.telemetry(now)
    return {'healthy': sdk.healthy(now), 'stationary': sdk.stopped(now),
            'keys': getattr(sdk, 'remote_keys', 0), 'state_code': None,
            'state_age_s': None, 'remote_age_s': None, 'velocity': None, 'yaw_speed': None,
            'takeover': bool(sdk.takeover), 'takeover_held': bool(getattr(sdk, 'takeover_held', False)),
            'received_at': now}


def _worker(pipe, robot, control, sdk_factory=UnitreeSport, test_mode=False):
    """One SDK writer; a test deadline and command TTL are enforced here."""
    control = dict(control)
    if test_mode:
        for key, limit in (('max_vx', .05), ('max_vy', .03), ('max_wz', .15)):
            control[key] = min(control[key], limit)
    lock = open(f'/tmp/inter-nav-motion-{os.getuid()}.lock', 'a')
    guard = CommandGuard(control)
    sdk = None
    stopping = None
    quit_requested = False
    last_stop = 0.
    stop_fault_reported = False
    stationary_since = None
    last_stop_error = None
    move_reported = stop_reported = False
    peak_speed = peak_yaw_speed = None

    def send(event):
        try:
            pipe.send(event)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def report(state, detail='', **facts):
        send(dict(state=state, detail=detail, **facts))

    def evidence(now):
        return dict(telemetry=_telemetry(sdk, now), peak_speed=peak_speed,
                    peak_yaw_speed=peak_yaw_speed) if test_mode else {}
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sdk = sdk_factory(robot, control)
        report('manual', 'MCF unchanged; joystick settings unchanged')
        while True:
            now = time.monotonic()
            was_armed = guard.armed
            # Bound draining so command floods cannot postpone watchdog checks.
            for _ in range(32):
                if not pipe.poll():
                    break
                try:
                    packet = pipe.recv()
                except EOFError:
                    quit_requested = True
                    guard.stop('parent_disconnected')
                    break
                now = time.monotonic()
                operation = packet['op']
                if operation == 'status':
                    send({'kind': 'reply', 'request_id': packet.get('request_id'),
                          'telemetry': _telemetry(sdk, now)})
                elif operation == 'arm':
                    reply = {'request_id': packet.get('request_id'), 'accepted': False}
                    duration = packet.get('duration_s')
                    invalid_duration = duration is not None and (
                        type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0)
                    if guard.armed or stopping is not None:
                        report('rejected', 'Motion is active or still stopping', **reply)
                    elif invalid_duration:
                        report('rejected', 'duration_s must be finite and positive', **reply)
                    elif test_mode and not 0 < int(robot.get('takeover_button_mask', 0)) <= 65535:
                        report('rejected', 'Read remote keys and configure takeover_button_mask before testing motion', **reply)
                    elif not sdk.stopped(now):
                        report('rejected', 'Need fresh telemetry and a stationary robot', **reply)
                    elif getattr(sdk, 'takeover_held', False):
                        report('rejected', 'Release the takeover button before arming', **reply)
                    elif sdk.takeover:
                        # Acknowledgement after release requires another action to move.
                        sdk.takeover = False
                        report('rejected', 'Takeover latch cleared; press the action key again to arm', **reply)
                    else:
                        duration = min(1., 1. if duration is None else duration) if test_mode else None
                        guard.arm(now, duration)
                        was_armed = True
                        move_reported = stop_reported = False
                        peak_speed = peak_yaw_speed = None
                        reply['accepted'] = True
                        report('active', duration_s=duration, **reply)
                elif operation == 'velocity':
                    guard.submit(packet['velocity'], packet['generated_at'], packet['expires_at'], now)
                elif operation in ('stop', 'close'):
                    guard.stop(packet.get('reason', operation))
                    quit_requested = quit_requested or operation == 'close'
                    # Act on stop before accepting any later command.
                    break
            now = time.monotonic()
            if guard.armed and (sdk.takeover or not sdk.healthy(now)):
                guard.stop('remote_takeover' if sdk.takeover else 'robot_state_stale_or_unsafe')
            if test_mode and (guard.armed or stopping is not None):
                sample = _telemetry(sdk, now)
                measured = sample.get('velocity')
                if measured and all(v is not None for v in measured):
                    speed = sum(v*v for v in measured)**.5
                    peak_speed = max(peak_speed or 0., speed)
                if sample.get('yaw_speed') is not None:
                    peak_yaw_speed = max(peak_yaw_speed or 0., abs(sample['yaw_speed']))
            velocity = guard.tick(now)
            if velocity is not None:
                try:
                    sdk.move(velocity)
                    if test_mode and not move_reported and any(abs(v) > 1e-6 for v in velocity):
                        move_reported = True
                        report('active', 'SportClient.Move returned successfully', event='move_accepted',
                               velocity=velocity.tolist(), **evidence(time.monotonic()))
                except Exception as error:
                    guard.stop(str(error))
            if was_armed and not guard.armed:
                stopping = time.monotonic()
                stop_fault_reported = False
                stationary_since = None
                last_stop_error = None
                report('stopping', guard.reason, **evidence(stopping))
            if stopping is not None:
                now = time.monotonic()
                if now-last_stop >= .1:
                    try:
                        sdk.stop()
                        if test_mode and not stop_reported:
                            stop_reported = True
                            report('stopping', guard.reason, event='stop_requested', **evidence(time.monotonic()))
                    except Exception as error:
                        if str(error) != last_stop_error:
                            report('stop_error', str(error))
                            last_stop_error = str(error)
                    last_stop = now
                now = time.monotonic()
                feedback_at = sdk.stationary_feedback_time(now)
                if feedback_at is None or feedback_at <= stopping:
                    stationary_since = None
                elif stationary_since is None:
                    stationary_since = feedback_at
                if stationary_since is not None and feedback_at-stationary_since >= control['stop_settle_s']:
                    stopping = None
                    report('stopped', guard.reason, **evidence(now))
                elif now-stopping > control['stop_timeout_s']:
                    if not stop_fault_reported:
                        report('fault', 'Stop not confirmed; use verified robot-side intervention', **evidence(now))
                        stop_fault_reported = True
                    # Do not claim handoff. Bounded exit cannot guarantee stop after network loss.
                    if quit_requested:
                        break
            if quit_requested and stopping is None:
                break
            time.sleep(1/control['poll_hz'])
    except BaseException as error:
        report('fault', f'{type(error).__name__}: {error}')
    finally:
        if sdk is not None and (guard.armed or stopping is not None):
            try:
                sdk.stop()
            except Exception:
                pass
        lock.close()
        pipe.close()


class MotionProcess:
    def __init__(self, robot, control, *, test_mode=False):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, robot, control),
                                       kwargs={'test_mode': test_mode}, daemon=False)
        self.state = 'starting'
        self.closing = False
        self._pending = []
        self._responses = {}
        self._request_id = 0
        self.process.start()
        child.close()
        if not self.pipe.poll(10):
            self.close()
            raise RuntimeError('Motion worker did not initialize')
        self._receive()
        if self.state == 'fault':
            events = self.close()
            raise RuntimeError(str(events))

    def _receive(self):
        # Requests are serialized by the console; bound late replies too.
        for _ in range(64):
            if not self.pipe.poll():
                break
            try:
                event = self.pipe.recv()
            except EOFError:
                break
            request_id = event.get('request_id')
            if request_id is not None:
                self._responses[request_id] = event
                if len(self._responses) > 16:
                    del self._responses[next(iter(self._responses))]
            if event.get('kind') != 'reply':
                self.state = event['state']
                self._pending.append(event)
        if not self.closing and not self.process.is_alive() and self.state != 'fault':
            self.state = 'fault'
            self._pending.append({'state': 'fault', 'detail': 'Motion worker exited'})

    def events(self):
        self._receive()
        events, self._pending = self._pending, []
        return events

    def _request(self, operation, timeout, **values):
        self._request_id += 1
        request_id = self._request_id
        try:
            self.pipe.send(dict(op=operation, request_id=request_id, **values))
        except (BrokenPipeError, EOFError, OSError) as error:
            raise RuntimeError('Motion worker disconnected') from error
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self._receive()
            if request_id in self._responses:
                return self._responses.pop(request_id)
            if not self.process.is_alive():
                break
            self.pipe.poll(min(.02, max(0., deadline-time.monotonic())))
        raise RuntimeError(f'Motion worker {operation} acknowledgement timed out')

    def telemetry(self, *, timeout_s=.25):
        """Request one snapshot; never enables output or consumes state events."""
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= .3:
            raise ValueError('telemetry timeout_s must be in (0, .3]')
        return self._request('status', timeout_s)['telemetry']

    def arm(self, *, duration_s=None):
        try:
            return bool(self._request('arm', 1., duration_s=duration_s)['accepted'])
        except RuntimeError:
            self.stop('arm_ack_timeout')
            return False

    def submit(self, velocity, generated_at, expires_at):
        self.pipe.send({'op': 'velocity', 'velocity': list(velocity),
                        'generated_at': generated_at, 'expires_at': expires_at})

    def stop(self, reason):
        if self.process.is_alive():
            try:
                self.pipe.send({'op': 'stop', 'reason': reason})
            except (BrokenPipeError, EOFError, OSError):
                pass

    def close(self):
        self.closing = True
        events = []
        if self.process.is_alive():
            try:
                self.pipe.send({'op': 'close'})
            except (BrokenPipeError, OSError):
                pass
            deadline = time.monotonic()+5
            while self.process.is_alive() and time.monotonic() < deadline:
                events.extend(self.events())
                self.process.join(.05)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1)
                events.append({'state': 'fault', 'detail': 'Writer unresponsive; physical stop NOT confirmed'})
        events.extend(self.events())
        self.pipe.close()
        return events
