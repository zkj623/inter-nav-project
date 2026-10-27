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
            if int.from_bytes(data[2:4], 'little') & int(self.robot['takeover_button_mask']):
                self.takeover = True

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


def _worker(pipe, robot, control, sdk_factory=UnitreeSport):
    """Owns the SDK and a host lock. Parent death also reaches the watchdog."""
    lock = open(f'/tmp/inter-nav-motion-{os.getuid()}.lock', 'a')
    guard = CommandGuard(control)
    sdk = None
    stopping = None
    quit_requested = False
    last_stop = 0.
    stop_fault_reported = False
    stationary_since = None
    last_stop_error = None
    def report(state, detail=''):
        try:
            pipe.send({'state': state, 'detail': detail})
        except (BrokenPipeError, EOFError, OSError):
            pass
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sdk = sdk_factory(robot, control)
        report('manual', 'MCF unchanged; joystick settings unchanged')
        while True:
            now = time.monotonic()
            was_armed = guard.armed
            while pipe.poll():
                try:
                    packet = pipe.recv()
                except EOFError:
                    quit_requested = True
                    guard.stop('parent_disconnected')
                    break
                operation = packet['op']
                if operation == 'arm':
                    if stopping is not None or not sdk.stopped(now):
                        report('rejected', 'Need fresh telemetry and a stationary robot')
                    elif sdk.takeover:
                        # Clear latch only on an explicit arm, but require another action.
                        sdk.takeover = False
                        report('rejected', 'Takeover latch cleared; verify remote released before retry')
                    else:
                        guard.arm(now)
                        was_armed = True
                        report('active')
                elif operation == 'velocity':
                    guard.submit(packet['velocity'], packet['generated_at'], packet['expires_at'], now)
                elif operation in ('stop', 'close'):
                    guard.stop(packet.get('reason', operation))
                    quit_requested = operation == 'close'
            if guard.armed and (sdk.takeover or not sdk.healthy(now)):
                guard.stop('remote_takeover' if sdk.takeover else 'robot_state_stale_or_unsafe')
            velocity = guard.tick(now)
            if velocity is not None:
                try:
                    sdk.move(velocity)
                except Exception as error:
                    guard.stop(str(error))
            if was_armed and not guard.armed:
                stopping = now
                stop_fault_reported = False
                stationary_since = None
                report('stopping', guard.reason)
            if stopping is not None:
                if now-last_stop >= .1:
                    try:
                        sdk.stop()
                    except Exception as error:
                        if str(error) != last_stop_error:
                            report('stop_error', str(error))
                            last_stop_error = str(error)
                    last_stop = now
                feedback_at = sdk.stationary_feedback_time(now)
                if feedback_at is None or feedback_at <= stopping:
                    stationary_since = None
                elif stationary_since is None:
                    stationary_since = feedback_at
                if stationary_since is not None and feedback_at-stationary_since >= control['stop_settle_s']:
                    stopping = None
                    report('stopped', guard.reason)
                elif now-stopping > control['stop_timeout_s']:
                    if not stop_fault_reported:
                        report('fault', 'Stop not confirmed; use verified robot-side intervention')
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
    def __init__(self, robot, control):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, robot, control), daemon=False)
        self.state = 'starting'
        self.closing = False
        self.process.start()
        child.close()
        if not self.pipe.poll(10):
            self.close()
            raise RuntimeError('Motion worker did not initialize')
        events = self.events()
        if self.state == 'fault':
            self.close()
            raise RuntimeError(str(events))

    def events(self):
        events = []
        while self.pipe.poll():
            try:
                event = self.pipe.recv()
            except EOFError:
                break
            self.state = event['state']
            events.append(event)
        if not self.closing and not self.process.is_alive() and self.state != 'fault':
            self.state = 'fault'
            events.append({'state': 'fault', 'detail': 'Motion worker exited'})
        return events

    def arm(self):
        self.pipe.send({'op': 'arm'})
        deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            if self.pipe.poll(.05):
                self.events()
                return self.state == 'active'
        return False

    def submit(self, velocity, generated_at, expires_at):
        self.pipe.send({'op': 'velocity', 'velocity': list(velocity),
                        'generated_at': generated_at, 'expires_at': expires_at})

    def stop(self, reason):
        if self.process.is_alive():
            self.pipe.send({'op': 'stop', 'reason': reason})

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
