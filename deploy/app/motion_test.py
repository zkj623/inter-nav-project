"""Short, explicitly triggered trials through the same guarded motion worker."""
import signal
import time

from .config import REPO, check_control_configuration
from .keyboard import Keyboard
from .session import Session
from deploy.robot.motion import MotionProcess, TEST_MAX_DURATION_S, TEST_SPEED_LIMITS

MENU = '\n[a]准备一次测试 [f]前行 [j]左转 [t]命令超时测试 [空格]停止 [q]退出\n'
VELOCITIES = {'f': (TEST_SPEED_LIMITS[0], 0., 0.),
              'j': (0., 0., TEST_SPEED_LIMITS[2]), 't': (.05, 0., 0.)}


class MotionTrial:
    """The console requests trials; the worker owns their speed and duration limits."""
    def __init__(self, config, motion, session):
        self.config, self.motion, self.session = config, motion, session
        self.ready_until = None
        self.started = None
        self.key = None
        self.number = 0
        self.last_stop_confirmed = None
        self.error = None

    @property
    def state(self):
        if self.started is not None:
            return 'running' if self.motion.state == 'active' else 'stopping'
        return 'ready' if self.ready_until is not None else 'idle'

    def operate(self, key, now):
        self.session.event('operator', key=key)
        if key == 'stop':
            self.ready_until = None
            self.motion.stop('operator_stop')
            return
        if key == 'a':
            self.ready_until = None
            if self.started is not None:
                raise ValueError('等待本次停止反馈，再按 a 准备下一次')
            check_control_configuration(self.config)
            status = self.motion.telemetry()
            self.session.event('motion_telemetry', **status)
            if not status['healthy'] or not status['stationary'] or status['takeover_held']:
                raise ValueError('准备失败：需要新鲜健康、静止的反馈，并松开接管键')
            self.ready_until = now+5.
            print('已准备；5 秒内按 f 前行、j 左转或 t 测超时。此时尚未发送运动指令。', flush=True)
            return
        if key not in VELOCITIES:
            return
        if self.ready_until is None or now > self.ready_until:
            self.ready_until = None
            raise ValueError('每次动作前先按 a；按住动作键不会连续执行')
        self.ready_until = None
        armed = self.motion.arm(duration_s=TEST_MAX_DURATION_S)
        # Finish the previous trial's events before assigning a new trial.
        # The correlated arm reply and any immediate fault are preserved too.
        self.events(self.motion.events())
        if not armed or self.motion.state != 'active':
            raise ValueError(f'测试未启用：{self.motion.state}；查看接管/停稳反馈，重新按 a 准备')
        self.number += 1
        self.started, self.key = time.monotonic(), key
        self.last_stop_confirmed = False
        self.session.event('trial_started', trial=self.number, key=key,
                           velocity=VELOCITIES[key], max_duration_s=TEST_MAX_DURATION_S,
                           watchdog_test=key == 't', hardware_validated=False)
        print(f'测试 {self.number} 开始；worker 在 {TEST_MAX_DURATION_S:g} 秒截止时请求停止，空格或配置接管键可提前停止。', flush=True)

    def events(self, events):
        for event in events:
            self.session.event('motion_state', **event)
            state = event['state']
            if state == 'fault':
                self.error = event['detail']
            if state in ('stopping', 'stopped', 'fault', 'stop_error', 'rejected'):
                print(f"运动={state} 原因={event['detail']}", flush=True)
            if state in ('stopping', 'fault', 'stop_error'):
                self.ready_until = None
            if state == 'stopped':
                self.last_stop_confirmed = True
                self.started = None
                self.key = None
                print('已收到停稳反馈；请结合现场动作和记录核对结果。下一次先按 a。', flush=True)

    def tick(self, now):
        if self.ready_until is not None and now > self.ready_until:
            self.ready_until = None
        if self.started is None or self.motion.state != 'active':
            return
        # Stop feeding the watchdog without disconnecting SDK/DDS. This is not a network-cut test.
        if self.key == 't' and now-self.started >= .12:
            return
        ttl = self.config['runtime']['control']['command_ttl_s']
        self.motion.submit(VELOCITIES[self.key], now, now+ttl)


def run(args, config):
    # An unset mask permits button inspection, never a trial. No ROS/SLAM/model construction here.
    check_control_configuration(config, require_takeover=False)
    session = Session(config, REPO, port=args.session_port, mode='motion_test')
    keyboard = motion = trial = None
    quitting = False
    reason, error = 'operator_exit', None
    handlers = {}

    def on_signal(signum, frame):
        nonlocal quitting, reason
        quitting, reason = True, f'signal_{signum}'

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, on_signal)
        motion = MotionProcess(config['robot'], config['runtime']['control'], test_mode=True)
        trial = MotionTrial(config, motion, session)
        keyboard = Keyboard(menu=MENU, commands={'a', 'f', 'j', 't', 'q'})
        print(f'单项运动验收，会话={session.info["session_id"]}；数据服务={session.url}', flush=True)
        print('另开终端 python record.py 保存事件。此模式不采集点云、不建图、不做避障。', flush=True)
        print('在平整空地、现场有人操作遥控器时测试。MCF 保持运行，启动不动。', flush=True)
        print(f'f 前行上限 {TEST_SPEED_LIMITS[0]:g} m/s；j 左转上限 {TEST_SPEED_LIMITS[2]:g} rad/s；'
              f'每次最多 {TEST_MAX_DURATION_S:g} 秒，较低的设备限速仍生效。', flush=True)
        print(f'接管 mask={config["robot"]["takeover_button_mask"]}；为 0 时只读按键和状态，拒绝动作。', flush=True)
        last_status = float('-inf')
        while not quitting:
            trial.events(motion.events())
            now = time.monotonic()
            operation = keyboard.poll()
            if operation:
                key, _ = operation
                if key == 'q':
                    session.event('operator', key=key)
                    break
                try:
                    trial.operate(key, now)
                except (ValueError, RuntimeError) as exc:
                    session.event('trial_rejected', reason=str(exc))
                    print(str(exc), flush=True)
            trial.tick(time.monotonic())
            # Blocking status requests never share time with active command feeding.
            if trial.started is None and now-last_status >= .5:
                try:
                    status = motion.telemetry()
                    session.event('motion_telemetry', **status)
                    print(f'keys={status["keys"]} state={status["state_code"]} '
                          f'healthy={status["healthy"]} stationary={status["stationary"]} '
                          f'velocity={status["velocity"]} yaw={status["yaw_speed"]}', flush=True)
                except RuntimeError as exc:
                    session.event('telemetry_error', reason=str(exc))
                    print(str(exc), flush=True)
                last_status = now
            session.set_status(trial.state, 'motion_test', motion.state)
            time.sleep(.05)
    except Exception as exc:
        reason, error = 'runtime_error', f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    finally:
        if motion:
            try:
                events = motion.close()
                if trial:
                    trial.events(events)
                for event in events:
                    if event['state'] == 'fault':
                        error = event['detail']
            except Exception as exc:
                error = f'motion cleanup: {exc}; physical stop NOT confirmed'
        if keyboard:
            try:
                keyboard.close()
            except Exception as exc:
                error = f'keyboard cleanup: {exc}'
        if trial:
            error = error or trial.error
        try:
            session.close({'reason': reason, 'error': error, 'completed': error is None,
                           'hardware_validated': False, 'trials': trial.number if trial else 0,
                           'last_autonomous_stop_confirmed': trial.last_stop_confirmed if trial else None})
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return 0 if error is None else 2
