"""Explicit orchestration: sensors -> original algorithms -> guarded SportClient."""
import importlib.util
import json
import signal
import time
from pathlib import Path
from .config import REPO, check_motion_configuration, check_task_inputs
from .controller import NavigationController
from .tasks import Tasks
from deploy.recording.recorder import Recorder
from deploy.robot.observation import ExperimentFrame
from deploy.robot.safety import check_observation, checked_velocity


def check(config):
    required = ('numpy', 'cv2', 'yaml', 'requests', 'scipy')
    optional = ('rclpy', 'tf2_ros', 'sensor_msgs', 'nav_msgs', 'unitree_sdk2py')
    status = {name: importlib.util.find_spec(name) is not None for name in required+optional}
    Tasks(config['navigation'])  # Validate mapping config without Isaac or robot.
    try:
        check_motion_configuration(config)
        motion = 'configured; hardware NOT verified by this check'
    except ValueError as error:
        motion = str(error)
    print(json.dumps({'check': 'offline_only', 'dependencies': status, 'motion': motion,
                      'planning_frame': ExperimentFrame.name, 'source_frame': config['robot']['map_frame'],
                      'camera_mode': config['robot']['camera_mode'],
                      'geometry_calibrated': config['robot']['ground_z'] is not None and
                                             config['robot']['lidar_origin_offset_base_m'] is not None,
                      'MCF': 'kept running; no mode release or joystick switching'}, indent=2))
    return 0 if all(status[name] for name in required) else 2


def _recorder(args, config, mode):
    root = args.output or Path(config['runtime']['recording']['root'])
    if not root.is_absolute():
        root = REPO / root
    recorder = Recorder(root, config, mode, REPO)
    print(f'运行记录：{recorder.directory}', flush=True)
    return recorder


def run_replay(args, config):
    from deploy.recording.replay import snapshots
    tasks = Tasks(config['navigation'])
    recorder = _recorder(args, config, 'offline_replay_no_motion')
    started = False
    reason, error = 'recording_exhausted', None
    count = 0
    try:
        for snapshot in snapshots(args.replay):
            now = snapshot.captured_at
            if snapshot.geometry_calibrated:
                check_observation(snapshot, now, config['robot'], config['runtime']['control'],
                                  config['navigation']['mapping'])
            else:
                snapshot.require_fresh(now, config['robot']['max_data_age_s'])
            if not started and args.task == 'navigate':
                check_task_inputs(config['robot'], snapshot)
                tasks.start('navigate', (*args.goal, float(snapshot.observation['position'][2])), now)
                started = True
            velocity = tasks.update(snapshot, now)
            recorder.event('decision', sequence=snapshot.sequence, state=tasks.state,
                           requested_velocity=None if velocity is None else checked_velocity(
                               velocity, config['runtime']['control']), sent_to_robot=False)
            count += 1
            if tasks.state == 'completed':
                reason = 'task_completed'
                break
        if count == 0:
            raise ValueError('Empty recording')
    except Exception as exc:
        error, reason = str(exc), 'replay_error'
    finally:
        try:
            if tasks.map_ready:
                tasks.component.mapping.save(str(recorder.directory / 'final_map'))
        except Exception as exc:
            error = f'Map save failed: {exc}'
        summary = {'reason': reason, 'error': error, 'task_result': tasks.result,
                   'decisions': count, 'map_created': tasks.map_ready, 'sent_to_robot': False, 'hardware_validated': False,
                   'completed': error is None}
        recorder.close(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if error is None else 2


def run_live(args, config):
    import sys
    from .keyboard import Keyboard
    from .session import Session
    from deploy.visualization.sources import SdkVideo
    from deploy.robot.sensors import RosSensors
    from deploy.robot.motion import MotionProcess
    from deploy.perception.client import AsyncPerception, AsyncScorer
    if args.allow_motion:
        check_motion_configuration(config)
    robot, nav, runtime = config['robot'], config['navigation'], config['runtime']
    control = runtime['control']
    reference = ExperimentFrame(0. if args.demo else robot['ground_z'])
    session = Session(config, REPO, port=args.session_port, demo=args.demo,
                      mode='synthetic_demo' if args.demo else ('live_motion_enabled' if args.allow_motion else 'live_observe_only'))
    print(f'本轮会话：{session.info["session_id"]}；数据服务：{session.url}', flush=True)
    print('另开终端运行 python view.py 查看三幅图；python record.py 保存实验。', flush=True)
    sensors = motion = keyboard = perception = scorer = tasks = video = controller = None
    snapshot = None
    started_at = time.monotonic()
    quitting = False
    reason, error = 'operator_exit', None
    handlers = {}
    def on_signal(signum, frame):
        nonlocal quitting, reason
        reason = f'signal_{signum}'
        quitting = True
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, on_signal)
    try:
        if args.demo:
            from .demo import DemoSensors
            sensors = DemoSensors(session)
        else:
            topic = args.rgb_topic or robot['rgb_topic']
            sensors = RosSensors(robot, session.rgb if args.camera == 'ros' else None,
                                 topic if args.camera == 'ros' else None)
            if args.camera == 'sdk' and robot['camera_mode'] != 'rgbd':
                video = SdkVideo(session.frames, robot['network_interface'], 2.)
            elif args.camera == 'none':
                session.frames.error('rgb', 'RGB preview disabled (--camera none)')
        # rclpy.init can replace SIGINT; restore coordinated shutdown after init.
        for signum in handlers:
            signal.signal(signum, on_signal)
        if not args.demo and robot['camera_mode'] == 'rgbd':
            perception = AsyncPerception(runtime['perception'])
        if perception and runtime['topology_scorer']['enabled']:
            scorer = AsyncScorer(runtime['topology_scorer'])
        tasks = Tasks(nav, perception, scorer)
        if args.allow_motion:
            motion = MotionProcess(robot, control)
        keyboard = Keyboard() if not args.demo or sys.stdin.isatty() else None
        controller = NavigationController(config, tasks, motion, session, perception)
        last_status = 0.
        print(('合成演示：未连接机器人。' if args.demo else 'MCF 保持运行。默认观察；') + ('运动需按任务键启动。' if motion else '本次会话禁止运动。'), flush=True)
        print(f"来源坐标系={'demo_odom' if args.demo else robot['map_frame']}；等待有效观测初始化 experiment；相机={robot['camera_mode']}", flush=True)
        if not args.demo and (robot['ground_z'] is None or robot['lidar_origin_offset_base_m'] is None):
            print('地面/雷达原点未标定：本次可记录观测，导航地图暂不生成。', flush=True)
        while not quitting:
            now = time.monotonic()
            if args.demo and args.demo_seconds and now-started_at >= args.demo_seconds:
                reason = 'demo_finished'
                break
            operation = keyboard.poll() if keyboard else None
            if motion:
                controller.motion_events(motion.events())
            snapshot = sensors.latest()
            # Sensor callbacks may advance while keyboard/events/locks run.
            # Compare against time AFTER acquiring the observation.
            now = time.monotonic()
            observation_error = None
            if snapshot is not None:
                try:
                    initializing = reference.metadata is None
                    snapshot = reference.apply(snapshot, now, robot['max_data_age_s'])
                    if initializing and reference.metadata is not None:
                        session.event('coordinate_frame_initialized', **reference.metadata)
                        print('实验坐标系已固定：起点 XY=(0,0)，X=初始前方，Y=初始左侧，Z=距地面高度。', flush=True)
                        print('点导航输入本轮 experiment 坐标；暂停、换任务不重设原点。', flush=True)
                        print(json.dumps(reference.metadata, ensure_ascii=False), flush=True)
                except ValueError as exc:
                    observation_error, snapshot = str(exc), None
            if operation:
                key, value = operation
                if key == 'q':
                    session.event('operator', key=key, value=value)
                    quitting = True
                    continue
                controller.operate(key, value, snapshot, now)
            step_error = controller.step(snapshot, now)
            observation_error = observation_error or step_error
            current_error = sensors.last_error or observation_error
            session.update(snapshot, tasks, motion.state if motion else 'disabled', current_error)
            if now-last_status >= 2 and (keyboard is None or keyboard.editing is None):
                print(f'状态={tasks.state} 任务={tasks.mode} 运动={motion.state if motion else "disabled"}'
                      f' 观测={"ready" if snapshot else "waiting"} 建图={tasks.map_ready} 最近错误={current_error}', flush=True)
                last_status = now
            time.sleep(max(0, 1/nav['tick_hz']-(time.monotonic()-now)))
    except Exception as exc:
        reason, error = 'runtime_error', f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    finally:
        # Stop the only command writer first, then close every observer even if one cleanup fails.
        if motion:
            try:
                events = motion.close()
                if controller:
                    controller.motion_events(events)
                for event in events:
                    if event['state'] == 'fault':
                        error = event['detail']
            except Exception as exc:
                error = f'motion cleanup: {exc}; physical stop NOT confirmed'
                session.event('cleanup_error', reason=error)
        for name, resource in (('keyboard', keyboard), ('perception', perception), ('scorer', scorer),
                               ('video', video), ('sensors', sensors)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    error = f'{name} cleanup: {exc}'
                    session.event('cleanup_error', reason=error)
        if controller:
            error = error or controller.motion_error
        try:
            session.close({'reason': reason, 'error': error, 'task_result': tasks.result if tasks else None,
                           'completed': error is None, 'hardware_validated': False,
                           'planning_frame': reference.name if reference.metadata else robot['map_frame'],
                           'coordinate_transform': reference.metadata, 'global_localization_verified': False,
                           'map_created': bool(tasks and tasks.map_ready), 'demo': args.demo,
                           'last_autonomous_stop_confirmed': controller.stop_confirmed if controller else None})
        except Exception as exc:
            error = f'session cleanup: {exc}'
            print(error, flush=True)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return 0 if error is None else 2
