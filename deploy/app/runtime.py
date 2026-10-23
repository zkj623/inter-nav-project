"""Explicit orchestration: sensors -> original algorithms -> guarded SportClient."""
import importlib.util
import json
import signal
import time
from pathlib import Path
from .config import REPO, check_motion_configuration, check_task_inputs
from .tasks import Tasks
from deploy.recording.recorder import Recorder
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
                      'planning_frame': config['robot']['map_frame'],
                      'camera_mode': config['robot']['camera_mode'],
                      'geometry_calibrated': config['robot']['ground_z'] is not None and
                                             config['robot']['lidar_origin_offset_base_m'] is not None,
                      'MCF': 'kept running; no mode release or joystick switching'}, indent=2))
    return 0 if all(status[name] for name in required) else 2


def _recorder(args, config, mode):
    root = args.output or Path(config['runtime']['recording']['root'])
    if not root.is_absolute():
        root = REPO / root
    recorder = Recorder(root, config, mode, REPO, asynchronous=mode.startswith('live_'))
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
    from .keyboard import Keyboard
    from deploy.robot.sensors import RosSensors
    from deploy.robot.motion import MotionProcess
    from deploy.perception.client import AsyncPerception, AsyncScorer
    if args.allow_motion:
        check_motion_configuration(config)
    robot, nav, runtime = config['robot'], config['navigation'], config['runtime']
    control = runtime['control']
    recorder = _recorder(args, config, 'live_motion_enabled' if args.allow_motion else 'live_observe_only')
    sensors = motion = keyboard = perception = scorer = tasks = None
    quitting = False
    reason, error = 'operator_exit', None
    stop_confirmed = None
    handlers = {}
    def on_signal(signum, frame):
        nonlocal quitting, reason
        reason = f'signal_{signum}'
        quitting = True
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, on_signal)
    try:
        sensors = RosSensors(robot)
        if robot['camera_mode'] == 'rgbd':
            perception = AsyncPerception(runtime['perception'])
        if perception and runtime['topology_scorer']['enabled']:
            scorer = AsyncScorer(runtime['topology_scorer'])
        tasks = Tasks(nav, perception, scorer)
        if args.allow_motion:
            motion = MotionProcess(robot, control)
        keyboard = Keyboard()
        last_record = last_status = 0.
        perception_started = None
        last_perception_delivery = None
        print('MCF 保持运行。默认观察；' + ('运动需按任务键启动。' if motion else '本次会话禁止运动。'), flush=True)
        print(f"规划坐标系={robot['map_frame']}（不代表全局定位）；相机={robot['camera_mode']}", flush=True)
        if robot['ground_z'] is None or robot['lidar_origin_offset_base_m'] is None:
            print('地面/雷达原点未标定：本次可记录观测，导航地图暂不生成。', flush=True)
        while not quitting:
            now = time.monotonic()
            operation = keyboard.poll()
            if motion:
                for event in motion.events():
                    recorder.event('motion_state', **event)
                    if event['state'] == 'stopped':
                        stop_confirmed = True
                    if event['state'] in ('stopping', 'fault', 'stop_error') and tasks.state == 'active':
                        tasks.pause()
                    if event['state'] == 'fault':
                        print(f"运动状态异常：{event['detail']}", flush=True)
            snapshot = sensors.latest()
            if operation:
                key, value = operation
                recorder.event('operator', key=key, value=value)
                if key == 'q':
                    quitting = True
                    continue
                if key in ('p', 'prepare', 'stop', 'o', 'm'):
                    if motion:
                        motion.stop(key)
                    if key == 'p':
                        tasks.pause()
                    else:
                        tasks.cancel()
                elif key in ('navigate', 'explore', 'r'):
                    try:
                        if motion is None:
                            raise ValueError('本次仅观察。实机配置验证后使用 --allow-motion 启动')
                        if snapshot is None:
                            raise ValueError('等待有效定位和点云')
                        exploring = key == 'explore' or (key == 'r' and tasks.mode == 'explore')
                        check_task_inputs(robot, snapshot, exploring)
                        check_observation(snapshot, now, robot, control, nav['mapping'], camera=exploring)
                        if snapshot.observation['position'][2] < nav['safe_base_height']:
                            raise ValueError('机器人机身高度不足')
                        if key == 'navigate':
                            xy = tuple(float(item) for item in value.split(','))
                            if len(xy) != 2:
                                raise ValueError('目标格式为 x,y，例如 2.0,1.0')
                            tasks.start('navigate', (*xy, float(snapshot.observation['position'][2])), now)
                        elif key == 'explore':
                            tasks.start('explore', value.replace('_', ' '), now)
                            perception_started = now
                        else:
                            tasks.resume(now)
                            if exploring:
                                perception_started = now
                        if not motion.arm():
                            tasks.pause()
                            raise ValueError(f'运动未启用：{motion.state}；检查机器人状态和遥控接管键')
                        stop_confirmed = False
                        recorder.event('motion_state', state='active', detail='explicit operator arm')
                    except (ValueError, RuntimeError) as exc:
                        print(str(exc), flush=True)
                        recorder.event('task_rejected', reason=str(exc))
            if snapshot is not None:
                try:
                    if snapshot.geometry_calibrated:
                        check_observation(snapshot, now, robot, control, nav['mapping'],
                                          camera=tasks.state == 'active' and tasks.mode == 'explore')
                    else:
                        snapshot.require_fresh(now, robot['max_data_age_s'])
                    if tasks.state == 'active' and snapshot.observation['position'][2] < nav['fall_height']:
                        raise ValueError('Robot below fall-height threshold')
                    if tasks.state == 'active' and tasks.mode == 'explore':
                        last_success = perception.last_success or perception_started or now
                        if now-last_success > runtime['perception']['failure_timeout_s']:
                            raise ValueError(f'Perception unavailable: {perception.last_error}')
                    velocity = tasks.update(snapshot, now)
                    if perception and perception.last_delivery is not None and perception.last_delivery is not last_perception_delivery:
                        recorder.event('perception_result', **perception.last_delivery)
                        last_perception_delivery = perception.last_delivery
                    if velocity is not None and motion and tasks.state == 'active':
                        expires = min(now+control['command_ttl_s'],
                                      snapshot.pose_at+robot['max_data_age_s'],
                                      snapshot.lidar_at+robot['max_data_age_s'])
                        if tasks.mode == 'explore':
                            expires = min(expires, snapshot.camera_at+robot['max_data_age_s'])
                        motion.submit(velocity, now, expires)
                        recorder.event('command_requested', generated_at=now, expires_at=expires,
                                       velocity=velocity, delivery='queued_not_execution_ack')
                    elif tasks.state == 'completed' and motion and motion.state == 'active':
                        motion.stop('task_completed')
                        recorder.event('task_result', result=tasks.result)
                    if runtime['recording']['observations'] and now-last_record >= .5:
                        recorder.observation(snapshot)
                        last_record = now
                except (ValueError, RuntimeError) as exc:
                    if tasks.state == 'active':
                        tasks.pause()
                        if motion:
                            motion.stop(str(exc))
                        recorder.event('task_paused', reason=str(exc))
                    sensors.last_error = str(exc)
            elif tasks.state == 'active':
                tasks.pause()
                if motion:
                    motion.stop('missing_sensor_data')
            if now-last_status >= 2 and keyboard.editing is None:
                print(f'状态={tasks.state} 任务={tasks.mode} 运动={motion.state if motion else "disabled"}'
                      f' 观测={"ready" if snapshot else "waiting"} 建图={tasks.map_ready} 最近错误={sensors.last_error}', flush=True)
                last_status = now
            time.sleep(max(0, 1/nav['tick_hz']-(time.monotonic()-now)))
    except Exception as exc:
        reason, error = 'runtime_error', f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    finally:
        # Stop/close the writer BEFORE expensive disk operations.
        if motion:
            for event in motion.close():
                recorder.event('motion_state', **event)
                if event['state'] == 'stopped':
                    stop_confirmed = True
                if event['state'] == 'fault':
                    error = event['detail']
        for name, resource in (('keyboard', keyboard), ('perception', perception), ('scorer', scorer), ('sensors', sensors)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    error = f'{name} cleanup: {exc}'
                    recorder.event('cleanup_error', reason=error)
        try:
            if tasks and tasks.map_ready:
                tasks.component.mapping.save(str(recorder.directory / 'final_map'))
        except Exception as exc:
            error = f'Map save failed: {exc}'
        recorder.close({'reason': reason, 'error': error, 'task_result': tasks.result if tasks else None,
                        'completed': error is None, 'hardware_validated': False,
                        'planning_frame': robot['map_frame'], 'global_localization_verified': False,
                        'map_created': bool(tasks and tasks.map_ready),
                        'last_autonomous_stop_confirmed': stop_confirmed})
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0 if error is None else 2
