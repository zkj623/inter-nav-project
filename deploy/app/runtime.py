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
from deploy.robot.world_boundary import WorldBoundary
from deploy.robot.safety import check_observation, checked_velocity
from deploy.interaction.config import box_window_config, box_window_height_limits, create_perception


def check(config):
    required = ('numpy', 'cv2', 'yaml', 'requests', 'scipy')
    optional = ('rclpy', 'sensor_msgs', 'nav_msgs', 'unitree_sdk2py')
    status = {name: importlib.util.find_spec(name) is not None for name in required+optional}
    Tasks(config['navigation'], interaction=config.get('interaction'))
    try:
        check_motion_configuration(config)
        motion = 'configured; hardware NOT verified by this check'
    except ValueError as error:
        motion = str(error)
    print(json.dumps({'check': 'offline_only', 'dependencies': status, 'motion': motion,
                      'planning_frame': ExperimentFrame.name, 'source_frame': config['robot']['map_frame'],
                      'lidar_topic': config['robot']['lidar_topic'],
                      'navigation_point_height_m': config['navigation']['mapping'].get('point_height'),
                      'box_window_point_height_m': box_window_height_limits(config),
                      'startup_clear_area': config['navigation'].get('startup_clear_area'),
                      'navigation_backend': 'simulation_release_online_navigation',
                      'interaction_enabled': config['navigation']['interaction_enabled'],
                      'first_push': config.get('interaction', {}).get('first_push', False),
                      'first_push_face': config['interaction'].get('first_push_face'),
                      'interaction_source': config['interaction']['source'],
                      'registered_boxes': len(config.get('interaction', {}).get('boxes', [])),
                      'contact_profile_configured': config.get('interaction', {}).get('contact_profile') is not None,
                      'interaction_geometry_verified': config['interaction']['geometry_verified'],
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
    from deploy.perception.box_memory import BoxMapMemory
    from deploy.recording.replay import ReplayClock, snapshots
    from deploy.interaction.replay import RecordedInteractionHandoff
    paced = getattr(args, 'realtime', False)
    memory = BoxMapMemory()
    tracking = box_window_config(config) if getattr(args, 'box_window', False) else None
    boundary = WorldBoundary.from_navigation(config['navigation'])
    window = None
    if tracking is not None and not paced:
        from deploy.perception.window_observer import WindowBoxObserver
        window = WindowBoxObserver(tracking, world_boundary=boundary)
    window_only = window is not None and args.task == 'observe'
    perception = window if window_only else create_perception(config, synchronous=True, box_memory=memory)
    tasks = Tasks(config['navigation'], None if window_only else perception,
                  interaction=config.get('interaction'), box_memory=memory)
    handoff = RecordedInteractionHandoff(config['runtime']['control'], config['robot']['max_data_age_s'])
    recorder = _recorder(args, config, 'offline_replay_no_motion')
    started = False
    reason, error = 'recording_exhausted', None
    count = 0
    previous_delivery = None
    clock = ReplayClock() if paced else None
    worker = None
    point_height = config['navigation']['mapping'].get('point_height')
    window_height = box_window_height_limits(config) if tracking is not None else point_height
    full_box_scan = config['interaction'].get('first_push', False) and not window_only
    packets = snapshots(args.replay)
    try:
        if paced:
            from deploy.perception.window_process import WindowProcess
            worker = WindowProcess.replay(tracking, recorder, args.replay, window_height, boundary, box_memory=memory)
            packets = clock.play(packets, on_start=worker.begin)
        for aligned_scan in packets:
            snapshot = aligned_scan.navigation_view(point_height)
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
            perception_scan = aligned_scan if full_box_scan else snapshot
            if window is not None:
                window_scan = aligned_scan.navigation_view(window_height, boundary)
                if window_only:
                    window.update(window_scan, now)
                else:
                    window.publish(window_scan, now, recorder)
                memory.observe(window.last_delivery, now)
            velocity, handoff_event = handoff.update(tasks, perception_scan, now)
            if handoff_event is not None:
                recorder.event(handoff_event.pop('event'), **handoff_event)
            if perception is not None:
                delivery = perception.last_delivery
                if delivery is not None and (not window_only or delivery is not previous_delivery):
                    recorder.event('perception_result', **delivery)
                previous_delivery = delivery
            while tasks.engine.events:
                event = dict(tasks.engine.events.popleft())
                kind = event.pop('event', 'navigation_event')
                if 'kind' in event:
                    event['decision_kind'] = event.pop('kind')
                recorder.event(kind, **event)
            recorder.event('decision', sequence=snapshot.sequence, state=tasks.state,
                           requested_velocity=None if velocity is None else checked_velocity(
                               velocity, config['runtime']['control']), sent_to_robot=False)
            count += 1
            if tasks.state in ('completed', 'failed'):
                reason = 'task_' + tasks.state
                break
        if count == 0:
            raise ValueError('Empty recording')
    except Exception as exc:
        error, reason = str(exc), 'replay_error'
    finally:
        if paced:
            packets.close()
        if perception is not None:
            perception.close()
        if worker is not None:
            try:
                worker.close(drain=error is None and reason == 'recording_exhausted')
            except RuntimeError as exc:
                error, reason = str(exc), 'box_window_replay_error'
        elif window is not None and not window_only:
            window.close()
        try:
            if tasks.map_ready:
                tasks.engine.layers.save(str(recorder.directory / 'final_map'))
        except Exception as exc:
            error = f'Map save failed: {exc}'
        if handoff.pending and error is None:
            reason = 'interaction_stop_handoff_required'
        summary = {'reason': reason, 'error': error, 'task_result': tasks.result,
                   'remembered_boxes': {key: loc.display for key, loc in memory.locations().items()},
                   'first_push_result': getattr(tasks.engine, 'trial_report', None),
                   'decisions': count, 'map_created': tasks.map_ready,
                   'mapping': tasks.engine.layers.statistics(), 'sent_to_robot': False, 'hardware_validated': False,
                   'recorded_stop_handoff': handoff.diagnostic(),
                   'completed': error is None and not handoff.pending}
        if clock is not None:
            summary['realtime'] = dict(first_arrival=clock.first, last_arrival=clock.last,
                                       input_packets=clock.packets, max_arrival_lateness_s=clock.max_lateness)
        recorder.close(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if error is None and not handoff.pending else 2


def run_live(args, config):
    import sys
    from .keyboard import Keyboard
    from .session import Session
    from .observations import LiveObservations
    from deploy.visualization.sources import SdkVideo
    from deploy.robot.sensors import RosSensors
    from deploy.robot.motion import MotionProcess
    if args.allow_motion:
        check_motion_configuration(config)
    tracking = box_window_config(config) if getattr(args, 'box_window', False) else None
    robot, nav, runtime = config['robot'], config['navigation'], config['runtime']
    control = runtime['control']
    reference = ExperimentFrame(0. if args.demo else robot['ground_z'])
    session = Session(config, REPO, port=args.session_port, demo=args.demo,
                      mode='synthetic_demo' if args.demo else ('live_motion_enabled' if args.allow_motion else 'live_observe_only'))
    box_memory = session.box_memory if config['interaction'].get('boxes') else None
    print(f'本轮会话：{session.info["session_id"]}；数据服务：{session.url}', flush=True)
    print('另开终端运行 python view.py 查看三幅图；python record.py 保存实验。', flush=True)
    sensors = motion = keyboard = perception = observations = tasks = video = controller = None
    window_process = None
    window_submit = None
    point_height = nav['mapping'].get('point_height')
    window_height = box_window_height_limits(config) if tracking is not None else point_height
    boundary = WorldBoundary.from_navigation(nav)
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
            print('正在采样传感器时间参照（约 3 秒）；不修改 Orin 系统时间。', flush=True)
            if args.allow_motion and tracking is not None and args.camera != 'ros':
                # Keep ROS subscriptions and high-rate IMU callbacks off the control thread.
                from deploy.robot.sensor_process import RosSensorProcess
                sensors = RosSensorProcess(robot)
            else:
                sensors = RosSensors(robot, session.rgb if args.camera == 'ros' else None,
                                     topic if args.camera == 'ros' else None)
            robot = sensors.config
            reference = ExperimentFrame(robot['ground_z'])
            session.event('lio_observation_reference', pose_frame=robot['base_frame'],
                          ground_z=robot['ground_z'], lidar_origin_offset=robot['lidar_origin_offset_base_m'])
            print(f"里程计 {robot['odom_topic']}，点云 {robot['lidar_topic']}，"
                  f"位姿参考点 {robot['base_frame']}，本次地面 Z={robot['ground_z']} m", flush=True)
            timebase = getattr(sensors, 'timebase', None)
            if timebase is not None:
                session.event('time_reference_initialized', **timebase.metadata)
                print('时间参照已固定：共用启动收包偏差，额外预留 50 ms；无需单独运行 sync_time.py。'
                      '这是相对延迟监控，启动前已有的共同固定延迟仍未知。', flush=True)
            if args.camera == 'sdk':
                video = SdkVideo(session.frames, robot['network_interface'], 2.)
            elif args.camera == 'none':
                session.frames.error('rgb', 'RGB preview disabled (--camera none)')
        # rclpy.init can replace SIGINT; restore coordinated shutdown after init.
        for signum in handlers:
            signal.signal(signum, on_signal)
        if not args.demo:
            # Verified box geometry must match the navigation scan. A display-only
            # fit can run on the provider's worker while navigation gets the scan.
            if tracking is None or args.allow_motion:
                coupled = tracking is None or config['interaction']['geometry_verified']
                perception = create_perception(config, synchronous=coupled, box_memory=box_memory)
            if tracking is not None:
                from deploy.perception.window_process import WindowProcess
                event = 'box_window_result' if perception is not None else 'perception_result'
                window_process = WindowProcess.from_navigation_scans(tracking, session, event, boundary)
                window_submit = window_process.submit
                session.event(event, geometry_source='window_two_faces', detections=[], error='等待导航坐标系')
        observations = LiveObservations(sensors, reference, session, hz=nav['tick_hz'],
                                        max_age=robot['max_data_age_s'], perception=perception,
                                        full_box_scan=config['interaction'].get('first_push', False),
                                        window_point_height=window_height if window_height != point_height else None,
                                        point_height=point_height, world_boundary=boundary, window_submit=window_submit)
        tasks = Tasks(nav, observations if perception is not None else None, interaction=config.get('interaction'),
                      box_memory=box_memory)
        if args.allow_motion:
            motion = MotionProcess(robot, control)
        keyboard = Keyboard() if not args.demo or sys.stdin.isatty() else None
        controller = NavigationController(config, tasks, motion, session)
        last_status = 0.
        # Pace decisions on measured scans; leave time to replace the current
        # command before its source or the writer's watchdog expires.
        period = 1/nav['tick_hz']
        # Swarm-LIO publishes at about 10 Hz. A separate wall-clock tick can skip a scan.
        pace_on_lidar = nav['tick_hz'] >= 10 and not args.demo
        decision_due = float('-inf')
        consumed = None
        print(('合成演示：未连接机器人。' if args.demo else 'MCF 保持运行。默认观察；') + ('运动需按任务键启动。' if motion else '本次会话禁止运动。'), flush=True)
        if boundary is not None:
            print(f'实验边界已启用，X 从 {boundary.x[0]:g} 到 {boundary.x[1]:g} 米，'
                  f'Y 从 {boundary.y[0]:g} 到 {boundary.y[1]:g} 米。', flush=True)
        if tracking is not None:
            for row in tracking['boxes']:
                size = ' × '.join(f'{d * 100:g}' for d in row['dimensions_m'])
                print(f"多帧 LiDAR 认箱已开启。{row['id']} 长 × 宽 × 高为 {size} 厘米。", flush=True)
            push_enabled = (args.allow_motion and nav['interaction_enabled']
                            and any(row['push_allowed'] for row in tracking['boxes']))
            if push_enabled and tasks.engine.first_push:
                if config['interaction'].get('first_push_face') == 'upper_short_end':
                    print('首次推箱已启用。从地图上方接近短端面，向下推，停稳复测后检查原目标通路。', flush=True)
                else:
                    print('首次推箱已启用。点导航时优先接近已登记箱体，目标推移 10 厘米，停稳后复测。', flush=True)
            else:
                print('自主推箱已启用，接触前用当前扫描确认箱体。' if push_enabled else '本轮自主推箱已关闭。', flush=True)
        print(f"来源坐标系={'demo_odom' if args.demo else robot['map_frame']}；等待有效观测初始化 experiment；视频预览={args.camera}", flush=True)
        if not args.demo and (robot['ground_z'] is None or robot['lidar_origin_offset_base_m'] is None):
            print('地面/雷达原点未标定：本次可记录观测，导航地图暂不生成。', flush=True)
        while not quitting:
            cycle_started = time.monotonic()
            now = time.monotonic()
            if args.demo and args.demo_seconds and now-started_at >= args.demo_seconds:
                reason = 'demo_finished'
                break
            operation = keyboard.poll() if keyboard else None
            if motion:
                controller.motion_events(motion.events())
            sensor_read_started = time.monotonic()
            snapshot = observations.latest()
            sensor_read_duration = time.monotonic()-sensor_read_started
            if observations.failure is not None:
                raise RuntimeError(observations.failure)
            # Sensor callbacks may advance while keyboard/events/locks run.
            # Compare against time AFTER acquiring the observation.
            now = time.monotonic()
            if snapshot is not None and snapshot.lidar_at != consumed:
                consumed = snapshot.lidar_at
                decision_due = max(min(decision_due, now), now-.5*period)+period
            observation_error = observations.last_error
            if operation:
                key, value = operation
                if key == 'q':
                    session.event('operator', key=key, value=value)
                    quitting = True
                    continue
                controller.operate(key, value, snapshot, now)
            step_error = None
            if not controller.fresh_cycle_required:
                step_error = controller.step(snapshot, now)
            if controller.fresh_cycle_required:
                # Both an operator start and a restart after planning need a
                # fresh scan. A repackaged pre-arm sweep still has its old expiry.
                controller.fresh_cycle_required = False
                if pace_on_lidar:
                    controller.last_control_lidar_at = snapshot.lidar_at
                observations.wait_next(snapshot, period, distinct_lidar=pace_on_lidar)
                continue
            observation_error = observation_error or step_error
            current_error = observations.last_error or observation_error
            publish_started = time.monotonic()
            session.update(snapshot, tasks, motion.state if motion else 'disabled', current_error,
                           decision_duration_s=controller.decision_duration_s, record_observation=False)
            if tasks.state == 'active' or controller.planning_hold is not None:
                session.event('control_cycle', sequence=None if snapshot is None else snapshot.sequence,
                              cycle_started_at=cycle_started, sensor_read_duration_s=sensor_read_duration,
                              pre_decision_duration_s=now-cycle_started,
                              publication_duration_s=time.monotonic()-publish_started,
                              elapsed_s=time.monotonic()-cycle_started,
                              planning_hold=None if controller.planning_hold is None else controller.planning_hold['token'])
            if now-last_status >= 2 and (keyboard is None or keyboard.editing is None):
                print(f'状态={tasks.state} 任务={tasks.mode} 运动={motion.state if motion else "disabled"}'
                      f' 观测={"ready" if snapshot else "waiting"} 建图={tasks.map_ready} 最近错误={current_error}', flush=True)
                last_status = now
            next_read = decision_due
            if controller.last_expires_at is not None:
                reserve = max(period, controller.last_decision_cost) + 1/control['poll_hz']
                next_read = min(next_read, controller.last_expires_at-reserve)
            if pace_on_lidar and motion and motion.state == 'active':
                observations.wait_next(snapshot, period, distinct_lidar=True)
            else:
                observations.wait_next(snapshot, period, not_before=next_read)
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
        for name, resource in (('keyboard', keyboard), ('observations', observations),
                               ('perception', perception if observations is None else None),
                               ('box_window', window_process),
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
