"""One place for operator intent, task state, and navigation-to-motion decisions."""
import time

from .config import check_task_inputs
from deploy.robot.safety import check_observation


class NavigationController:
    def __init__(self, config, tasks, motion, session, perception=None):
        self.robot = config['robot']
        self.navigation = config['navigation']
        self.control = config['runtime']['control']
        self.tasks, self.motion, self.session = tasks, motion, session
        self.perception = perception
        self.last_perception_delivery = None
        self.stop_confirmed = None
        self.motion_error = None
        self.motion_detail = ''
        self.decision_duration_s = None
        self.task_error = None

    @property
    def motion_state(self):
        return self.motion.state if self.motion else 'disabled'

    def motion_events(self, events):
        for event in events:
            self.session.event('motion_state', **event)
            state = event['state']
            self.motion_detail = event.get('detail', '')
            if state in ('stopping', 'stopped', 'rejected', 'stop_error'):
                print(f"运动状态={state} 原因={self.motion_detail or '未提供'}", flush=True)
                if state == 'rejected' and event.get('telemetry'):
                    print(f"拒绝时反馈={event['telemetry']}", flush=True)
            if state == 'active':
                self.stop_confirmed = False
            elif state == 'stopped':
                self.stop_confirmed = True
            if state in ('stopping', 'fault', 'stop_error'):
                self.tasks.pause()
            if state == 'fault':
                self.motion_error = event['detail']
                print(f"运动状态异常：{event['detail']}", flush=True)

    def pause(self, reason):
        if self.tasks.state == 'active':
            self.task_error = reason
            self.tasks.pause()
            if self.motion:
                self.motion.stop(reason)
            self.session.event('task_paused', reason=reason)

    def operate(self, key, value, snapshot, now):
        """Apply a terminal operation. Only explicit start/resume can arm motion."""
        self.session.event('operator', key=key, value=value)
        if key in ('p', 'prepare', 'stop', 'o', 'm'):
            if self.motion:
                self.motion.stop(key)
            if key == 'p':
                self.tasks.pause()
            else:
                self.tasks.cancel()
                self.task_error = None
            return
        if key not in ('navigate', 'r'):
            return
        try:
            if self.motion is None:
                raise ValueError('本次仅观察。需要真机导航时请退出，使用 python run.py 启动')
            if snapshot is None:
                raise ValueError('等待有效定位和点云')
            check_task_inputs(self.robot, snapshot)
            check_observation(snapshot, now, self.robot, self.control,
                              self.navigation['mapping'], camera=False)
            if snapshot.observation['position'][2] < self.navigation['safe_base_height']:
                raise ValueError('机器人机身高度不足')
            if key == 'navigate':
                xy = tuple(float(item) for item in value.split(','))
                if len(xy) != 2:
                    raise ValueError('目标格式为 x,y，例如 2.0,1.0')
                self.tasks.check_navigation_goal(xy)
                self.tasks.start('navigate', (*xy, float(snapshot.observation['position'][2])), now)
            else:
                if self.tasks.mode == 'navigate':
                    self.tasks.check_navigation_goal(self.tasks.engine.goal)
                self.tasks.resume(now)
            # Wait in a non-running task state. arm() may receive the previous
            # stop's events while waiting for this request's acknowledgement.
            self.tasks.pause()
            self.motion_detail = ''
            armed = self.motion.arm()
            self.motion_events(self.motion.events())
            if not armed or self.motion.state != 'active':
                raise ValueError(f'运动未启用：{self.motion.state}；原因={self.motion_detail or "未收到具体原因"}')
            self.tasks.resume(now)
            self.task_error = None
            self.stop_confirmed = False
        except (ValueError, RuntimeError) as exc:
            self.task_error = str(exc)
            self.pause(str(exc))
            print(str(exc), flush=True)
            self.session.event('task_rejected', reason=str(exc))

    def step(self, snapshot, now):
        """Validate this observation, update the original algorithm, queue at most one command."""
        self.decision_duration_s = None
        if snapshot is None:
            self.pause('missing_sensor_data')
            return self.task_error
        try:
            if snapshot.geometry_calibrated:
                check_observation(snapshot, now, self.robot, self.control,
                                  self.navigation['mapping'], camera=False)
            else:
                snapshot.require_fresh(now, self.robot['max_data_age_s'])
            if self.tasks.state == 'active' and snapshot.observation['position'][2] < self.navigation['fall_height']:
                raise ValueError('Robot below fall-height threshold')
            previous_state = self.tasks.state
            decision_started = time.monotonic()
            velocity = self.tasks.update(snapshot, now)
            engine = getattr(self.tasks, 'engine', None)
            if engine is not None:
                while engine.events:
                    event = dict(engine.events.popleft())
                    kind = event.pop('event', 'navigation_event')
                    if 'kind' in event:
                        event['decision_kind'] = event.pop('kind')
                    self.session.event(kind, **event)
            decision_duration = time.monotonic() - decision_started
            self.decision_duration_s = decision_duration
            if self.perception:
                delivery = getattr(self.perception, 'last_delivery', None)
                if delivery is not None and delivery is not self.last_perception_delivery:
                    self.session.event('perception_result', **delivery)
                    self.last_perception_delivery = delivery
            if velocity is not None and self.motion and self.tasks.state == 'active':
                expires = min(now+self.control['command_ttl_s'],
                              snapshot.pose_at+self.robot['max_data_age_s'],
                              snapshot.lidar_at+self.robot['max_data_age_s'])
                dispatch_at = time.monotonic()
                if expires <= dispatch_at:
                    self.session.event('command_discarded', reason='decision_expired',
                                       generated_at=now, expires_at=expires, discarded_at=dispatch_at,
                                       velocity=velocity, decision_duration_s=decision_duration,
                                       decision_stages_s=dict(self.tasks.last_timings),
                                       remaining_valid_s=expires-dispatch_at,
                                       pose_age_at_discard=dispatch_at-snapshot.pose_at,
                                       lidar_age_at_discard=dispatch_at-snapshot.lidar_at,
                                       delivery='not_queued')
                    self.pause('decision_expired')
                    return self.task_error
                self.motion.submit(velocity, now, expires)
                submitted_at = time.monotonic()
                self.session.event('command_requested', generated_at=now, expires_at=expires,
                                   velocity=velocity, decision_duration_s=decision_duration,
                                   decision_stages_s=dict(self.tasks.last_timings),
                                   submitted_at=submitted_at, remaining_valid_s=expires-submitted_at,
                                   pose_age_at_submit=submitted_at-snapshot.pose_at,
                                   lidar_age_at_submit=submitted_at-snapshot.lidar_at,
                                   delivery='queued_not_execution_ack')
            if previous_state == 'active' and self.tasks.state == 'completed':
                if self.motion:
                    self.motion.stop('task_completed')
                self.session.event('task_result', result=self.tasks.result)
            return self.task_error
        except (ValueError, RuntimeError) as exc:
            self.pause(str(exc))
            return str(exc)
