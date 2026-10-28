"""One place for operator intent, task state, and navigation-to-motion decisions."""
import time

from .config import check_task_inputs
from deploy.robot.safety import check_observation


class NavigationController:
    def __init__(self, config, tasks, motion, session, perception=None):
        self.robot = config['robot']
        self.navigation = config['navigation']
        self.control = config['runtime']['control']
        self.perception_timeout = config['runtime']['perception']['failure_timeout_s']
        self.tasks, self.motion, self.session = tasks, motion, session
        self.perception = perception
        self.perception_started = None
        self.last_perception_delivery = None
        self.stop_confirmed = None
        self.motion_error = None
        self.motion_detail = ''

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
            return
        if key not in ('navigate', 'explore', 'r'):
            return
        try:
            if self.motion is None:
                raise ValueError('本次仅观察。实机配置验收后使用 --allow-motion 启动')
            if snapshot is None:
                raise ValueError('等待有效定位和点云')
            exploring = key == 'explore' or (key == 'r' and self.tasks.mode == 'explore')
            check_task_inputs(self.robot, snapshot, exploring)
            check_observation(snapshot, now, self.robot, self.control,
                              self.navigation['mapping'], camera=exploring)
            if snapshot.observation['position'][2] < self.navigation['safe_base_height']:
                raise ValueError('机器人机身高度不足')
            if key == 'navigate':
                xy = tuple(float(item) for item in value.split(','))
                if len(xy) != 2:
                    raise ValueError('目标格式为 x,y，例如 2.0,1.0')
                self.tasks.start('navigate', (*xy, float(snapshot.observation['position'][2])), now)
            elif key == 'explore':
                if not value.strip():
                    raise ValueError('目标物品名称不能为空')
                self.tasks.start('explore', value.replace('_', ' '), now)
            else:
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
            self.stop_confirmed = False
            if exploring:
                self.perception_started = now
        except (ValueError, RuntimeError) as exc:
            self.pause(str(exc))
            print(str(exc), flush=True)
            self.session.event('task_rejected', reason=str(exc))

    def step(self, snapshot, now):
        """Validate this observation, update the original algorithm, queue at most one command."""
        if snapshot is None:
            self.pause('missing_sensor_data')
            return None
        try:
            exploring = self.tasks.state == 'active' and self.tasks.mode == 'explore'
            if snapshot.geometry_calibrated:
                check_observation(snapshot, now, self.robot, self.control,
                                  self.navigation['mapping'], camera=exploring)
            else:
                snapshot.require_fresh(now, self.robot['max_data_age_s'])
            if self.tasks.state == 'active' and snapshot.observation['position'][2] < self.navigation['fall_height']:
                raise ValueError('Robot below fall-height threshold')
            if exploring:
                # A resume starts a new grace interval; an old failure cannot immediately re-pause it.
                last_success = max(self.perception.last_success or 0., self.perception_started or now)
                if now-last_success > self.perception_timeout:
                    raise ValueError(f'Perception unavailable: {self.perception.last_error}')
            previous_state = self.tasks.state
            decision_started = time.monotonic()
            velocity = self.tasks.update(snapshot, now)
            decision_duration = time.monotonic() - decision_started
            if self.perception:
                delivery = self.perception.last_delivery
                if delivery is not None and delivery is not self.last_perception_delivery:
                    self.session.event('perception_result', **delivery)
                    self.last_perception_delivery = delivery
            if velocity is not None and self.motion and self.tasks.state == 'active':
                expires = min(now+self.control['command_ttl_s'],
                              snapshot.pose_at+self.robot['max_data_age_s'],
                              snapshot.lidar_at+self.robot['max_data_age_s'])
                if exploring:
                    expires = min(expires, snapshot.camera_at+self.robot['max_data_age_s'])
                self.motion.submit(velocity, now, expires)
                self.session.event('command_requested', generated_at=now, expires_at=expires,
                                   velocity=velocity, decision_duration_s=decision_duration,
                                   delivery='queued_not_execution_ack')
            if previous_state == 'active' and self.tasks.state == 'completed':
                if self.motion:
                    self.motion.stop('task_completed')
                self.session.event('task_result', result=self.tasks.result)
            return None
        except (ValueError, RuntimeError) as exc:
            self.pause(str(exc))
            return str(exc)
