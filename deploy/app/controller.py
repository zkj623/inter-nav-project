"""One place for operator intent, task state, and navigation-to-motion decisions."""
import time

from .config import check_task_inputs
from .navigation import PlanningRequired
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
        self.wait_reason = None
        self.planning_hold = None
        self.hold_sequence = 0
        self.last_expires_at = None
        self.last_decision_cost = 0.
        self.pending_operation = None
        self.fresh_cycle_required = False
        self.tasks.engine.before_planning = self._before_planning

    def _before_planning(self, reason):
        if self.motion and self.motion.state == 'active':
            raise PlanningRequired(reason)

    def _hold(self, reason, *, operation=None):
        self.hold_sequence += 1
        token = f'planning_hold:{self.hold_sequence}:{reason}'
        self.planning_hold = {'token': token, 'reason': reason, 'stopped_at': None,
                              'prepared_at': None}
        self.pending_operation = operation
        self.last_expires_at = None
        self.motion.stop(token)
        self.session.event('planning_hold_requested', hold_id=token, reason=reason)
        print(f'正在停稳后规划：{reason}', flush=True)

    def _clear_hold(self):
        self.planning_hold = self.pending_operation = None
        self.last_expires_at = None

    @property
    def motion_state(self):
        return self.motion.state if self.motion else 'disabled'

    def motion_events(self, events):
        for event in events:
            if event.get('kind') == 'command_ack':
                self.session.event('command_accepted' if event['accepted'] else 'command_rejected',
                                   **{key: value for key, value in event.items() if key != 'kind'})
                if not event['accepted']:
                    self.pause('motion_command_rejected:'+event.get('detail', 'unknown'))
                continue
            self.session.event('motion_state', **event)
            state = event['state']
            self.motion_detail = event.get('detail', '')
            planned = self.planning_hold is not None and self.motion_detail == self.planning_hold['token']
            if state in ('stopping', 'stopped', 'rejected', 'stop_error'):
                print(f"运动状态={state} 原因={self.motion_detail or '未提供'}", flush=True)
                if state == 'rejected' and event.get('telemetry'):
                    print(f"拒绝时反馈={event['telemetry']}", flush=True)
            if state == 'active':
                self.stop_confirmed = False
            elif state == 'stopped':
                self.stop_confirmed = True
                if planned:
                    self.planning_hold['stopped_at'] = event.get('worker_at', time.monotonic())
                    self.tasks.engine.interaction_stopped()
            if state in ('stopping', 'fault', 'stop_error', 'rejected') and not planned:
                self._clear_hold()
                if self.tasks.state == 'active':
                    self.tasks.pause()
            if state == 'fault':
                self.motion_error = event['detail']
                print(f"运动状态异常：{event['detail']}", flush=True)

    def pause(self, reason):
        self._clear_hold()
        if self.tasks.state == 'active':
            self.task_error = reason
            self.tasks.pause()
            if self.motion:
                self.motion.stop(reason)
            self.session.event('task_paused', reason=reason)

    def operate(self, key, value, snapshot, now, *, deferred=False):
        """Only explicit goal intent, or its confirmed planning hold, can arm."""
        if not deferred:
            self.session.event('operator', key=key, value=value)
            self._clear_hold()
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
            elif self.tasks.mode == 'navigate':
                self.tasks.check_navigation_goal(self.tasks.engine.goal)
            if self.motion.state in ('active', 'stopping'):
                # Replacing an active goal must never run cold planning while
                # the previous velocity is still executing. Retain the user
                # operation, not an old snapshot, until matching stopped feedback.
                self.tasks.pause()
                self._hold('operator_goal_change', operation=(key, value))
                return
            if key == 'navigate':
                self.tasks.start('navigate', (*xy, float(snapshot.observation['position'][2])), now)
            else:
                self.tasks.resume(now)
            # Cold global planning can exceed the command TTL on Orin. Do it
            # BEFORE arming, never by extending a command's validity or feeding
            # a watchdog from an old observation. Preserve this prepared route
            # across arm acknowledgement instead of resetting it in resume().
            print('正在准备导航路径，暂不启用运动…', flush=True)
            self.wait_reason = None
            prepared_at = time.monotonic()
            prepared = self.tasks.prepare_for_motion(snapshot, now)
            preparation_duration = time.monotonic()-prepared_at
            self.session.event('navigation_prepared', ready=prepared,
                               duration_s=preparation_duration,
                               decision_stages_s=dict(self.tasks.last_timings),
                               phase=self.tasks.engine.phase, sent_to_robot=False)
            if not prepared:
                if self.tasks.result is not None:
                    self.session.event('task_result', result=self.tasks.result)
                    print(f'导航准备结束：{self.tasks.result}', flush=True)
                else:
                    self.wait_reason = (self.tasks.engine.decision or {}).get('reason', 'no_executable_route')
                    self.task_error = f'导航等待：{self.wait_reason}；保持停止，地图更新后按 r 重新规划。'
                    self.session.event('navigation_wait', reason=self.wait_reason, sent_to_robot=False)
                    print(self.task_error, flush=True)
                return
            print(f'导航准备完成，耗时 {preparation_duration*1000:.0f} ms；正在启用运动。', flush=True)
            self.motion_detail = ''
            armed = self.motion.arm()
            self.motion_events(self.motion.events())
            if not armed or self.motion.state != 'active':
                raise ValueError(f'运动未启用：{self.motion.state}；原因={self.motion_detail or "未收到具体原因"}')
            self.tasks.activate_prepared(time.monotonic())
            self.last_expires_at = None
            self.fresh_cycle_required = True
            self.task_error = None
            self.stop_confirmed = False
        except (ValueError, RuntimeError) as exc:
            self.task_error = str(exc)
            self.pause(str(exc))
            self.tasks.pause()  # Also invalidate a route whose arm request was rejected.
            print(str(exc), flush=True)
            self.session.event('task_rejected', reason=str(exc))

    def _continue_hold(self, snapshot, now):
        hold = self.planning_hold
        if hold['stopped_at'] is None or self.motion.state != 'stopped':
            return
        # Both pose and LiDAR must be captured after this physical stop. An
        # old-but-within-max-age scan must not authorize an automatic restart.
        if min(snapshot.pose_at, snapshot.lidar_at) <= hold['stopped_at']:
            return
        if self.pending_operation is not None:
            key, value = self.pending_operation
            self._clear_hold()
            self.operate(key, value, snapshot, now, deferred=True)
            return
        if self.tasks.state != 'active':
            self._clear_hold()
            return
        if hold['prepared_at'] is None:
            started = time.monotonic()
            ready = self.tasks.prepare_for_motion(snapshot, now)
            self.decision_duration_s = time.monotonic()-started
            self.last_decision_cost = 0.
            if not ready:
                self.session.event('planning_hold_finished', hold_id=hold['token'], ready=False,
                                   result=self.tasks.result, decision=self.tasks.engine.decision)
                if self.tasks.result is not None:
                    self.session.event('task_result', result=self.tasks.result)
                self.task_error = None if self.tasks.result is not None else '导航等待：当前地图没有可执行路线；更新后按 r 重新规划。'
                self._clear_hold()
                return
            self._finish_stopped_interaction()
            hold['prepared_at'] = time.monotonic()
            self.session.event('planning_hold_prepared', hold_id=hold['token'],
                               duration_s=self.decision_duration_s, sent_to_robot=False)
            return

    def _finish_stopped_interaction(self):
        """Complete a skill proved while its writer has remained disarmed."""
        hold = self.planning_hold
        engine = self.tasks.engine
        if (hold is not None and hold['stopped_at'] is not None
                and self.motion.state == 'stopped' and engine.execution.awaiting_stop):
            engine.interaction_stopped()
            self.session.event('interaction_handoff_completed', hold_id=hold['token'],
                               evidence='writer_remained_stopped', sent_to_robot=False)

    def _resume_hold(self, snapshot, now):
        hold = self.planning_hold
        if hold is None or hold['prepared_at'] is None:
            return False
        if min(snapshot.pose_at, snapshot.lidar_at) <= hold['prepared_at']:
            return True
        if snapshot.observation['position'][2] < self.navigation['safe_base_height']:
            raise ValueError('机器人机身高度不足，不能恢复导航')
        # Revalidate the retained route against a new map/pose while disarmed.
        # This update may discover another planning need; no velocity is sent.
        self.tasks.activate_prepared(now)
        started = time.monotonic()
        ready = self.tasks.prepare_for_motion(snapshot, now)
        finished = time.monotonic()
        self.decision_duration_s = finished-started
        if not ready:
            if self.tasks.result is not None:
                self.session.event('task_result', result=self.tasks.result)
            self.task_error = None if self.tasks.result is not None else '导航等待：新观测下无可执行路线；按 r 重新规划。'
            self._clear_hold()
            return True
        self._finish_stopped_interaction()
        # A slow revalidation must itself finish disarmed and request another
        # fresh observation. Never retimestamp its result or extend sensor age.
        remaining = min(snapshot.pose_at, snapshot.lidar_at)+self.robot['max_data_age_s']-finished
        if finished-started > 1/self.navigation['tick_hz'] or remaining < 1/self.navigation['tick_hz']+.05:
            hold['prepared_at'] = finished
            return True
        armed = self.motion.arm()
        self.motion_events(self.motion.events())
        if self.planning_hold is not hold or not armed or self.motion.state != 'active':
            self.pause('planning_resume_rejected')
            self.tasks.pause()
            return True
        self.tasks.activate_prepared(time.monotonic())
        self.fresh_cycle_required = True
        self._clear_hold()
        self.task_error = None
        self.session.event('planning_hold_resumed', hold_id=hold['token'], sent_to_robot=False)
        print('规划已更新，继续当前目标。', flush=True)
        return True

    def _record_perception(self):
        if self.perception is None:
            return
        delivery = getattr(self.perception, 'last_delivery', None)
        if delivery is not None and delivery is not self.last_perception_delivery:
            self.session.event('perception_result', **delivery)
            self.last_perception_delivery = delivery

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
            if (self.tasks.state == 'active' or self.planning_hold is not None) and snapshot.observation['position'][2] < self.navigation['fall_height']:
                raise ValueError('Robot below fall-height threshold')
            if self.planning_hold is not None:
                # Physical stopping can outlast the target geometry lease.
                # Continue observing fresh scans while disarmed; never keep a
                # target alive by retimestamping its old pose or stepping the skill.
                self.tasks.observe_only(snapshot, now)
                self._record_perception()
                if not self._resume_hold(snapshot, now):
                    self._continue_hold(snapshot, now)
                return self.task_error
            if self.tasks.state == 'active' and self.motion and self.last_expires_at is not None:
                remaining = self.last_expires_at-time.monotonic()
                if remaining <= 0:
                    self.pause('control_deadline_missed')
                    return self.task_error
                # Admit a control cycle only when the preceding lease covers
                # its recent computation cost and transport reserve. Otherwise
                # explicitly stop BEFORE computing, preserving the same goal.
                reserve = max(.05, 2/self.control['poll_hz'])
                budget = max(1/self.navigation['tick_hz'], self.last_decision_cost*1.5)+reserve
                if self.tasks.engine.execution.active:
                    # Fitting runs asynchronously: the accepted target is
                    # commonly one tick older than this snapshot. Budget the
                    # computation ahead and one writer poll plus queue slack,
                    # not a second observation cycle. This is an admission
                    # estimate only; dispatch and writer enforce both leases.
                    budget = max(.005, self.last_decision_cost*1.5)+1/self.control['poll_hz']+.005
                if remaining < budget:
                    self._hold('control_budget')
                    return self.task_error
            previous_state = self.tasks.state
            decision_started = time.monotonic()
            velocity = self.tasks.update(snapshot, now)
            if self.motion:
                self.motion_events(self.motion.events())
            engine = getattr(self.tasks, 'engine', None)
            if engine is not None:
                while engine.events:
                    event = dict(engine.events.popleft())
                    kind = event.pop('event', 'navigation_event')
                    if 'kind' in event:
                        event['decision_kind'] = event.pop('kind')
                    self.session.event(kind, **event)
                    if kind == 'decision':
                        reason = event.get('reason') if event.get('decision_kind') == 'wait' else None
                        if reason and reason != self.wait_reason:
                            print(f'导航等待：{reason}；当前观测地图没有满足净空的路径或可达探索点。', flush=True)
                        self.wait_reason = reason
            decision_duration = time.monotonic() - decision_started
            self.decision_duration_s = decision_duration
            self.last_decision_cost = decision_duration
            self._record_perception()
            if engine is not None and engine.execution.awaiting_stop and self.motion and self.tasks.state == 'active':
                # A completed contact skill cannot silently drop its writer
                # envelope. Obtain physical stop feedback before ordinary nav.
                self._hold('interaction_completed')
                return self.task_error
            if velocity is not None and self.motion and self.tasks.state == 'active':
                expires = min(now+self.control['command_ttl_s'],
                              snapshot.pose_at+self.robot['max_data_age_s'],
                              snapshot.lidar_at+self.robot['max_data_age_s'])
                interaction = engine.interaction_command if engine is not None else None
                if interaction is not None:
                    expires = min(expires, interaction['source_expires_at'], interaction['deadline'])
                dispatch_at = time.monotonic()
                if expires <= dispatch_at:
                    print(f'导航命令已过期：计算 {decision_duration*1000:.0f} ms，'
                          f'定位年龄 {(dispatch_at-snapshot.pose_at)*1000:.0f} ms，'
                          f'点云年龄 {(dispatch_at-snapshot.lidar_at)*1000:.0f} ms；暂停导航。', flush=True)
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
                if self.last_expires_at is not None and dispatch_at >= self.last_expires_at:
                    self.session.event('command_discarded', reason='previous_command_expired',
                                       previous_expires_at=self.last_expires_at, discarded_at=dispatch_at,
                                       decision_duration_s=decision_duration, delivery='not_queued')
                    self.pause('control_deadline_missed')
                    return self.task_error
                command_id = (self.motion.submit(velocity, now, expires, interaction=interaction)
                              if interaction is not None else self.motion.submit(velocity, now, expires))
                self.last_expires_at = expires
                submitted_at = time.monotonic()
                self.session.event('command_requested', generated_at=now, expires_at=expires,
                                   velocity=velocity, decision_duration_s=decision_duration,
                                   decision_stages_s=dict(self.tasks.last_timings),
                                   submitted_at=submitted_at, remaining_valid_s=expires-submitted_at,
                                   pose_age_at_submit=submitted_at-snapshot.pose_at,
                                   lidar_age_at_submit=submitted_at-snapshot.lidar_at,
                                   command_id=command_id, delivery='queued_not_execution_ack')
                if interaction is not None:
                    self.session.event('interaction_command', command_id=command_id, **interaction)
            if previous_state == 'active' and self.tasks.state == 'completed':
                if self.motion:
                    self.motion.stop('task_completed')
                self.session.event('task_result', result=self.tasks.result)
            return self.task_error
        except PlanningRequired as exc:
            self._hold(str(exc))
            return self.task_error
        except (ValueError, RuntimeError) as exc:
            self.pause(str(exc))
            return str(exc)
