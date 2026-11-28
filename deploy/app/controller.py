"""One place for operator intent, task state, and navigation-to-motion decisions."""
import time

from .config import check_task_inputs
from .navigation import PlanningRequired
from .tasks import BOX_SCAN_WAITS
from deploy.interaction.execution import TIMED_PUSH_PHASES
from deploy.robot.safety import check_observation


def wait_message(decision):
    reason = (decision or {}).get('reason', 'no_executable_route')
    if reason == 'all_push_faces_rejected':
        return '当前已知箱面的推前复核均未通过，导航等待。重新下达导航任务可重试这些面。'
    detail = {
        'waiting_for_field_scan': '正在等场内有效雷达回波',
        'waiting_for_registered_box': '正在等点云确认登记箱位',
        'waiting_for_motion_feedback': '正在等机身速度反馈',
        'reselect_push_face': '推前复核未通过，正在从剩余箱面重新选择',
        'no_reachable_push_face': '当前地图上没有找到既能安全推完行程、又能让出去目标路线的可达箱面',
        'no_reachable_upper_short_end': '当前地图没有可达的上方短端面站位',
        'waiting_for_post_push_scan': '推后已请求停车，等待停稳后的点云',
        'waiting_for_post_push_exit': '正在用新点云寻找安全脱离膨胀区的方向',
        'waiting_for_post_push_route': '推箱动作已结束，正在等新扫描确认通往原目标的路径',
    }.get(reason, reason)
    retry = '收到新观测后自动重试' if reason in BOX_SCAN_WAITS else '地图更新后按 r 重新规划'
    return f'导航等待：{detail}。{retry}。'


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
        self.last_contact_source_deadline = None
        self.last_control_lidar_at = None
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
        self.last_contact_source_deadline = None
        self.motion.stop(token)
        self.session.event('planning_hold_requested', hold_id=token, reason=reason)
        print(f'正在停稳后规划：{reason}', flush=True)

    def _clear_hold(self):
        self.planning_hold = self.pending_operation = None
        self.last_expires_at = None
        self.last_contact_source_deadline = None
        self.last_control_lidar_at = None

    def _contact_observation_stop(self, reason):
        if (reason != 'interaction_observation_expired' or self.tasks.state != 'active'
                or self.pending_operation is not None or not self.tasks.engine.wait_for_contact_scan()):
            return False
        # The independent writer has already applied its watchdog stop. Wait
        # for that stop's actual acknowledgement without sending a second stop.
        if self.planning_hold is None or self.planning_hold['token'] != reason:
            self._clear_hold()
            self.planning_hold = dict(token=reason, reason='first_push_wait_scan', stopped_at=None, prepared_at=None)
        return True

    @property
    def motion_state(self):
        return self.motion.state if self.motion else 'disabled'

    def motion_events(self, events):
        for event in events:
            if event.get('kind') == 'command_ack':
                self.session.event('command_accepted' if event['accepted'] else 'command_rejected',
                                   **{key: value for key, value in event.items() if key != 'kind'})
                if not event['accepted']:
                    reason = event.get('detail', 'unknown')
                    if not self._contact_observation_stop(reason):
                        self.pause('motion_command_rejected:'+reason)
                continue
            self.session.event('motion_state', **event)
            if event.get('event') == 'push_speed_sent':
                self.tasks.engine.push_speed_sent(event.get('object_id'), interaction_id=event.get('interaction_id'),
                                                  worker_at=event.get('move_at', event.get('worker_at')))
                print(f"已向 Go2 发送顶箱速度，前进 {event['velocity'][0]*100:.1f} cm/s。", flush=True)
            elif event.get('event') == 'timed_push_speed_reached':
                print(f"已向 Go2 发送目标推箱速度 {event['velocity'][0]*100:.1f} cm/s。", flush=True)
            state = event['state']
            self.motion_detail = event.get('detail', '')
            if state in ('stopping', 'stopped'):
                self._contact_observation_stop(self.motion_detail)
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
                    self.tasks.engine.interaction_stopped(self.planning_hold['stopped_at'])
            if state in ('stopping', 'fault', 'stop_error', 'rejected') and not planned:
                requested_stop = self.motion_detail in ('p', 'prepare', 'stop', 'o', 'm', 'task_completed', 'task_failed', 'close', 'manual')
                if self.tasks.mode == 'navigate' and (not requested_stop or state in ('fault', 'stop_error', 'rejected')):
                    self.task_error = self.motion_detail or state
                self._clear_hold()
                if self.tasks.state == 'active':
                    self.tasks.pause()
            if state == 'fault':
                self.motion_error = event['detail']
                print(f"运动状态异常：{event['detail']}", flush=True)

    def pause(self, reason):
        was_running = self.tasks.state == 'active' or self.planning_hold is not None
        self._clear_hold()
        if was_running:
            self.task_error = reason
            self.tasks.pause()
            if self.motion and self.motion.state not in ('stopped', 'manual', 'disabled'):
                self.motion.stop(reason)
            self.session.event('task_paused', reason=reason)

    def _record_result(self):
        result = self.tasks.result
        self.task_error = None if result['success'] else result['reason']
        self.session.event('task_result', result=result)
        print('导航已到达目标并停稳。' if result['success'] else f"导航失败：{result['reason']}", flush=True)

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
            prepared = self.tasks.prepare_for_motion(snapshot, now, retain_box_wait=True)
            preparation_duration = time.monotonic()-prepared_at
            self.session.event('navigation_prepared', ready=prepared,
                               duration_s=preparation_duration,
                               decision_stages_s=dict(self.tasks.last_timings),
                               phase=self.tasks.engine.phase, sent_to_robot=False)
            if not prepared:
                if self.tasks.result is not None:
                    self._record_result()
                else:
                    self.wait_reason = (self.tasks.engine.decision or {}).get('reason', 'no_executable_route')
                    self.task_error = wait_message(self.tasks.engine.decision)
                    self.session.event('navigation_wait', reason=self.wait_reason, sent_to_robot=False)
                    print(self.task_error, flush=True)
                    if self.tasks.state == 'active':
                        self._hold(self.wait_reason)
                        self.planning_hold['route_check_scan'] = snapshot.lidar_at
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
            if (self.tasks.engine.decision or {}).get('reason') in BOX_SCAN_WAITS and (
                    snapshot.lidar_at <= hold.get('route_check_scan', -float('inf')) or now < self.tasks.engine.next_decision):
                return
            started = time.monotonic()
            ready = self.tasks.prepare_for_motion(snapshot, now, retain_box_wait=True)
            self.decision_duration_s = time.monotonic()-started
            self.last_decision_cost = 0.
            if not ready:
                if self.tasks.state == 'active':
                    if 'route_check_scan' not in hold:
                        print(wait_message(self.tasks.engine.decision), flush=True)
                    hold['route_check_scan'] = snapshot.lidar_at
                    return
                self.session.event('planning_hold_finished', hold_id=hold['token'], ready=False,
                                   result=self.tasks.result, decision=self.tasks.engine.decision)
                if self.tasks.result is not None:
                    self._record_result()
                else:
                    self.task_error = wait_message(self.tasks.engine.decision)
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
        ready = self.tasks.prepare_for_motion(snapshot, now, retain_box_wait=True)
        finished = time.monotonic()
        self.decision_duration_s = finished-started
        if not ready:
            if self.tasks.state == 'active':
                hold['prepared_at'] = None
                hold['route_check_scan'] = snapshot.lidar_at
                return True
            if self.tasks.result is not None:
                self._record_result()
            else:
                self.task_error = wait_message(self.tasks.engine.decision)
            self._clear_hold()
            return True
        self._finish_stopped_interaction()
        # The source timestamps, not a nominal 100 ms tick budget, determine
        # whether this preparation remains valid. An expired result still
        # requires another fresh observation while the writer stays disarmed.
        remaining = min(snapshot.pose_at, snapshot.lidar_at)+self.robot['max_data_age_s']-finished
        interaction = self.tasks.engine.execution.active
        if remaining <= 0 or (interaction and (finished-started > 1/self.navigation['tick_hz']
                                               or remaining < 1/self.navigation['tick_hz']+.05)):
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

    def _resume_contact_scan(self, snapshot, now):
        hold = self.planning_hold
        if now >= self.tasks.engine.execution.deadline:
            self.pause('first_push_approach_time_limit')
            return False
        if (hold['stopped_at'] is None or self.motion.state != 'stopped'
                or min(snapshot.lidar_at, snapshot.pose_at) <= hold['stopped_at']
                or not self.tasks.contact_scan_ready(snapshot, now)):
            return False
        if snapshot.observation['position'][2] < self.navigation['safe_base_height']:
            self.pause('contact_scan_resume_low_height')
            return False
        armed = self.motion.arm()
        self.motion_events(self.motion.events())
        if not armed or self.motion.state != 'active' or self.planning_hold is not hold:
            self.pause('contact_scan_resume_rejected')
            return False
        self._clear_hold()
        return True

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
                if self.planning_hold['reason'] != 'first_push_wait_scan':
                    if not self._resume_hold(snapshot, now):
                        self._continue_hold(snapshot, now)
                    return self.task_error
                if not self._resume_contact_scan(snapshot, now):
                    return self.task_error
                now = time.monotonic()
            if self.tasks.state == 'active' and self.motion and self.last_expires_at is not None:
                remaining = self.last_expires_at-time.monotonic()
                if remaining <= 0:
                    if self.last_contact_source_deadline == self.last_expires_at:
                        # Only the writer can distinguish an expired scan from
                        # a concurrent takeover, stop fault or elapsed budget.
                        return self.task_error
                    self.pause('control_deadline_missed')
                    return self.task_error
                if self.tasks.engine.execution.active and self.tasks.engine.phase not in TIMED_PUSH_PHASES:
                    # Retain the existing contact-skill admission contract.
                    # This change targets ordinary coordinate navigation.
                    budget = max(.005, self.last_decision_cost*1.5)+1/self.control['poll_hz']+.005
                    if remaining < budget:
                        if self.last_contact_source_deadline == self.last_expires_at:
                            return self.task_error
                        self._hold('control_budget')
                        return self.task_error
            if (self.tasks.state == 'active' and self.motion and snapshot.lidar_at == self.last_control_lidar_at
                    and (self.last_expires_at is None or not self.tasks.engine.execution.active)):
                # This sweep was used to arm or already supplied a navigation
                # velocity. A timeout republication cannot renew its validity.
                return self.task_error
            previous_state = self.tasks.state
            decision_started = time.monotonic()
            velocity = self.tasks.update(snapshot, now)
            if self.motion:
                self.motion_events(self.motion.events())
                if self.planning_hold is not None:
                    return self.task_error
            engine = getattr(self.tasks, 'engine', None)
            if engine is not None:
                while engine.events:
                    event = dict(engine.events.popleft())
                    kind = event.pop('event', 'navigation_event')
                    if 'kind' in event:
                        event['decision_kind'] = event.pop('kind')
                    self.session.event(kind, **event)
                    if kind == 'post_push_navigation_resumed':
                        print('推箱已停稳，导航正在用新点云检查通往原目标的路线。', flush=True)
                    if kind == 'post_push_buffer_recovery':
                        print('Go2 在障碍膨胀区内，正在调整站位。', flush=True)
                    if kind == 'push_face_rejected':
                        dx, dy = event['push_direction']
                        print(f"箱号 {event['box_id']} 推前复核未通过（{event['reason']}），"
                              f"排除推向 ({dx:.2f}, {dy:.2f}) 的箱面，重新选面。", flush=True)
                    if kind == 'decision':
                        if event.get('decision_kind') == 'push':
                            detail = (f"从地图上方推短端面，计划向下推 {event['push_distance_m']*100:.0f} cm。"
                                      if event.get('reason') == 'upper_short_end_approach' else '')
                            if event.get('reason') == 'automatic_face_approach':
                                dx, dy = event['push_direction']
                                detail = f"自动选面，推向 ({dx:.2f}, {dy:.2f})，机身计划推进 {event['push_distance_m']*100:.1f} cm。"
                            print(f"导航选中推箱，箱号 {event['box_id']}。{detail}", flush=True)
                        reason = event.get('reason') if event.get('decision_kind') == 'wait' else None
                        if reason and reason != self.wait_reason:
                            print(wait_message(event), flush=True)
                        self.wait_reason = reason
            decision_duration = time.monotonic() - decision_started
            self.decision_duration_s = decision_duration
            self.last_decision_cost = decision_duration
            self._record_perception()
            if engine is not None and engine.execution.awaiting_stop and self.motion and self.tasks.state == 'active':
                # A completed contact skill cannot silently drop its writer
                # envelope. Obtain physical stop feedback before ordinary nav.
                self._hold('first_push_wait_scan' if engine.phase == 'first_push_wait_scan' else 'interaction_completed')
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
                    if (interaction and expires == interaction['source_expires_at'] < interaction['deadline']
                            and engine.wait_for_contact_scan()):
                        if self.last_contact_source_deadline is None or dispatch_at < self.last_contact_source_deadline:
                            self._hold('first_push_wait_scan')
                        return self.task_error
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
                    if self.last_contact_source_deadline == self.last_expires_at:
                        return self.task_error
                    self.session.event('command_discarded', reason='previous_command_expired',
                                       previous_expires_at=self.last_expires_at, discarded_at=dispatch_at,
                                       decision_duration_s=decision_duration, delivery='not_queued')
                    self.pause('control_deadline_missed')
                    return self.task_error
                command_id = (self.motion.submit(velocity, now, expires, interaction=interaction)
                              if interaction is not None else self.motion.submit(velocity, now, expires))
                self.last_expires_at = expires
                self.last_contact_source_deadline = (interaction['source_expires_at']
                    if interaction and engine.first_push and interaction['phase'] in ('engage', 'push')
                    and interaction['source_expires_at'] < interaction['deadline'] else None)
                self.last_control_lidar_at = snapshot.lidar_at
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
            if previous_state == 'active' and self.tasks.state in ('completed', 'failed'):
                if self.motion:
                    self.motion.stop('task_' + self.tasks.state)
                self._record_result()
            return self.task_error
        except PlanningRequired as exc:
            self._hold(str(exc))
            return self.task_error
        except (ValueError, RuntimeError) as exc:
            self.pause(str(exc))
            return str(exc)
