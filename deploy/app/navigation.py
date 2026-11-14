"""Real-sensor orchestration of the shared simulation-release algorithms.

The planner sees measured map-frame LiDAR and tracked object geometry.  This
module never imports the Isaac runner, scene layouts or simulator body handles.
Commands are body-frame (forward, lateral, yaw rate) for the SDK adapter.
"""
import math
from collections import deque
from dataclasses import replace

import numpy as np
from deploy.robot.safety import checked_velocity
from deploy.interaction.execution import InteractionExecution
from deploy.interaction.collision import ContactCollisionChecker
from deploy.interaction.profile import ContactProfile
from deploy.interaction.motion import SdkNavigationMotion
from deploy.interaction.skill_backend import create_push_skill
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState

from grutopia_extension.interactive_navigation.mapping import MappingConfig, OccupancyGridMap
from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ExplorationVoronoiRoutes, ForwardExplorationPathTracker, decision_recovery_required, body_motion_clear)
from grutopia_extension.interactive_navigation.go2_exploration_tracking import decision_buffer_recovery
from grutopia_extension.interactive_navigation.smooth_go2_navigation import (
    SmoothExplorationTracker, VelocitySlewLimiter, smooth_recovery_action)
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionCatalog
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill, PushAsideConfig
from grutopia_extension.interactive_navigation.go2_interaction_motion import (
    MeasuredStopGate, ObservedSubgoalArrival, checked_skill_action, navigation_recovery_enabled)
from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery
from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl, checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_passage_guidance import PassageEntryGuide
from grutopia_extension.interactive_navigation.object_inspection import ObjectInspectionPlanner
from grutopia_extension.interactive_navigation.local_path_guidance import local_guidance_path


class PlanningRequired(RuntimeError):
    """Yield before unbounded global work while a motion lease is active."""


class DeploymentNavigationMap(OnlineNavigationMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = deque(maxlen=2000)

    def robot_footprint_box(self, forward):
        geometry = getattr(self.planner, 'body_geometry', None)
        if geometry is None:
            return super().robot_footprint_box(forward)
        center = (geometry.x_min+geometry.x_max)/2
        return BoxState('__robot__', tuple(self.robot_xy+center*forward),
                        (geometry.x_max-geometry.x_min, 2*geometry.half_width, .5), np.degrees(self.robot_yaw))

    def measured_scan_masks(self, hit, free):
        if getattr(self, 'capture_contact_evidence', False):
            self.measured_hit, self.measured_free = hit.copy(), free.copy()

    def statistics(self):
        return {**super().statistics(),
                'movable_pose_source': getattr(self, 'interaction_source', 'measured_rgbd_geometry_admitted_by_current_lidar_visibility'),
                'uses_simulator_object_poses': False,
                'free_ray_model': 'height_clipped_measured_return_rays_only_no_synthetic_misses'}


class DeploymentVelocityLimiter(VelocitySlewLimiter):
    """Apply device limits before the release checks the resulting body sweep."""
    def __init__(self, limits, **kwargs):
        super().__init__(**kwargs)
        self.limits = limits

    def preview(self, target, now):
        return super().preview(checked_velocity(target, self.limits), now)

    def preview_directional_translation(self, target, now):
        return super().preview_directional_translation(checked_velocity(target, self.limits), now)


class NavigationEngine:
    """Keep one observed map while tasks replace their execution state."""

    def __init__(self, config, interaction=None):
        self.config = config
        self.interaction = interaction or {}
        self.interaction_source = self.interaction.get('source', 'rgbd')
        raw_profile = self.interaction.get('contact_profile')
        self.contact_profile = None if raw_profile is None else ContactProfile.from_dict(raw_profile)
        self.execution = InteractionExecution(self.contact_profile, self.interaction.get('max_skill_seconds', 180.))
        self.interaction_command = None
        if not 0. < float(config.get('success_distance', .25)) <= .25:
            raise ValueError('success_distance must be in (0, 0.25] for the release goal contract')
        # Exact measured rays are part of the release map contract.
        self.mapping_config = replace(MappingConfig(**config['mapping']), ray_traversal='exact')
        blank = OccupancyGridMap(self.mapping_config)
        self.planner = SidePushPlanner(
            np.zeros_like(blank.observed), [self.mapping_config.x_limits, self.mapping_config.y_limits],
            resolution=self.mapping_config.grid_resolution, robot_radius=self.mapping_config.robot_radius,
            passages={})
        self.planner.navigation_route_radius = float(config.get('route_clearance', .30))
        self.planner.navigation_half_width = float(config.get('navigation_half_width', .22))
        self.planner.voronoi_clearance_weight = float(config.get('voronoi_clearance_weight', .30))
        self.planner.full_navigation_footprint = True
        self.planner.layered_local_control = True
        self.planner.interactable_ids = frozenset()
        if self.contact_profile is not None:
            SdkNavigationMotion(self.contact_profile).bind(self.planner)
        self.layers = DeploymentNavigationMap(
            self.planner, self.mapping_config, floor_height=0.,
            local_hz=float(config.get('local_map_hz', 10.)), global_hz=float(config.get('global_map_hz', 1.)),
            local_radius=float(config.get('local_map_radius', 3.)))
        self.layers.capture_contact_evidence = self.contact_profile is not None
        self.contact_collision = (ContactCollisionChecker(self.mapping_config, self.planner, self.contact_profile)
                                  if self.contact_profile is not None else None)
        self.routes = ExplorationVoronoiRoutes(self.planner)
        self.catalog = InteractionCatalog()
        self.visible_at = {}
        self.latest_visible_ids = frozenset()
        self.current_proposals = {}
        self.observation = None
        self.map_observed_at = self.map_observation_id = None
        self.now = 0.
        self.events = deque(maxlen=2000)
        self.skill_reports = deque(maxlen=100)
        self.status = 'idle'
        self.goal = None
        self.result = None
        self.before_planning = None
        self.planner.before_contact_planning = self._planning_boundary
        self._reset_execution()
        self.skill_reports.clear()

    @property
    def path(self):
        if self.tracker is not None:
            return tuple(self.tracker._planned_path[self.tracker._waypoint_index:])
        return tuple(self.layers.global_path)

    @property
    def decision(self):
        return self.layers.decision

    def _inside_bounds(self, point):
        return all(low <= float(value) < high for value, (low, high) in zip(point[:2], self.planner.limits))

    def _clear_object_tracks(self):
        """Retire perception-epoch IDs without interpreting their poses as free.

        Tasks reset recognition IDs on start/resume/cancel. The release map
        keeps simulator IDs indefinitely, so its owned layer must be reset at
        that boundary. Unknown cells need fresh LiDAR before being traversed.
        """
        self._retire_object_tracks(set(self.layers.detected_boxes), 'perception_epoch_reset')
        self.visible_at.clear()
        self.latest_visible_ids = frozenset()
        self.current_proposals.clear()
        self.planner.interactable_ids = frozenset()

    def _retire_object_tracks(self, retired_ids, reason):
        layers = self.layers
        if not retired_ids:
            return
        retained = [box for key, box in layers.detected_boxes.items() if key not in retired_ids]
        removed = [box for key, box in layers.detected_boxes.items() if key in retired_ids]
        owned = layers._mask(removed)
        retained_sensor = layers._mask(retained)
        for box in removed:
            part = self.planner.box_mask(box, padding=.04)
            if part is not None:
                owned |= part
        for box in retained:
            part = self.planner.box_mask(box, padding=.04)
            if part is not None:
                retained_sensor |= part
        # A new measured identity can overlap an expired identity. Its live
        # object footprint remains occupied while only unowned cells retire.
        owned &= ~retained_sensor
        if owned.any():
            layers.evidence.log_odds[owned] = 0.
            layers.observed[owned] = False
            layers.global_observed[owned] = False
            layers.local_hits[owned] = False
            layers.hit_time[owned] = -math.inf
            layers.vacated_time[owned] = self.now
        layers.detected_boxes = {box.id: box for box in retained}
        layers.current_boxes = retained
        layers.global_boxes = [box for box in layers.global_boxes if box.id not in retired_ids]
        layers.visible_box_ids = [key for key in layers.visible_box_ids if key not in retired_ids]
        layers.tracked_occupied = layers._mask(retained)
        layers.tracked_sensor_mask = retained_sensor | layers.tracked_occupied
        layers._latest_static = layers.evidence.occupied_mask() & ~layers.tracked_occupied
        layers.refresh_global(reason=reason)
        layers._bind_local()
        for key in retired_ids:
            self.visible_at.pop(key, None)
        self.latest_visible_ids -= retired_ids
        self.planner.interactable_ids -= retired_ids
        layers.events.append({'event': 'object_tracks_retired', 'time_s': self.now,
                              'object_ids': sorted(retired_ids), 'reason': reason,
                              'cells_reset_to_unknown': int(owned.sum())})

    def _reset_execution(self):
        self.execution.reset()
        self.interaction_command = None
        self.selector = OnlineGoalSelector()
        self.velocity_limits = {
            'max_vx': float(self.config.get('max_forward_speed', .5)),
            'max_vy': float(self.config.get('max_lateral_speed', .18)),
            'max_wz': float(self.config.get('max_yaw_speed', .5)),
        }
        self.limiter = DeploymentVelocityLimiter(self.velocity_limits,
            linear_acceleration=float(self.config.get('linear_acceleration', .50)),
            yaw_acceleration=float(self.config.get('yaw_acceleration', 1.50)))
        self.tracker = None
        self.skill = None
        self.phase = 'idle'
        self.endpoint = None
        self.goal_kind = 'task'
        self.task_start = None
        self.next_decision = 0.
        self.next_reconsider = 0.
        self.last_replan = -math.inf
        self.goal_stop = MeasuredStopGate()
        self.sub_arrival = ObservedSubgoalArrival()
        self.subtracker = None
        self.subtarget = None
        self.sub_started = None
        self.sub_reached = False
        self.passage = None
        self.inspection_planner = ObjectInspectionPlanner()
        self.inspection = None
        self.inspection_turn = None
        self.inspection_started = None
        self.inspection_stop = MeasuredStopGate()
        self.layers.global_path = self.layers.local_path = ()
        self.layers.decision = None
        self.planner.passages = {}
        self.planner.active_passages = {}
        self.planner._global_buffer_exit = None
        self.last_tracking = None
        self.recovery_tracker = None
        self.recovery_started = None
        self.recovery_pending_settle = False
        hz = float(self.config.get('tick_hz', 10.))
        stable = max(2, int(math.ceil(.4*hz)))
        minimum = max(stable, int(math.ceil(.8*hz)))
        self.attitude = NavigationAttitudeRecovery(
            stable_ticks=stable, minimum_hold_ticks=minimum,
            maximum_hold_ticks=max(minimum+1, int(math.ceil(6.*hz))))

    def set_goal(self, goal, now):
        point = np.asarray(goal, float)
        if point.ndim != 1 or len(point) not in (2, 3) or not np.isfinite(point).all():
            raise ValueError('Goal must contain finite map coordinates')
        if not self._inside_bounds(point) or not self.planner.valid_cell(self.planner.cell(point)):
            raise ValueError('Goal is outside map bounds')
        self._clear_object_tracks()
        self._reset_execution()
        self.skill_reports.clear()
        self.goal = tuple(map(float, point))
        self.now = float(now)
        self.status, self.phase = 'running', 'decide'
        self.result = None

    def pause(self):
        # Resuming uses set_goal: a contact maneuver cannot survive a pause.
        self._reset_execution()
        self.status = 'idle'
        self.phase = 'paused'

    def cancel(self):
        self.pause()
        self._clear_object_tracks()
        self.goal = None
        self.result = None
        self.phase = 'idle'

    def _finish(self, success, reason):
        self.status = 'completed' if success else 'failed'
        self.phase = self.status
        self.result = {'success': bool(success), 'reason': reason,
                       'goal': self.goal, 'skill_reports': list(self.skill_reports)}
        self.limiter.accept(np.zeros(3), self.now)
        return None

    def observe(self, observation, now, boxes=()):
        """Update mapping and measured feedback without advancing a task."""
        self.now = float(now)
        self.observation = observation
        self.position = np.asarray(observation['position'], float)
        q = np.asarray(observation['orientation'], float)
        if self.position.shape != (3,) or q.shape != (4,) or not np.isfinite(self.position).all() or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError('Navigation requires a finite pose and wxyz orientation')
        self.orientation = q / np.linalg.norm(q)
        w, x, y, z = self.orientation
        self.heading = float(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        self.tilt = float(math.degrees(math.acos(np.clip(1-2*(x*x+y*y), -1., 1.))))
        supplied_linear = observation.get('body_linear_velocity')
        supplied_angular = observation.get('body_angular_velocity')
        self.full_motion_feedback = supplied_linear is not None and supplied_angular is not None
        planar = np.asarray(observation.get('velocity', (0., 0., 0.)), float)
        if planar.shape != (3,) or not np.isfinite(planar).all():
            raise ValueError('Navigation requires finite (vx, vy, wz) feedback')
        self.linear = np.asarray(supplied_linear if supplied_linear is not None else (*planar[:2], 0.), float)
        self.angular = np.asarray(supplied_angular if supplied_angular is not None else (0., 0., planar[2]), float)
        if self.linear.shape != (3,) or self.angular.shape != (3,) or not np.isfinite(self.linear).all() or not np.isfinite(self.angular).all():
            raise ValueError('Navigation motion feedback must be finite body-frame XYZ')
        self.limiter.observe([*self.linear[:2], self.angular[2]], self.now,
                             body_state=(self.position, self.linear, self.angular, self.tilt))
        # The SDK does not expose the release's internal 45D learned policy.
        self.limiter.policy_command = None
        self.planner.control_time = self.now
        self.planner.robot_heading = self.heading
        self.current_proposals = {box.id: box for box in boxes}
        previous_scan_count = self.layers.scan_count
        previous_scan = self.layers.last_scan
        # ROS clouds contain measured returns, not PhysX no-hit range shells.
        # The shared ray mapper reserves >=99% range for simulated misses, so
        # drop those real boundary returns instead of falsely clearing space.
        sensors = dict(observation.get('sensors', {}))
        lidar = sensors.get('lidar')
        if lidar is not None:
            points = np.asarray(lidar.get('pointcloud', ()), float)
            if points.size:
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError('LiDAR requires finite measured N x 3 points')
                origin = np.asarray(lidar.get('position', self.position), float)
                valid = np.linalg.norm(points-origin, axis=1) < .99*float(lidar.get('max_range', 5.))
                sensors['lidar'] = {**lidar, 'pointcloud': points[valid]}
        updated = self.layers.update(self.now, {**observation, 'sensors': sensors}, tuple(self.current_proposals.values()), force=self.layers.local_updates == 0)
        if self.contact_collision is not None:
            self.contact_collision.update({**observation, 'sensors': sensors}, self.layers)
        if self.layers.scan_count != previous_scan_count:
            # Map processing time is not the source observation time. Keep
            # provenance only for the scan actually admitted by the mapper.
            stamp = observation.get('lidar_captured_at')
            valid_stamp = (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                           and math.isfinite(stamp) and stamp <= self.now)
            self.map_observed_at = float(stamp) if valid_stamp else None
            self.map_observation_id = self.layers.last_scan
            self.latest_visible_ids = frozenset(self.layers.visible_box_ids)
            self.visible_at.update({object_id: self.now for object_id in self.latest_visible_ids})
        elif updated and lidar is not None and (lidar.get('physics_step') is None or lidar.get('physics_step') != previous_scan):
            # A new real scan can contain only discarded boundary returns.
            # It provides no current target visibility despite no map ingest.
            self.latest_visible_ids = frozenset()
        stale_ids = {key for key in self.layers.detected_boxes
                     if not self._fresh_object(key)
                     and self.now-self.visible_at.get(key, -math.inf) > float(self.config.get('object_max_age_s', 1.))}
        self._retire_object_tracks(stale_ids, 'object_geometry_expired')
        return self.layers

    def _fresh_object(self, object_id):
        age = self.now-self.visible_at.get(object_id, -math.inf)
        return (object_id in self.current_proposals and object_id in self.latest_visible_ids
                and 0. <= age <= float(self.config.get('object_max_age_s', 1.)))

    def _permission(self, object_id, recognition):
        if not self.config.get('interaction_enabled', False) or not self.full_motion_feedback or not self._fresh_object(object_id) or recognition is None:
            return None
        if self.interaction_source == 'registered_box_lidar':
            from deploy.perception.registered_boxes import RegisteredBoxPerception
            if not isinstance(recognition, RegisteredBoxPerception):
                return None
            permission = recognition.permission(object_id, self.now)
            if permission is None or permission.source != 'registered_box_lidar':
                return None
            return permission
        item = recognition.confirmed(object_id, self.now)
        if item is None:
            return None
        max_age = float(self.config.get('recognition_max_age_s', 10.))
        fresh = [e for e in item.evidence if 0. <= self.now-e.observed_at <= max_age]
        if len({e.frame_id for e in fresh}) < 2 or any(e.source != 'rgb_dino_qwen_vl' for e in fresh):
            return None
        permission = self.catalog.permission(object_id, 'push_aside', recognition, self.now)
        return permission if permission is not None and permission.source == 'rgb_dino_qwen_vl' else None

    def _output(self, velocity):
        command = np.array(velocity, float, copy=True)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Navigation emitted an invalid velocity')
        limited = checked_velocity(command, self.velocity_limits)
        if not np.array_equal(command, limited):
            # Normal tracking already limits BEFORE its collision check. If a
            # skill bypasses preview, independent yaw clipping changes its arc;
            # check that actual bounded arc instead of assuming it stays safe.
            horizon = .60+max(np.linalg.norm(limited[:2]), np.linalg.norm(self.limiter.measured[:2]))/(2*self.limiter.linear_acceleration)
            clear = self.observation is not None and body_motion_clear(
                self.planner, self.layers.current_boxes, self.position[:2], self.heading,
                limited[0], limited[2], horizon=horizon, lateral=limited[1])
            if not clear:
                self.events.append({'event': 'velocity_limit_blocked', 'time_s': self.now,
                                    'requested': command.tolist(), 'limited': limited.tolist()})
                limited = np.zeros(3)
        command = limited
        if self.skill is not None and self.contact_profile is not None:
            command = self.contact_profile.limit(command)
            # Includes nonzero braking produced AFTER the shared skill's stop
            # intent, and any final limit-induced change in its checked arc.
            if np.any(command) and getattr(self.skill, 'contact_started', None) is not None:
                context = getattr(self, '_contact_context', None)
                if context is None or not self._check_contact_motion(command, context):
                    return self._finish(False, 'push_aside:contact_motion_blocked')
        self.limiter.accept(command, self.now)
        return command

    def _check_contact_motion(self, command, context):
        if self.contact_collision is None:
            return False
        pushing = self.skill is not None and self.skill.state in ('engage', 'push', 'settle_push_progress', 'brake_after_push')
        return self.contact_collision(command, context, pushing=pushing)

    def _interaction_lease(self, recognition):
        if not self.execution.active or self.skill is None or self.contact_profile is None:
            return
        if self.interaction_source == 'registered_box_lidar':
            expiry = recognition.expires_at(self.skill.plan.box_id)
        else:
            item = recognition.confirmed(self.skill.plan.box_id, self.now)
            expiry = None if item is None else item.latest_at+float(self.config.get('object_max_age_s', .75))
        self.interaction_command = self.execution.metadata(self.skill.state, self.now, expiry)

    def interaction_stopped(self):
        completed = self.execution.awaiting_stop
        self.execution.stopped()
        self.interaction_command = None
        if completed:
            self.velocity_limits = dict(max_vx=float(self.config['max_forward_speed']),
                                        max_vy=float(self.config['max_lateral_speed']),
                                        max_wz=float(self.config['max_yaw_speed']))
            self.limiter.limits = self.velocity_limits
            self.limiter.linear_acceleration = float(self.config['linear_acceleration'])
            self.limiter.yaw_acceleration = float(self.config['yaw_acceleration'])
            self.limiter.accept(np.zeros(3), self.now)

    def _brake(self):
        command, _ = checked_contact_turn_command(
            [0., 0., 0.], self.position, self.heading, self.planner,
            self.layers.current_boxes, self.limiter, self.now)
        return self._output(command)

    def _route(self, target):
        self._planning_boundary('global_route')
        with self.layers.global_planning() as boxes:
            path = self.routes.plan(self.position[:2], target, boxes)
        self.layers.global_path = path
        self.layers.local_path = ()
        self.layers.guidance_revision = self.layers.global_revision
        return path

    def _planning_boundary(self, reason):
        # The live controller installs this hook; offline replay/simulation
        # keep their synchronous algorithm. A yield occurs BEFORE mutations.
        if self.before_planning is not None:
            self.before_planning(reason)

    def _tracker(self, path, *, short_goal=False):
        return SmoothExplorationTracker(
            self.position[:2], path, self.limiter,
            max_forward=float(self.config.get('max_forward_speed', .5)),
            max_lateral=float(self.config.get('max_lateral_speed', .18)),
            final_goal_slow_radius=.30 if short_goal else float(self.config.get('final_goal_slow_radius', .75)),
            final_task_goal=None if short_goal else self.goal[:2])

    def _reset_subnavigation(self):
        self.subtracker = self.subtarget = self.sub_started = None
        self.sub_reached = False
        self.sub_arrival.reset()

    def _subnavigation(self, target):
        target = np.asarray(target[:2], float)
        if self.subtarget is None or np.linalg.norm(target-self.subtarget) > .02:
            self._reset_subnavigation()
            self.subtarget, self.sub_started = target.copy(), self.now
        if self.now-self.sub_started > float(self.config.get('subnavigation_timeout_s', 80.)):
            return self._output(np.zeros(3)), False, 'subnavigation_timeout'
        arrival = self.sub_arrival.update(target, self.position, self.linear, self.angular, self.tilt, self.now,
                                         heading=self.heading, command=self.limiter.command)
        if arrival['replan']:
            self.subtracker = None
        if arrival['subgoal_braking']:
            if arrival['subgoal_reached']:
                self.limiter.clear_guidance(self.position, self.now)
            return self._brake(), arrival['subgoal_reached'], None
        if self.subtracker is None:
            path = self._route(target)
            if not path:
                return self._brake(), False, None
            self.subtracker = self._tracker(path, short_goal=True)
        action, detail = self.subtracker.action(self.position[:2], self.heading, self.planner, self.layers.current_boxes)
        self.last_tracking = detail
        if detail.get('needs_replan'):
            self.subtracker = None
        return self._output(action['move_by_speed']), False, None

    def _skill_step(self, recognition):
        object_id = self.skill.plan.box_id
        # Never let cached map geometry substitute for a live contact target.
        permission = self._permission(object_id, recognition)
        if permission is None:
            return self._finish(False, 'push_aside:live_geometry_or_recognition_lost')
        target = self.current_proposals[object_id]
        current_boxes = [target if box.id == object_id else box for box in self.layers.current_boxes]
        source = recognition.snapshot(self.now).get(object_id, {})
        context = SkillContext(
            self.now, self.position, self.heading, self.linear, float(self.angular[2]), self.tilt,
            self.tilt < 30. and self.position[2] >= float(self.config.get('fall_height', .16)),
            target, current_boxes, self.planner, self.layers, self.limiter, self.sub_reached,
            body_angular_velocity=self.angular, orientation=self.orientation, policy_command=None,
            contact_motion_check=self._check_contact_motion if self.contact_profile is not None else None,
            box_observed_at=source.get('latest_at'),
            box_observation_id=permission.evidence_frames[-1] if permission.evidence_frames else None,
            map_observed_at=self.map_observed_at, map_observation_id=self.map_observation_id)
        self._contact_context = context
        command = self.skill.step(context)
        self._interaction_lease(recognition)
        self.events.extend(command.events)
        self.layers.decision = {**(self.layers.decision or {}), 'skill': 'push_aside',
                                'skill_state': command.state, 'interaction_allowed': True,
                                'execution_state': command.state}
        self.sub_reached = False
        if command.status != 'running':
            report = self.skill.report()
            report['object_pose_feedback'] = self.interaction_source
            self.skill_reports.append(report)
            if command.status != 'succeeded':
                return self._finish(False, 'push_aside:'+command.reason)
            if command.certificate is None:
                return self._finish(False, 'push_aside:missing_passage_certificate')
            self.passage = PassageEntryGuide(command.certificate, self.goal[:2])
            self.layers.refresh_global(reason='skill_success_replan_through_passage')
            self.execution.completed()
            self.skill = None
            self.tracker = None
            self._reset_subnavigation()
            self.phase = 'decide'
            return self._output(np.zeros(3))
        if command.navigation_goal is not None:
            velocity, self.sub_reached, failed = self._subnavigation(command.navigation_goal)
            return self._finish(False, 'push_aside:'+failed) if failed else velocity
        self._reset_subnavigation()
        action = checked_skill_action(command, context, forward_stop_feedback=False,
                                      recovery_from_contact=self.skill.recovery_from_contact)
        return self._output(action['move_by_speed'])

    def _inspect_step(self, recognition):
        if self.phase == 'inspect_navigate':
            velocity, arrived, failed = self._subnavigation(self.inspection.position)
            if failed:
                self.phase = 'decide'
            elif arrived:
                self.phase = 'inspect_turn'
                self.inspection_started = self.now
                self.inspection_stop.reset()
            return velocity
        settled = self.inspection_stop.update(self.position, self.linear, self.angular, self.tilt, self.now)
        if self.phase == 'inspect_turn':
            if self.inspection_turn is None:
                if not settled:
                    return self._brake()
                self.inspection_turn = ContactTurnControl(self.position, self.heading, self.inspection.heading,
                                                         self.planner, self.layers.current_boxes, self.now)
            answer = self.inspection_turn.update(self.position, self.heading, self.planner,
                                                 self.layers.current_boxes, self.limiter, self.now)
            error = (self.inspection.heading-self.heading+math.pi) % (2*math.pi)-math.pi
            if abs(error) < .10:
                self.phase = 'inspect_wait'
                self.inspection_started = self.now
                self.inspection_stop.reset()
                return self._brake()
            if self.inspection_turn.failed or self.now-self.inspection_started > 40.:
                self.phase = 'decide'
                return self._brake()
            return self._output(np.zeros(3) if answer is None else answer[0])
        item = recognition.confirmed(self.inspection.object_id, self.now) if recognition is not None else None
        if item is not None and len({e.frame_id for e in item.evidence if e.observed_at >= self.inspection_started}) >= 2 and settled:
            self.inspection_planner.review(self.inspection.object_id, item.category, self.now)
            self.phase = 'decide'
        elif self.now-self.inspection_started > float(self.config.get('inspection_timeout_s', 20.)):
            self.phase = 'decide'
        return self._brake()

    def _decide(self, recognition):
        if self.now < self.next_decision:
            return self._brake()
        self.layers.refresh_global(reason='decision')
        # Match the release's decision-time body recovery before interpreting
        # a blocked circular-plan seed as an unreachable task.
        current = self.layers.current_boxes
        recovery = None
        if decision_recovery_required(self.planner, current, self.position[:2], self.heading, self.recovery_tracker):
            if self.recovery_tracker is None:
                self.recovery_tracker = ForwardExplorationPathTracker(
                    self.position[:2], [self.goal[:2]], guard_motion=False)
            recovery = self.recovery_tracker.action(self.position[:2], self.heading, self.planner, current)
        else:
            self.recovery_tracker = None
        if recovery is not None and recovery[1].get('needs_replan'):
            recovery = None
        if recovery is None:
            self._planning_boundary('decision_buffer_recovery')
            local_free = self.planner.free.copy()
            with self.layers.global_planning() as global_boxes:
                recovery = decision_buffer_recovery(
                    self.position[:2], self.heading, self.planner, global_boxes, self.goal[:2], self.heading,
                    local_free=local_free, local_boxes=current, require_goal_connection=False)
        if recovery is not None:
            if self.recovery_started is None:
                self.recovery_started = self.now
            if self.now-self.recovery_started > float(self.config.get('recovery_timeout_s', 20.)):
                return self._finish(False, 'decision_buffer_recovery_timeout')
            action, self.last_tracking = recovery
            self.recovery_pending_settle = True
            action = smooth_recovery_action(action, self.position[:2], self.heading, self.planner,
                                            current, self.limiter, self.now)
            self.layers.decision = {'kind': 'recover', 'reason': 'observed_body_buffer_recovery'}
            return self._output(action['move_by_speed'])
        self.recovery_started = None
        if self.recovery_pending_settle:
            if not self.limiter.recovery_gate.leave_recovery(self.limiter.measured, self.now,
                                                           body_state=self.limiter.body_state, command=self.limiter.command):
                return self._output(np.zeros(3))
            self.recovery_pending_settle = False
            self.limiter.clear_guidance(self.position, self.now)
        if self.passage is not None:
            if not self.passage.entry_reached:
                arrived = self.passage.observe_entry(self.position, self.linear, self.angular, self.tilt,
                                                     self.now, heading=self.heading, command=self.limiter.command)
                if arrived['subgoal_braking'] and not arrived['entry_advanced']:
                    return self._brake()
            self._planning_boundary('passage_route')
            with self.layers.global_planning() as boxes:
                path = self.passage.path(self.planner, self.position[:2], boxes)
            if path:
                self.endpoint = tuple(self.passage.target)
                self.goal_kind = 'task' if self.passage.entry_reached else 'passage_entry'
                self.layers.global_path = path
                self.tracker = self._tracker(path, short_goal=self.goal_kind != 'task')
                self.phase = 'navigate'
                return self._brake()
            # A changed map invalidates execution of the old certificate.
            self.passage = None
        if (self.interaction_source == 'rgbd' and self.config.get('interaction_enabled', False)
                and recognition is not None and self.full_motion_feedback):
            self._planning_boundary('object_inspection')
            inspection = self.inspection_planner.propose(self.layers, self.position[:2], self.heading,
                                                        self.goal[:2], recognition, self.now)
            if inspection is not None:
                self.inspection = inspection
                self.inspection_turn = None
                self.inspection_started = self.now
                self._reset_subnavigation()
                self.phase = 'inspect_navigate'
                self.layers.decision = {'kind': 'inspect', 'reason': inspection.reason,
                                        'object_id': inspection.object_id, 'interaction_allowed': False}
                return self._brake()
        self._planning_boundary('global_decision')
        decision, endpoint = self.selector.decide(self.layers, self.position[:2], self.goal[:2])
        self.layers.decision = {**(decision.explanation or {}), 'kind': decision.kind,
                                'reason': decision.reason, 'decision_time_s': self.now}
        self.events.append({'event': 'decision', 'time_s': self.now, 'kind': decision.kind, 'reason': decision.reason})
        if decision.kind == 'failed':
            return self._finish(False, decision.reason)
        if decision.kind == 'wait':
            self.next_decision = self.now+self.layers.global_period
            return self._brake()
        if decision.kind == 'push':
            permission = self._permission(decision.push.box_id, recognition)
            if permission is None:
                self.next_decision = self.now+self.layers.global_period
                return self._brake()
            speed = min(float(self.config.get('push_max_speed', .30)), float(self.config.get('max_forward_speed', .5)))
            skill_config = PushAsideConfig(
                push_start_speed=min(float(self.config.get('push_start_speed', .15)), speed), push_max_speed=speed,
                lateral_speed=min(.35, float(self.config.get('max_lateral_speed', .18))),
                forward_speed=min(.25, float(self.config.get('max_forward_speed', .5))),
                reverse_speed=.35,
                max_skill_seconds=float(self.interaction.get('max_skill_seconds', 180.)))
            self.skill = (PushAsideSkill(permission, decision.push, self.task_start, self.goal[:2], config=skill_config)
                          if self.contact_profile is None else
                          create_push_skill(permission, decision.push, self.task_start, self.goal[:2],
                                            self.contact_profile, config=skill_config))
            if self.contact_profile is not None:
                self.execution.start(decision.push.box_id, self.now)
                self._interaction_lease(recognition)
                self.velocity_limits = {
                    'max_vx': min(self.velocity_limits['max_vx'], max(self.contact_profile.max_forward_mps, self.contact_profile.max_reverse_mps)),
                    'max_vy': min(self.velocity_limits['max_vy'], self.contact_profile.max_lateral_mps),
                    'max_wz': min(self.velocity_limits['max_wz'], self.contact_profile.max_yaw_rps)}
                self.limiter.limits = self.velocity_limits
                self.limiter.linear_acceleration = self.contact_profile.linear_acceleration_mps2
                self.limiter.yaw_acceleration = self.contact_profile.yaw_acceleration_rps2
            self.phase = 'skill'
            self._reset_subnavigation()
            return self._output(np.zeros(3))
        self.endpoint = tuple(endpoint)
        self.goal_kind = 'frontier' if decision.kind == 'explore' else 'task'
        path = self._route(endpoint)
        if not path:
            if self.goal_kind == 'frontier':
                self.selector.defer_frontier(endpoint, self.now)
            self.next_decision = self.now+self.layers.global_period
            return self._brake()
        self.tracker = self._tracker(path, short_goal=self.goal_kind != 'task')
        self.phase = 'navigate'
        self.last_replan = self.now
        return self._brake()

    def update(self, observation, now, boxes=(), recognition=None):
        self.observe(observation, now, boxes)
        self.planner.interactable_ids = frozenset(
            box.id for box in self.layers.current_boxes if self._permission(box.id, recognition) is not None)
        self.layers.interactable_ids = self.planner.interactable_ids
        self.layers.interaction_source = ('registered_box_lidar' if self.interaction_source == 'registered_box_lidar' else 'rgb_dino_qwen_vl')
        self.layers.recognized_objects = {} if recognition is None else recognition.snapshot(self.now)
        if self.status != 'running':
            return None
        if self.execution.awaiting_stop:
            # Observe while the writer/recorded-stop gate completes the old
            # interaction. A new decision must not replace its execution id.
            return self._output(np.zeros(3))
        if self.task_start is None:
            self.task_start = tuple(self.position[:2])
        if not self._inside_bounds(self.position) or not self.planner.valid_cell(self.planner.cell(self.position[:2])):
            return self._finish(False, 'robot_outside_map_bounds')
        if self.tilt >= 30. or self.position[2] < float(self.config.get('fall_height', .16)):
            return self._finish(False, 'robot_unstable')
        if self.phase == 'skill' and self._permission(self.skill.plan.box_id, recognition) is None:
            return self._finish(False, 'push_aside:live_geometry_or_recognition_lost')
        if self.phase == 'skill':
            self._interaction_lease(recognition)
        attitude = self.attitude.update(
            self.position, self.orientation,
            enabled=navigation_recovery_enabled(self.phase, None if self.skill is None else self.skill.state))
        if attitude['state'] == 'failed':
            return self._finish(False, 'attitude_recovery_timeout')
        if attitude['state'] == 'hold':
            self.layers.decision = {**(self.layers.decision or {}), 'execution_state': 'attitude_recovery_hold'}
            return self._output(np.zeros(3))
        if attitude['state'] == 'released':
            self.limiter.clear_guidance(self.position, self.now)
            self.tracker = None
            self._reset_subnavigation()
            if self.phase == 'navigate':
                self.phase = 'decide'
        if self.phase == 'skill':
            return self._skill_step(recognition)
        if self.phase.startswith('inspect_'):
            return self._inspect_step(recognition)
        distance = float(np.linalg.norm(self.position[:2]-np.asarray(self.goal[:2])))
        tolerance = float(self.config.get('success_distance', .25))
        if distance < tolerance:
            if self.goal_stop.update(self.position, self.linear, self.angular, self.tilt, self.now):
                return self._finish(True, 'goal_reached_and_stopped')
            return self._brake()
        self.goal_stop.reset()
        if self.phase == 'decide':
            return self._decide(recognition)
        if self.goal_kind == 'passage_entry' and self.passage is not None:
            arrived = self.passage.observe_entry(self.position, self.linear, self.angular, self.tilt, self.now,
                                                 heading=self.heading, command=self.limiter.command)
            if arrived['entry_advanced']:
                self.tracker = None
                self.limiter.clear_guidance(self.position, self.now)
                self.phase = 'decide'
                return self._brake()
            if arrived['subgoal_braking']:
                return self._brake()
        if self.goal_kind == 'frontier':
            close = np.linalg.norm(self.position[:2]-np.asarray(self.endpoint)) < .35
            reconsider = close
            if self.now >= self.next_reconsider and not self.tracker.staging_active:
                self._planning_boundary('frontier_reconsideration')
                reconsider |= self.selector.task_decision(self.layers, self.position[:2], self.goal[:2]) is not None
                self.next_reconsider = self.now+1.
            if reconsider:
                if close:
                    self.selector.reached_frontier(self.endpoint, self.layers.observed.sum())
                self.phase = 'decide'
                self.tracker = None
                return self._brake()
        action, detail = self.tracker.action(self.position[:2], self.heading, self.planner, self.layers.current_boxes)
        self.last_tracking = detail
        if detail.get('needs_replan') and self.now-self.last_replan >= self.layers.global_period:
            if self.goal_kind == 'frontier':
                self.selector.consume_execution_failure(self.endpoint, self.layers.observed.sum(), detail)
            self.tracker = None
            self.phase = 'decide'
            self.last_replan = self.now
            return self._brake()
        if self.now-self.last_replan >= self.layers.global_period and not self.tracker.staging_active:
            path, details = local_guidance_path(self.planner, self.position[:2], self.layers.global_path,
                                               self.layers.current_boxes, self.heading, self.layers.local_radius)
            self.layers.local_path = details.get('local_prefix', ())
            if path:
                self.tracker = self._tracker(path, short_goal=self.goal_kind != 'task')
                self.layers.local_replans += 1
            self.last_replan = self.now
        return self._output(action['move_by_speed'])
