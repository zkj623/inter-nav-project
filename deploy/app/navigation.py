"""Real-sensor orchestration of the shared simulation-release algorithms.

The planner sees measured map-frame LiDAR and tracked object geometry.  This
module never imports the Isaac runner, scene layouts or simulator body handles.
Commands are body-frame (forward, lateral, yaw rate) for the SDK adapter.
"""
import math
from collections import deque
from dataclasses import replace

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from deploy.robot.safety import checked_velocity
from deploy.interaction.execution import InteractionExecution
from deploy.interaction.collision import ContactCollisionChecker
from deploy.interaction.profile import ContactProfile
from deploy.interaction.motion import SdkNavigationMotion
from deploy.interaction.skill_backend import create_push_skill
from deploy.interaction.first_push import RegisteredFacePlanner
from deploy.robot.observation import navigation_lidar
from deploy.robot.world_boundary import WorldBoundary
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState

from grutopia_extension.interactive_navigation.mapping import MappingConfig, OccupancyGridMap
from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ExplorationVoronoiRoutes, ForwardExplorationPathTracker, decision_recovery_required, body_motion_clear,
    route_blocked, navigation_footprint_samples)
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
    def __init__(self, *args, world_boundary=None, **kwargs):
        self.world_boundary = world_boundary
        self.interior = None
        super().__init__(*args, **kwargs)
        self.events = deque(maxlen=2000)
        if world_boundary is not None:
            self.interior = world_boundary.interior_mask(self.config, self.observed.shape)
            wall = binary_dilation(self.interior, structure=np.ones((3, 3), bool)) & ~self.interior
            clearance = distance_transform_edt(self.interior)*self.planner.resolution
            self.boundary_goal_blocked = clearance <= self.planner.navigation_route_radius
            self.evidence.static_occupied[~self.interior] = True
            self._latest_static[~self.interior] = True
            self.static_occupied[~self.interior] = True
            self.observed[wall] = self.global_observed[wall] = True
            self._bind_local()

    def _mask(self, boxes):
        mask = super()._mask(boxes)
        if self.interior is not None:
            mask &= self.interior
        return mask

    def robot_footprint_box(self, forward):
        geometry = getattr(self.planner, 'body_geometry', None)
        if geometry is None:
            return super().robot_footprint_box(forward)
        center = (geometry.x_min+geometry.x_max)/2
        return BoxState('__robot__', tuple(self.robot_xy+center*forward),
                        (geometry.x_max-geometry.x_min, 2*geometry.half_width, .5), np.degrees(self.robot_yaw))

    def measured_scan_masks(self, hit, free):
        if self.interior is not None:
            hit &= self.interior
            free &= self.interior
        if getattr(self, 'capture_contact_evidence', False):
            self.measured_hit, self.measured_free = hit.copy(), free.copy()

    def statistics(self):
        return {**super().statistics(),
                'world_boundary': ({'x_limits': self.world_boundary.x, 'y_limits': self.world_boundary.y}
                                   if self.world_boundary is not None else None),
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

    def __init__(self, config, interaction=None, box_memory=None):
        self.config = config
        self.interaction = interaction or {}
        self.interaction_source = self.interaction.get('source', 'rgbd')
        self.box_memory = box_memory
        self.registered_ids = {row['id'] for row in self.interaction.get('boxes', ())}
        raw_profile = self.interaction.get('contact_profile')
        self.contact_profile = None if raw_profile is None else ContactProfile.from_dict(raw_profile)
        self.first_push = self.contact_profile is not None and self.contact_profile.mode == 'first_push'
        self.trial_used = False
        self.trial_report = None
        self.pushable_ids = {row['id'] for row in self.interaction.get('boxes', ()) if row['push_allowed']}
        self.execution = InteractionExecution(self.contact_profile, self.interaction.get('max_skill_seconds', 180.))
        self.interaction_command = None
        if not 0. < float(config.get('success_distance', .25)) <= .25:
            raise ValueError('success_distance must be in (0, 0.25] for the release goal contract')
        # Exact measured rays are part of the release map contract.
        self.mapping_config = replace(MappingConfig(**config['mapping']), ray_traversal='exact')
        self.world_boundary = WorldBoundary.from_navigation(config)
        if self.world_boundary is not None:
            self.mapping_config = self.world_boundary.grid_config(self.mapping_config)
        blank = OccupancyGridMap(self.mapping_config)
        planner_type = RegisteredFacePlanner if self.first_push else SidePushPlanner
        self.planner = planner_type(
            np.zeros_like(blank.observed), [self.mapping_config.x_limits, self.mapping_config.y_limits],
            resolution=self.mapping_config.grid_resolution, robot_radius=self.mapping_config.robot_radius,
            passages={})
        self.planner.navigation_route_radius = float(config.get('route_clearance', .30))
        self.planner.navigation_half_width = float(config.get('navigation_half_width', .22))
        self.planner.voronoi_clearance_weight = float(config.get('voronoi_clearance_weight', .30))
        self.planner.full_navigation_footprint = True
        self.planner.layered_local_control = True
        self.planner.interactable_ids = frozenset()
        if self.contact_profile is not None and not self.first_push:
            SdkNavigationMotion(self.contact_profile).bind(self.planner)
        self.navigation_motion = {name: getattr(self.planner, name) for name in
            ('motion_backend', 'body_geometry', 'release_braking_margin', 'contact_offset') if hasattr(self.planner, name)}
        if self.world_boundary is not None:
            xx, yy = navigation_footprint_samples(self.planner)
            self.body_corners = np.array([(x, y) for x in (xx.min(), xx.max()) for y in (yy.min(), yy.max())])
        self.layers = DeploymentNavigationMap(
            self.planner, self.mapping_config, floor_height=0., world_boundary=self.world_boundary,
            local_hz=float(config.get('local_map_hz', 10.)), global_hz=float(config.get('global_map_hz', 1.)),
            local_radius=float(config.get('local_map_radius', 3.)))
        self.layers.capture_contact_evidence = self.contact_profile is not None
        self.contact_collision = (ContactCollisionChecker(self.mapping_config, self.planner, self.contact_profile,
                                                          world_boundary=self.world_boundary)
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
        self.planner.before_turn_search = self._planning_boundary
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
        if self.world_boundary is not None:
            return self.world_boundary.contains(point)
        return all(low <= float(value) < high for value, (low, high) in zip(point[:2], self.planner.limits))

    def check_boundary_goal(self, point):
        if self.world_boundary is None:
            return
        if not self.world_boundary.contains(point):
            raise ValueError('目标超出实验边界的可用范围')
        if self.layers.boundary_goal_blocked[self.planner.cell(point[:2])]:
            raise ValueError(f'目标太靠近实验边界，请向内移。当前路线预留 {self.planner.navigation_route_radius:g} 米净空')

    def _clear_object_tracks(self):
        """Reset current-scan permissions. Registry IDs belong to the experiment map."""
        self._retire_object_tracks(set(self.layers.detected_boxes)-self.registered_ids, 'perception_epoch_reset')
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
        if layers.interior is not None:
            owned &= layers.interior
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
        self._restore_navigation_motion()
        self.execution.reset()
        self.interaction_command = None
        self.selector = OnlineGoalSelector()
        self._failed_push_probe = None
        self.velocity_limits = {
            'max_vx': float(self.config.get('max_forward_speed', .5)),
            'max_vy': float(self.config.get('max_lateral_speed', .18)),
            'max_wz': float(self.config.get('max_yaw_speed', .5)),
        }
        self.limiter = DeploymentVelocityLimiter(self.velocity_limits,
            linear_acceleration=float(self.config.get('linear_acceleration', .50)),
            yaw_acceleration=float(self.config.get('yaw_acceleration', 1.50)))
        self.tracker = None
        self._prepared_tracker = None
        self.skill = None
        self.box_approach = None
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
        self.check_boundary_goal(point)
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
        if not success and self.first_push and self.skill is not None and self.trial_report is None:
            self.trial_report = dict(self.skill.report(), status='failed', reason=reason)
            self.events.append(dict(event='first_push_result', time_s=self.now, report=self.trial_report))
            if not self.skill_reports or self.skill_reports[-1].get('object_id') != self.skill.plan.box_id:
                self.skill_reports.append(self.trial_report)
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
        self.current_proposals = {box.id: box for box in boxes
                                  if self.world_boundary is None or self.world_boundary.contains_box(
                                      box.xy, box.dimensions, math.radians(box.yaw))}
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
                cropped = navigation_lidar(lidar, points, self.layers.config.point_height, self.world_boundary)
                sensors['lidar'] = {**lidar, 'pointcloud': cropped.measured_returns(self.position),
                                    'measured_returns_only': True}
        self.observation = {**observation, 'sensors': sensors}
        if self.box_memory is not None:
            for key, location in self.box_memory.locations(self.now).items():
                if key in self.registered_ids and (self.world_boundary is None or self.world_boundary.contains_box(
                        location.center, location.box.local_dimensions, location.yaw_rad)):
                    self.layers.detected_boxes[key] = location.box
        updated = self.layers.update(self.now, self.observation, tuple(self.current_proposals.values()), force=self.layers.local_updates == 0)
        if self.contact_collision is not None:
            self.contact_collision.update(self.observation, self.layers)
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
                     if key not in self.registered_ids and not self._fresh_object(key)
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
            from deploy.perception.registered_boxes import RegisteredBoxPerception, RegisteredBoxEvidence
            if not isinstance(recognition, (RegisteredBoxPerception, RegisteredBoxEvidence)):
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
            if self.first_push and self.skill.contact_started is not None:
                command[:2] /= max(1., np.linalg.norm(command[:2])/self.contact_profile.push_max_mps)
            # Includes nonzero braking produced AFTER the shared skill's stop
            # intent, and any final limit-induced change in its checked arc.
            if np.any(command) and getattr(self.skill, 'contact_started', None) is not None:
                context = getattr(self, '_contact_context', None)
                if context is None or not self._check_contact_motion(command, context):
                    return self._finish(False, 'push_aside:contact_motion_blocked')
        if self.world_boundary is not None:
            # Cover both the issued twist and feedback during reaction/braking,
            # including turns in place and recovery commands that may allow unknown space.
            for twist in (command, self.limiter.measured):
                horizon = .60 + max(math.hypot(*twist[:2])/(2*self.limiter.linear_acceleration),
                                    abs(twist[2])/(2*self.limiter.yaw_acceleration))
                if not self.world_boundary.contains_motion(self.position[:2], self.heading, twist,
                                                           horizon, self.body_corners):
                    return self._finish(False, 'world_boundary_stop')
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
        if self.first_push and self.skill.contact_started is not None:
            p = self.contact_profile
            self.interaction_command['limits'].update(max_vx=p.push_max_mps, max_vy=p.push_max_mps)
            if self.skill.near_at is not None:
                self.interaction_command['deadline'] = min(self.interaction_command['deadline'],
                                                          self.skill.near_at+self.skill.near_seconds)

    def _restore_navigation_motion(self):
        if self.first_push:
            for name in ('motion_backend', 'body_geometry', 'release_braking_margin', 'contact_offset'):
                if name in self.navigation_motion:
                    setattr(self.planner, name, self.navigation_motion[name])
                elif hasattr(self.planner, name):
                    delattr(self.planner, name)

    def interaction_stopped(self):
        completed = self.execution.awaiting_stop
        self.execution.stopped()
        self.interaction_command = None
        if completed and self.phase != 'first_push_wait_scan':
            self._restore_after_contact()

    def _restore_after_contact(self):
        self._restore_navigation_motion()
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

    def _tracker_route_start_blocked(self, boxes):
        """Whether the route tracker will reject a segment from this pose.

        SmoothExplorationTracker checks route segments with the global route
        clearance applied to the current (local, 10 Hz) map. The decision's
        buffer exit uses the 1 Hz global map. When only the fresher local map
        puts the robot cell inside that clearance, the decision keeps choosing
        an ordinary route that the tracker refuses to start, forever.
        """
        scope = getattr(self.planner, 'planning_scope', 'local')
        try:
            self.planner.planning_scope = 'global'
            return bool(route_blocked(self.planner, boxes)[self.planner.cell(self.position[:2])])
        finally:
            self.planner.planning_scope = scope

    def _tracker_view_buffer_exit(self, boxes, local_free):
        """Run the same body-checked buffer exit in the tracker's own view."""
        scope = getattr(self.planner, 'planning_scope', 'local')
        try:
            self.planner.planning_scope = 'global'
            return decision_buffer_recovery(
                self.position[:2], self.heading, self.planner, boxes, self.goal[:2], self.heading,
                local_free=local_free, local_boxes=boxes, require_goal_connection=False,
                before_search=lambda: self._planning_boundary('decision_buffer_recovery'))
        finally:
            self.planner.planning_scope = scope

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

    def _reject_failed_frontier(self, detail):
        if self.goal_kind != 'frontier' or not detail.get('needs_replan'):
            return False
        failure = self.selector.consume_execution_failure(self.endpoint, self.layers.observed.sum(), detail)
        if failure is None:
            return False
        self.events.append({'event': 'frontier_execution_rejected', 'time_s': self.now, **failure})
        self.tracker = self._prepared_tracker = self.endpoint = None
        self.layers.global_path = self.layers.local_path = ()
        self.limiter.clear_guidance(self.position, self.now)
        self.phase = 'replan_frontier'
        self.next_decision = self.now
        return True

    def _preview_tracker(self, tracker):
        try:
            _, detail = tracker.action(self.position[:2], self.heading, self.planner, self.layers.current_boxes)
            return detail
        finally:
            # A preview never seeds the next command with unsent speed.
            self.limiter.accept(np.zeros(3), self.now)

    def prepare_tracking(self, recognition=None):
        """Check the first local maneuver on the stopped robot's measured map."""
        while True:
            if self.phase == 'replan_frontier':
                # Each rejected destination is excluded on this snapshot. The
                # selector eventually returns another action or an explicit wait.
                self.phase = 'decide'
                self._decide(recognition)
                continue
            tracker = self.tracker
            if self.phase != 'navigate' or tracker is None or tracker is self._prepared_tracker:
                return
            self.last_tracking = self._preview_tracker(tracker)
            if self.goal_kind == 'frontier' and self.selector.execution_failure_reason(self.last_tracking):
                # The Voronoi guide can add a corner to a short approach.
                # Try the selector's A* route on the current observed map too.
                with self.layers.global_planning() as boxes:
                    path = self.planner.path(self.position[:2], self.endpoint, boxes)
                if path and [tuple(pt[:2]) for pt in path] != [tuple(pt[:2]) for pt in self.path]:
                    self.limiter.clear_guidance(self.position, self.now)
                    alternate = self._tracker(path, short_goal=True)
                    detail = self._preview_tracker(alternate)
                    if detail.get('body_motion_safe') and not detail.get('needs_replan'):
                        self.tracker = alternate
                        self.layers.global_path, self.layers.local_path = path, ()
                        self.last_tracking, self.last_replan = detail, self.now
            if not self._reject_failed_frontier(self.last_tracking):
                self._prepared_tracker = self.tracker
                return

    def _task_route_available(self):
        """Check observed connectivity before stopping to compare actual pushes."""
        start, goal = self.planner.cell(self.position[:2]), self.planner.cell(self.goal[:2])
        if not self.planner.valid_cell(start) or not self.planner.valid_cell(goal):
            return False
        if not self.layers.global_observed[start] or not self.layers.global_observed[goal]:
            return False

        def connected(boxes):
            # Four neighbors match A* with diagonal corner cutting prohibited.
            blocked = self.planner.blocked(boxes)
            if blocked[start] or blocked[goal]:
                return False
            components, _ = label(~blocked)
            return bool(components[start] == components[goal])

        eligible = self.planner.interactable_ids
        with self.layers.global_planning() as boxes:
            fixed = [box for box in boxes if box.id not in eligible]
            if not connected(fixed):
                return False
            if len(fixed) == len(boxes) or connected(boxes):
                return True
        # Only moving an eligible box could connect these two cells. Recheck
        # a rejected push when the map, robot cell or eligibility changes.
        # A short retry also picks up sub-cell changes in the fitted geometry.
        key = (self.layers.global_revision, start, goal, frozenset(eligible))
        if self._failed_push_probe is not None:
            previous, retry_at = self._failed_push_probe
            if key == previous and self.now < retry_at:
                return False
        self._planning_boundary('frontier_reconsideration')
        feasible = self.selector.task_decision(self.layers, self.position[:2], self.goal[:2]) is not None
        self._failed_push_probe = None if feasible else (key, self.now + 5.)
        return feasible

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
        permission = self._permission(object_id, recognition)
        preparing = self.first_push and self.skill.contact_started is None
        if permission is None and not preparing:
            if self.first_push and self.execution.active:
                # A zero request may use the unexpired old lease, never extend it.
                lease = self.interaction_command
                if lease and min(lease['source_expires_at'], lease['deadline'])-self.now > .10:
                    return self._output(np.zeros(3))
                # Outside the contact zone, stop the writer and wait at this
                # pose. Retain the same trial, original travel and time budget.
                last = getattr(self, '_contact_context', None)
                if last is not None and self.skill.may_recheck_before_touch(replace(last, position=self.position,
                        heading=self.heading, orientation=self.orientation)):
                    self.phase = 'first_push_wait_scan'
                    self.skill.contact_deadline = self.execution.deadline
                    self.layers.decision.update(skill_state='wait_at_box')
                    self.execution.completed()
                    return self._output(np.zeros(3))
                self.skill.stop_trial(self.now, self.position, 'waiting_for_new_face_scan')
                return self._stop_first_push()
            return self._finish(False, 'push_aside:live_geometry_or_recognition_lost')
        target = self.current_proposals[object_id] if permission else self.layers.detected_boxes.get(object_id)
        if target is None:
            return self._finish(False, 'push_aside:box_location_unavailable')
        current_boxes = [target if box.id == object_id else box for box in self.layers.current_boxes]
        source = recognition.snapshot(self.now).get(object_id, {}) if permission else {}
        context = SkillContext(
            self.now, self.position, self.heading, self.linear, float(self.angular[2]), self.tilt,
            self.tilt < 30. and self.position[2] >= float(self.config.get('fall_height', .16)),
            target, current_boxes, self.planner, self.layers, self.limiter, self.sub_reached,
            body_angular_velocity=self.angular, orientation=self.orientation, policy_command=None,
            contact_motion_check=self._check_contact_motion if self.contact_profile is not None else None,
            box_observed_at=source.get('latest_at'),
            box_observation_id=permission.evidence_frames[-1] if permission and permission.evidence_frames else None,
            map_observed_at=self.map_observed_at, map_observation_id=self.map_observation_id)
        self._contact_context = context
        command = self.skill.step(context)
        if self.first_push and self.skill.contact_started is not None and not self.execution.active:
            if permission is None or self.trial_used:
                return self._finish(False, 'push_aside:first_push_requires_current_face')
            self.execution.start(object_id, self.now)
            self.trial_used = True
        if command.status == 'checking':
            return self._stop_first_push()
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

    def _stop_first_push(self):
        self.phase = 'first_push_review'
        self.layers.decision = dict(kind='push', skill_state='first_push_stop', reason=self.skill.reason)
        self.events.append(dict(event='first_push_stop_requested', reason=self.skill.reason, time_s=self.now))
        self.execution.completed()
        return self._output(np.zeros(3))

    def _review_first_push(self, recognition):
        key = self.skill.plan.box_id
        permission = self._permission(key, recognition)
        quiet = np.linalg.norm(self.linear) < .04 and np.max(np.abs(self.angular)) < .06
        box = self.current_proposals.get(key) if permission and quiet else None
        stamp = recognition.snapshot(self.now)[key]['latest_at'] if permission else None
        result = self.skill.check_stopped_box(self.now, self.position, box, stamp,
            permission.evidence_frames[-1] if permission else None, self.map_observed_at)
        if result is None:
            return self._output(np.zeros(3))
        self.trial_report = self.skill.report()
        self.skill_reports.append(self.trial_report)
        self.events.append(dict(event='first_push_result', time_s=self.now, report=self.trial_report))
        reason = self.skill.reason
        self.skill = None
        self._restore_after_contact()
        if not result:
            return self._finish(False, 'push_aside:'+reason)
        self.layers.refresh_global(reason='first_push_reobserve_route')
        self._reset_subnavigation()
        self.tracker, self.phase = None, 'decide'
        return self._output(np.zeros(3))

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
            # Yield only before a new exit-target search. The target selected
            # while stopped is reused and rechecked while moving; an
            # unconditional boundary here stopped again on every resume.
            local_free = self.planner.free.copy()
            cached_exit = getattr(self.planner, '_global_buffer_exit', None)
            with self.layers.global_planning() as global_boxes:
                recovery = decision_buffer_recovery(
                    self.position[:2], self.heading, self.planner, global_boxes, self.goal[:2], self.heading,
                    local_free=local_free, local_boxes=current, require_goal_connection=False,
                    before_search=lambda: self._planning_boundary('decision_buffer_recovery'))
            if recovery is None and self._tracker_route_start_blocked(current):
                # The global view saw no buffer and dropped the cached exit.
                # Restore it: the tracker view rechecks it with the body
                # sweep before reuse, and a fresh search would stop again.
                self.planner._global_buffer_exit = cached_exit
                recovery = self._tracker_view_buffer_exit(current, local_free)
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
        event = {'event': 'decision', 'time_s': self.now, 'kind': decision.kind, 'reason': decision.reason}
        if decision.kind == 'push':
            event['box_id'] = decision.push.box_id
        self.events.append(event)
        if decision.kind == 'failed':
            return self._finish(False, decision.reason)
        if decision.kind == 'wait':
            self.tracker = self.endpoint = None
            self.layers.global_path = self.layers.local_path = ()
            self.next_decision = self.now+self.layers.global_period
            return self._brake()
        if decision.kind == 'push':
            permission = self._permission(decision.push.box_id, recognition)
            if permission is None:
                if (self.first_push and not self.trial_used and decision.push.box_id in self.pushable_ids
                        and decision.push.box_id in self.layers.detected_boxes):
                    self.box_approach = decision.push
                    self._reset_subnavigation()
                    self.phase = 'box_approach'
                    self.layers.decision.update(skill_state='approach_known_box', interaction_allowed=False)
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
                if self.first_push:
                    SdkNavigationMotion(self.contact_profile).bind(self.planner)
                else:
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
                self.phase = 'replan_frontier'
                self.next_decision = self.now
            else:
                self.next_decision = self.now+self.layers.global_period
            return self._brake()
        self.tracker = self._tracker(path, short_goal=self.goal_kind != 'task')
        self.phase = 'navigate'
        self.last_replan = self.now
        self.next_reconsider = self.now+1.
        return self._brake()

    def update(self, observation, now, boxes=(), recognition=None):
        self.observe(observation, now, boxes)
        self.planner.interactable_ids = frozenset(
            box.id for box in self.layers.current_boxes if self._permission(box.id, recognition) is not None)
        self.layers.interactable_ids = self.planner.interactable_ids
        if self.first_push:
            # Selection may consider walking to a known box. The SDK lease is
            # issued separately at the face, on the original scan's expiry.
            candidates = self.pushable_ids.intersection(self.layers.detected_boxes) if (
                self.config.get('interaction_enabled') and self.full_motion_feedback and not self.trial_used) else ()
            self.planner.interactable_ids = frozenset(candidates)
        self.layers.interaction_source = ('registered_box_lidar' if self.interaction_source == 'registered_box_lidar' else 'rgb_dino_qwen_vl')
        self.layers.recognized_objects = {} if recognition is None else recognition.snapshot(self.now)
        if self.status != 'running':
            return None
        if self.world_boundary is not None and not len(self.observation['sensors'].get('lidar', {}).get('pointcloud', ())):
            return self._finish(False, 'no_lidar_returns_in_world_boundary')
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
        if self.phase == 'first_push_review':
            return self._review_first_push(recognition)
        if self.phase == 'first_push_wait_scan':
            if self.now >= self.skill.contact_deadline:
                return self._finish(False, 'push_aside:first_push_approach_time_limit')
            key = self.skill.plan.box_id
            if self._permission(key, recognition) is None:
                return self._output(np.zeros(3))
            context = replace(self._contact_context, box=self.current_proposals[key], position=self.position,
                              heading=self.heading, orientation=self.orientation)
            if not self.skill.may_recheck_before_touch(context):
                self.skill.stop_trial(self.now, self.position, 'box_rechecked_after_stop', context.box)
                self.phase = 'first_push_review'
                return self._output(np.zeros(3))
            self.execution.start(key, self.now)
            self.execution.deadline = min(self.execution.deadline, self.skill.contact_deadline)
            self.phase = 'skill'
            return self._skill_step(recognition)
        if self.phase == 'box_approach':
            if self._permission(self.box_approach.box_id, recognition):
                self.phase = 'decide'
                return self._brake()
            velocity, arrived, failed = self._subnavigation(self.box_approach.approach_xy)
            if failed or arrived:
                self.phase = 'decide'
                self.next_decision = self.now+self.layers.global_period
            return velocity
        if self.phase == 'skill' and not self.first_push and self._permission(self.skill.plan.box_id, recognition) is None:
            return self._finish(False, 'push_aside:live_geometry_or_recognition_lost')
        if self.phase == 'skill' and not self.first_push:
            self._interaction_lease(recognition)
        if self.phase == 'replan_frontier':
            self.phase = 'decide'
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
            if not close and self.now >= self.next_reconsider and not self.tracker.staging_active:
                reconsider = self._task_route_available()
                self.next_reconsider = self.now+1.
            if reconsider:
                if close:
                    self.selector.reached_frontier(self.endpoint, self.layers.observed.sum())
                self.phase = 'decide'
                self.tracker = None
                return self._brake()
        action, detail = self.tracker.action(self.position[:2], self.heading, self.planner, self.layers.current_boxes)
        self.last_tracking = detail
        if self._reject_failed_frontier(detail):
            return self._brake()
        if detail.get('needs_replan') and self.now-self.last_replan >= self.layers.global_period:
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
