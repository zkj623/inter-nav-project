"""Real-sensor orchestration of the shared simulation-release algorithms.

The planner sees measured map-frame LiDAR and tracked object geometry.  This
module never imports the Isaac runner, scene layouts or simulator body handles.
Commands are body-frame (forward, lateral, yaw rate) for the SDK adapter.
"""
import math
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from deploy.app.field_survey import FieldSurvey
from deploy.app.narrow_passage import PassageGrid, angle_error, find_passage
from deploy.robot.safety import checked_velocity
from deploy.interaction.execution import InteractionExecution
from deploy.interaction.collision import ContactCollisionChecker
from deploy.interaction.profile import ContactProfile
from deploy.interaction.motion import SdkHeadingTrim, SdkNavigationMotion
from deploy.interaction.skill_backend import create_push_skill
from deploy.interaction.first_push import RegisteredFacePlanner
from deploy.robot.observation import navigation_lidar
from deploy.robot.world_boundary import WorldBoundary
from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision, BoxState

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
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideConfig
from grutopia_extension.interactive_navigation.go2_interaction_motion import (
    MeasuredStopGate, ObservedSubgoalArrival, checked_skill_action, navigation_recovery_enabled)
from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_passage_guidance import PassageEntryGuide
from grutopia_extension.interactive_navigation.local_path_guidance import local_guidance_path


class PlanningRequired(RuntimeError):
    """Yield before unbounded global work while a motion lease is active."""


class DeploymentNavigationMap(OnlineNavigationMap):
    def __init__(self, *args, world_boundary=None, startup_clear_area=None, box_map_occupied_threshold=None,
                 box_approach_margin=0., **kwargs):
        self.world_boundary = world_boundary
        self.startup_clear_area = startup_clear_area
        self.interior = None
        self.field_survey = None
        self.measured_inside = False
        super().__init__(*args, **kwargs)
        self.box_map_occupied_threshold = box_map_occupied_threshold
        self.box_approach_margin = box_approach_margin
        if box_map_occupied_threshold is not None and (
                type(box_map_occupied_threshold) not in (float, int)
                or not self.config.occupied_threshold <= box_map_occupied_threshold <= 4.):
            raise ValueError('box_map_occupied_threshold must be between occupied_threshold and 4')
        self.events = deque(maxlen=2000)
        if world_boundary is not None:
            self.interior = world_boundary.interior_mask(self.config, self.observed.shape)
            self.boundary_wall = binary_dilation(self.interior, structure=np.ones((3, 3), bool)) & ~self.interior
            self.field_survey = FieldSurvey(self.config, self.observed.shape)
            clearance = distance_transform_edt(self.interior)*self.planner.resolution
            # Three 10 cm cells give 0.30000000000000004 m on this grid.
            self.boundary_goal_blocked = clearance <= np.nextafter(self.planner.navigation_route_radius, np.inf)
            self.evidence.static_occupied[~self.interior] = True
            self._latest_static[~self.interior] = True
            self.static_occupied[~self.interior] = True
            self.observed[self.boundary_wall] = self.global_observed[self.boundary_wall] = True
            self._bind_local()

    def _mask(self, boxes):
        mask = super()._mask(boxes)
        if self.interior is not None:
            mask &= self.interior
        return mask

    def _global_static_mask(self):
        occupied = super()._global_static_mask()
        threshold = self.box_map_occupied_threshold
        if threshold is None or not self.stationary_box_ids or self.box_approach_margin <= 0.:
            return occupied
        # Only reconsider weak hits in cells the previous global map knew as free.
        weak = (occupied & self.global_observed & ~self.global_occupied
                & ~self.evidence.static_occupied & (self.evidence.log_odds < threshold)
                & (self.vacated_time < self.last_global))
        if weak.any():
            footprint = self._mask(b for b in self.current_boxes if b.id in self.stationary_box_ids)
            if footprint.any():
                distance = distance_transform_edt(~footprint) * self.planner.resolution
                occupied[weak & (distance <= self.box_approach_margin)] = False
        return occupied

    def robot_footprint_box(self, forward):
        geometry = getattr(self.planner, 'body_geometry', None)
        if geometry is None:
            return super().robot_footprint_box(forward)
        center = (geometry.x_min+geometry.x_max)/2
        return BoxState('__robot__', tuple(self.robot_xy+center*forward),
                        (geometry.x_max-geometry.x_min, 2*geometry.half_width, .5), np.degrees(self.robot_yaw))

    def scan_masks(self, origin, points, *, max_range, min_range):
        if self.field_survey is None:
            return super().scan_masks(origin, points, max_range=max_range, min_range=min_range)
        return self.field_survey.scan_masks(origin, points, max_range, min_range)

    def frontier_grid(self):
        grid = super().frontier_grid()
        if self.interior is not None:
            # Ignore unknown slivers where the robot could never stand by the
            # wall. Known floor keeps its true clearance; the survey stays raw.
            excluded = ~self.interior | (self.boundary_goal_blocked & ~grid.observed)
            grid.observed[excluded] = True
            grid.static_occupied[excluded] = True
        return grid

    def measured_scan_masks(self, hit, free):
        if self.interior is not None:
            hit &= self.interior
            free &= self.interior
            self.measured_inside = bool(hit.any() or free.any())
        if self.scan_count == 0 and self.startup_clear_area is not None:
            area = self.startup_clear_area
            cells = area.interior_mask(self.config, self.observed.shape) & ~self.observed
            if self.interior is not None:
                cells &= self.interior
            # Prior knowledge of the fixed site. Keep neutral log odds so the
            # very first measured obstacle wins. Contact checks use actual rays.
            self.observed[cells] = True
            self.events.append(dict(event='startup_clear_area_initialized', time_s=self.now,
                                    x_limits=area.x, y_limits=area.y, cells=int(cells.sum())))
        if getattr(self, 'capture_contact_evidence', False):
            self.measured_hit, self.measured_free = hit.copy(), free.copy()

    def save(self, directory):
        super().save(directory)
        if self.field_survey is not None:
            field = self.field_survey.snapshot(self)
            if field is not None:
                np.savez_compressed(Path(directory)/'field_map.npz', **field, resolution=self.planner.resolution)

    def statistics(self):
        return {**super().statistics(),
                'world_boundary': ({'x_limits': self.world_boundary.x, 'y_limits': self.world_boundary.y}
                                   if self.world_boundary is not None else None),
                'startup_clear_area': ({'x_limits': self.startup_clear_area.x, 'y_limits': self.startup_clear_area.y}
                                       if self.startup_clear_area is not None else None),
                'movable_pose_source': 'registered_box_lidar',
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

    interaction_source = 'registered_box_lidar'

    def __init__(self, config, interaction=None, box_memory=None):
        self.config = config
        self.interaction = interaction or {}
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
            startup_clear_area=WorldBoundary.startup_area(config),
            stationary_box_ids=self.registered_ids,
            box_map_occupied_threshold=config.get('box_map_occupied_threshold'),
            box_approach_margin=(self.contact_profile.nose_offset_m + PushAsideConfig.front_gap
                                 + self.planner.navigation_route_radius if self.contact_profile else 0.),
            local_hz=float(config.get('local_map_hz', 10.)), global_hz=float(config.get('global_map_hz', 1.)),
            local_radius=float(config.get('local_map_radius', 3.)))
        self.layers.capture_contact_evidence = self.contact_profile is not None
        self.contact_collision = (ContactCollisionChecker(self.mapping_config, self.planner, self.contact_profile,
                                                          world_boundary=self.world_boundary)
                                  if self.contact_profile is not None else None)
        self.routes = ExplorationVoronoiRoutes(self.planner)
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

    def push_speed_sent(self, object_id):
        """The writer has sent forward contact motion for this box in this map."""
        self.layers.stationary_box_ids.discard(object_id)

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
        self.trial_targets = frozenset()
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
        self.narrow = None
        self.narrow_attempt = None
        self.narrow_stop = MeasuredStopGate()
        self.box_approach = None
        self.box_turn = None
        self.box_turn_started = self.box_aligned_at = None
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
                cropped = navigation_lidar(lidar, points, self.layers.config.point_height, None)
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
        from deploy.perception.registered_boxes import RegisteredBoxPerception, RegisteredBoxEvidence
        if self.contact_profile is None or not isinstance(recognition, (RegisteredBoxPerception, RegisteredBoxEvidence)):
            return None
        permission = recognition.permission(object_id, self.now)
        return permission if permission is not None and permission.source == self.interaction_source else None

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
        expiry = recognition.expires_at(self.skill.plan.box_id)
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

    def _use_contact_motion(self):
        profile = self.contact_profile
        if self.first_push:
            SdkNavigationMotion(profile).bind(self.planner)
        self.velocity_limits = {
            'max_vx': min(self.velocity_limits['max_vx'], max(profile.max_forward_mps, profile.max_reverse_mps)),
            'max_vy': min(self.velocity_limits['max_vy'], profile.max_lateral_mps),
            'max_wz': min(self.velocity_limits['max_wz'], profile.max_yaw_rps)}
        self.limiter.limits = self.velocity_limits
        self.limiter.linear_acceleration = profile.linear_acceleration_mps2
        self.limiter.yaw_acceleration = profile.yaw_acceleration_rps2

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

    def _approach_box_face(self, recognition):
        plan = self.box_approach
        if not self.full_motion_feedback or plan.box_id not in self.layers.detected_boxes:
            return self._finish(False, 'push_aside:box_approach_inputs_unavailable')

        # Seeing the box on the way updates the map. Keep the selected face and
        # let the normal tracker check the route to its original standing point.
        if self.phase == 'box_approach':
            velocity, arrived, failed = self._subnavigation(plan.approach_xy)
            if velocity is None:
                return None
            if failed:
                self.events.append(dict(event='box_approach_failed', time_s=self.now,
                                        box_id=plan.box_id, reason=failed))
                self.box_approach = None
                self.phase = 'decide'
                self.next_decision = self.now + self.layers.global_period
                return velocity
            if arrived:
                self._use_contact_motion()
                self.box_turn_started = self.now
                self.phase = 'box_turn'
                self._reset_subnavigation()
                self.layers.global_path = self.layers.local_path = ()
                self.layers.decision.update(skill_state=self.phase)
            return velocity

        heading = math.atan2(plan.direction[1], plan.direction[0])
        tolerance = PushAsideConfig.yaw_tolerance
        if self.phase == 'box_confirm' and abs(angle_error(heading, self.heading)) > tolerance:
            self.box_turn = None
            self.phase = 'box_turn'
            self.layers.decision.update(skill_state=self.phase)
        if self.phase == 'box_turn':
            if self.tilt >= self.contact_profile.max_tilt_degrees:
                return self._finish(False, 'push_aside:robot_unstable')
            if self.now - self.box_turn_started >= PushAsideConfig.max_turn_seconds:
                return self._finish(False, 'push_aside:box_face_turn_timeout')
            if self.box_turn is None:
                self.box_turn = SdkHeadingTrim(heading, tolerance, profile=self.contact_profile)
            command, aligned, detail = self.box_turn.update(
                self.position, self.heading, self.planner, self.layers.current_boxes,
                self.limiter, self.now, self.linear, self.angular, self.tilt)
            self.layers.decision.update(heading_error_rad=detail['heading_error_rad'],
                                        turn_clear=detail['body_motion_safe'])
            if self.box_turn.failed:
                return self._finish(False, 'push_aside:'+self.box_turn.failed)
            if aligned:
                self.box_aligned_at = self.now
                self.phase = 'box_confirm'
                self.layers.decision.update(skill_state=self.phase)
                self.events.append(dict(event='box_face_aligned', time_s=self.now, box_id=plan.box_id,
                                        heading_error_rad=detail['heading_error_rad']))
            return self._output(command)

        if self.now - self.box_aligned_at >= PushAsideConfig.max_alignment_seconds:
            return self._finish(False, 'push_aside:no_fresh_box_face_after_turn')
        permission = self._permission(plan.box_id, recognition)
        if permission is not None:
            stamp = recognition.snapshot(self.now)[plan.box_id]['latest_at']
            if stamp is not None and stamp > self.box_aligned_at:
                self.events.append(dict(event='box_face_confirmed', time_s=self.now, box_id=plan.box_id,
                                        scan_at=stamp, aligned_at=self.box_aligned_at))
                return self._start_push_skill(plan, permission, recognition)
        return self._brake()

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
                                'skill_state': command.state,
                                'interaction_allowed': not self.first_push or self.execution.active,
                                'execution_state': command.state}
        if self.first_push:
            self.layers.decision['precontact_wait'] = self.skill.precontact_wait
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

    def _advance_first_push(self, recognition):
        if self.phase == 'first_push_review':
            return self._review_first_push(recognition)
        if self.phase != 'first_push_wait_scan':
            return self._approach_box_face(recognition)
        # Waiting for a new face after the writer has stopped before contact.
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

    def _try_narrow_passage(self):
        start, goal = self.position[:2], self.goal[:2]
        key = (self.layers.global_revision, self.layers.local_revision, self.planner.cell(start),
               round(self.heading, 1))
        if (self.narrow_attempt == key or not self.planner.valid_cell(self.planner.cell(start))
                or not self.layers.global_observed[self.planner.cell(goal)]):
            return False
        with self.layers.global_planning() as boxes:
            blocked = self.planner.blocked(boxes)
            regions, _ = label(~blocked)
            here, there = regions[self.planner.cell(start)], regions[self.planner.cell(goal)]
            if here and here == there:
                self.narrow_attempt = key
                return False
            if not here and not self._between_obstacles(start):
                # A single nearby wall belongs to ordinary buffer recovery. Its
                # cached exit must be allowed to run without another planning hold.
                self.narrow_attempt = key
                return False
            self._planning_boundary('narrow_passage_search')
            free = self.planner.free & self.layers.observed & ~self.layers.local_occupied & ~self.layers.global_tracked
            plan = find_passage(self.planner, PassageGrid(self.planner, free), blocked, start, self.heading, goal)
        self.narrow_attempt = key
        if plan is None:
            return False
        self.narrow = plan
        self.narrow_stop.reset()
        self.phase = 'cross_passage'
        self.goal_kind = 'narrow_entry'
        self.narrow_stage = 'arrive'
        self.tracker = None
        if plan.approach and np.linalg.norm(start-plan.entry) > .13:
            self.phase = 'navigate'
            self.endpoint = tuple(plan.entry)
            self.tracker = self._tracker(plan.approach, short_goal=True)
        self.layers.global_path = plan.approach if self.tracker else (tuple(plan.entry), tuple(plan.exit))
        self.layers.local_path = ()
        self.layers.decision = dict(kind='navigate', reason='oriented_narrow_passage',
            entry=plan.entry.tolist(), exit=plan.exit.tolist(), heading_rad=plan.heading)
        self.events.append(dict(event='narrow_passage_selected', time_s=self.now, **self.layers.decision))
        return True

    def _between_obstacles(self, point):
        occupied = self.layers.global_occupied | self.layers.local_occupied
        planner = self.planner
        radius = planner.navigation_route_radius + planner.resolution
        distances = np.arange(planner.resolution / 2, radius, planner.resolution / 2)
        for theta in np.arange(8) * math.pi / 8:
            offset = np.array([math.cos(theta), math.sin(theta)])
            sides = [[planner.cell(point + sign*d*offset) for d in distances] for sign in (-1, 1)]
            if all(any(planner.valid_cell(cell) and occupied[cell] for cell in side) for side in sides):
                return True
        return False

    def _leave_narrow(self, reason):
        self.events.append(dict(event=reason, time_s=self.now))
        self.narrow = self.tracker = self.endpoint = None
        self.goal_kind = None
        self.phase = 'decide'
        self.layers.global_path = self.layers.local_path = ()
        self.layers.decision = dict(kind='replan', reason=reason)
        return self._output(np.zeros(3))

    def _cross_narrow(self):
        plan = self.narrow
        grid = PassageGrid(self.planner, self.layers.observed & ~self.layers.local_occupied)
        pos = self.position[:2]
        error = angle_error(plan.heading, self.heading)
        target = plan.entry if self.narrow_stage != 'cross' else plan.exit
        delta = target-pos
        distance = np.linalg.norm(delta)
        request = np.zeros(3)
        aligned = abs(error) <= .035
        axes = np.array([[math.cos(self.heading), math.sin(self.heading)],
                         [-math.sin(self.heading), math.cos(self.heading)]])
        if not grid.clear(plan.entry if self.narrow_stage != 'cross' else pos, plan.exit, plan.heading):
            return self._leave_narrow('narrow_passage_changed')
        # At arrival, keep a zero request until measured approach motion settles.
        if self.narrow_stage == 'align':
            if not grid.can_turn(pos, self.heading, plan.heading):
                return self._leave_narrow('narrow_turn_blocked')
            if not aligned:
                request[2] = np.clip(1.8*error, -.4, .4)
        elif self.narrow_stage in ('center', 'cross'):
            if not grid.clear(pos, target, self.heading) or not grid.can_turn(pos, self.heading, plan.heading):
                return self._leave_narrow('narrow_sweep_blocked')
            if distance > .025:
                request[:2] = axes @ delta * min(1.2, .18/distance)
            request[2] = np.clip(1.5*error, -.22, .22)
            if aligned and distance <= .025:
                request[:] = 0.
        command, safe = checked_contact_turn_command(request, self.position, self.heading, self.planner,
            self.layers.current_boxes, self.limiter, self.now)
        if not safe:
            return self._leave_narrow('narrow_command_blocked')
        if np.any(request) or np.linalg.norm(command) > .02:
            self.narrow_stop.reset()
        elif self.narrow_stop.update(self.position, self.linear, self.angular, self.tilt, self.now):
            self.narrow_stop.reset()
            if self.narrow_stage == 'cross':
                self.narrow_attempt = None
                return self._leave_narrow('narrow_passage_crossed')
            self.narrow_stage = {'arrive': 'align', 'align': 'center', 'center': 'cross'}[self.narrow_stage]
        self.layers.decision['stage'] = self.narrow_stage
        self.last_tracking = dict(controller='narrow_passage', stage=self.narrow_stage,
                                  heading_error_rad=error, distance_m=float(distance))
        return self._output(command)

    def _decide(self, recognition):
        if self.now < self.next_decision:
            return self._brake()
        self.layers.refresh_global(reason='decision')
        # A straight throat can put the robot inside both circular wall buffers.
        # Check for an oriented way through before trying to back out of it.
        if (not self.trial_targets and self.passage is None
                and not self.recovery_pending_settle and self._try_narrow_passage()):
            return self._brake()
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
        self._planning_boundary('global_decision')
        if self.trial_targets:
            self.layers.refresh_global(reason='registered_first_contact')
            free = self.planner.free
            with self.layers.global_planning():
                plan = self.planner.first_contact_plan(self.position[:2], current, self.trial_targets,
                                                       self.planner.box_free & free, self.contact_profile)
            if plan is not None:
                decision = BoxDecision('push', 'registered_first_contact', push=plan, explanation={
                    'box_id': plan.box_id, 'push_distance_m': plan.push_distance,
                    'initial_xy': list(plan.initial_xy), 'destination_xy': list(plan.destination_xy),
                    'push_direction': list(plan.direction), 'approach_xy': list(plan.approach_xy)})
                return self._apply_decision(decision, None, recognition)
            if self.passage is None and self._try_narrow_passage():
                return self._brake()
        decision, endpoint = self.selector.decide(self.layers, self.position[:2], self.goal[:2])
        return self._apply_decision(decision, endpoint, recognition)

    def _start_push_skill(self, plan, permission, recognition):
        speed = min(float(self.config.get('push_max_speed', .30)), float(self.config.get('max_forward_speed', .5)))
        skill_config = PushAsideConfig(
            push_start_speed=min(float(self.config.get('push_start_speed', .15)), speed), push_max_speed=speed,
            lateral_speed=min(.35, float(self.config.get('max_lateral_speed', .18))),
            forward_speed=min(.25, float(self.config.get('max_forward_speed', .5))), reverse_speed=.35,
            max_skill_seconds=float(self.interaction.get('max_skill_seconds', 180.)))
        self.skill = create_push_skill(permission, plan, self.task_start, self.goal[:2],
                                       self.contact_profile, config=skill_config)
        self._use_contact_motion()
        if not self.first_push:
            self.execution.start(plan.box_id, self.now)
            self._interaction_lease(recognition)
        self.box_approach = self.box_turn = None
        self.tracker = None
        self.phase = 'skill'
        self.layers.decision.update(skill_state=self.skill.state)
        self._reset_subnavigation()
        return self._output(np.zeros(3))

    def _apply_decision(self, decision, endpoint, recognition):
        self.layers.decision = {**(decision.explanation or {}), 'kind': decision.kind,
                                'reason': decision.reason, 'decision_time_s': self.now}
        event = {'event': 'decision', 'time_s': self.now, 'kind': decision.kind, 'reason': decision.reason}
        if decision.kind == 'push':
            event.update(box_id=decision.push.box_id, approach_xy=list(decision.push.approach_xy),
                         push_direction=list(decision.push.direction))
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
                    self.tracker = None
                    self.layers.global_path = decision.push.approach_path
                    self.layers.local_path = ()
                    self._reset_subnavigation()
                    self.phase = 'box_approach'
                    self.layers.decision.update(skill_state='approach_known_box', interaction_allowed=False,
                                                approach_xy=list(decision.push.approach_xy),
                                                push_direction=list(decision.push.direction))
                self.next_decision = self.now+self.layers.global_period
                return self._brake()
            return self._start_push_skill(decision.push, permission, recognition)
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
            previous_targets = self.trial_targets
            # Walk to a known registered box even between face observations;
            # starting contact still requires its current LiDAR permission.
            self.trial_targets = (self.pushable_ids.intersection(self.layers.detected_boxes)
                                  if self.config.get('interaction_enabled') and self.full_motion_feedback
                                  and not self.trial_used else frozenset())
            self.planner.interactable_ids = frozenset(self.trial_targets)
            if (self.phase == 'navigate' and self.trial_targets - previous_targets
                    and self.goal_kind != 'narrow_entry'):
                self.tracker = None
                self.phase = 'decide'
                self.next_decision = self.now
        self.layers.recognized_objects = {} if recognition is None else recognition.snapshot(self.now)
        if self.status != 'running':
            return None
        if self.world_boundary is not None:
            lidar = self.observation['sensors'].get('lidar', {})
            points = lidar.get('pointcloud', ())
            interior_ray = self.layers.measured_inside and self.layers.last_scan == lidar.get('physics_step')
            if not len(points) or not (interior_ray or self.world_boundary.point_mask(points).any()):
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
        if self.box_approach is not None or self.phase in ('first_push_review', 'first_push_wait_scan'):
            return self._advance_first_push(recognition)
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
        if self.phase == 'navigate' and self.goal_kind == 'narrow_entry' and np.linalg.norm(
                self.position[:2]-self.narrow.entry) < .13:
            self.phase = 'cross_passage'
            self.tracker = None
            self.layers.global_path = (tuple(self.narrow.entry), tuple(self.narrow.exit))
            self.layers.local_path = ()
            return self._output(np.zeros(3))
        if self.phase == 'cross_passage':
            return self._cross_narrow()
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
            if self.goal_kind == 'narrow_entry':
                return self._leave_narrow('narrow_approach_blocked')
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
