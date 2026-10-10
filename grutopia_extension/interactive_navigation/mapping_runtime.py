"""Runtime glue from GRUtopia sensor observations to the fused map."""

import json
from dataclasses import asdict, replace
from enum import Enum
from math import atan2, cos, pi, sin
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from grutopia_extension.interactive_navigation.mapping import FusedMap, MappingConfig, PlanningError, SemanticDetection
from grutopia_extension.interactive_navigation.state_machine import (
    ControllerCommand,
    InteractionState,
    StateMachineDecision,
)

_NON_OBJECT_SEMANTIC_LABELS = {
    'ceiling',
    'floor',
    'g1',
    'go2',
    'ground',
    'other',
    'robot',
    'unitree_go2',
    'wall',
}


_SEMANTIC_DETECTION_MODE_ALIASES = {
    'isaac': 'ISAAC',
    'gt': 'ISAAC',
    'ground_truth': 'ISAAC',
    'open_vocab': 'OPEN_VOCABULARY',
    'open_vocabulary': 'OPEN_VOCABULARY',
    'openvocab': 'OPEN_VOCABULARY',
    'hybrid': 'HYBRID',
    'fused': 'HYBRID',
}


class SemanticDetectionMode(str, Enum):
    """Which perception source supplies object detections to the fused map.

    ``isaac`` uses only the simulator's semantic bounding boxes, i.e. the
    ground-truth labels of the scene. ``open_vocab`` uses only the
    GroundingDINO/MobileSAM detections, which is what a real robot would see.
    ``hybrid`` fuses both: a scene-graph node keeps every source that agreed
    on it, and geometry comes from the denser observation.
    """

    ISAAC = 'isaac'
    OPEN_VOCABULARY = 'open_vocab'
    HYBRID = 'hybrid'

    @classmethod
    def parse(cls, value) -> 'SemanticDetectionMode':
        """Normalize a CLI string or enum member into a member."""

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold().replace('-', '_').replace(' ', '_')
        try:
            return cls[_SEMANTIC_DETECTION_MODE_ALIASES[normalized]]
        except KeyError:
            supported = ', '.join(mode.value for mode in cls)
            raise ValueError(
                f'unknown semantic detection mode {value!r}; expected one of {supported}'
            ) from None

    @property
    def uses_isaac(self) -> bool:
        return self is not SemanticDetectionMode.OPEN_VOCABULARY

    @property
    def uses_open_vocabulary(self) -> bool:
        return self is not SemanticDetectionMode.ISAAC


class MapNavigationRuntime:
    """Fuse live sensors and replace path commands with A* map plans."""

    def __init__(
        self,
        mapping_config: MappingConfig = MappingConfig(),
        lidar_interval: int = 8,
        rgb_interval: int = 24,
        use_planner: bool = True,
        use_scene_graph: bool = True,
        safe_base_height: float = 0.95,
        replan_interval_steps: Optional[int] = None,
        allow_goal_door_traversal: bool = True,
        use_semantic_occupancy: bool = True,
        use_rgb_occupancy: bool = True,
        rgb_clears_free_space: bool = True,
        replan_only_if_blocked: bool = False,
        replan_lookahead_distance: Optional[float] = None,
        semantic_target: str = '',
        open_vocabulary_perception=None,
        semantic_detection_mode=SemanticDetectionMode.HYBRID,
        open_vocabulary_startup_error: Optional[str] = None,
        open_vocabulary_failure_limit: int = 3,
        use_semantic_voronoi: bool = False,
        semantic_voronoi_config=None,
        topology_update_interval: int = 40,
        prefer_voronoi_paths: bool = False,
        safe_path_tracking: bool = False,
        observed_only: bool = False,
        buffer_recovery_timeout: int = 600,
    ):
        if lidar_interval <= 0 or rgb_interval <= 0:
            raise ValueError('sensor update intervals must be positive')
        if replan_lookahead_distance is not None and replan_lookahead_distance <= 0:
            raise ValueError('replan_lookahead_distance must be positive when provided')
        if topology_update_interval <= 0:
            raise ValueError('topology_update_interval must be positive')
        if open_vocabulary_failure_limit <= 0:
            raise ValueError('open_vocabulary_failure_limit must be positive')
        if buffer_recovery_timeout <= 0:
            raise ValueError('buffer_recovery_timeout must be positive')
        safe_path_tracking = safe_path_tracking or observed_only
        self.observed_only = safe_path_tracking
        self.safe_path_tracking = safe_path_tracking
        self.buffer_recovery_timeout = buffer_recovery_timeout
        self.buffer_recovery_active = False
        self.buffer_recovery_failed = False
        self._buffer_started = None
        self._buffer_exit = None
        self.buffer_history = []
        self.tracking_chord_fallbacks = 0
        self.tracking_segment_stops = 0
        self.map = FusedMap(mapping_config)
        if safe_path_tracking:
            from .mapping import AStarMapPlanner
            self.map.planner = AStarMapPlanner(self.map.occupancy, observed_only=True)
        self.lidar_interval = lidar_interval
        self.rgb_interval = rgb_interval
        self.use_planner = use_planner
        self.use_scene_graph = use_scene_graph
        self.safe_base_height = safe_base_height
        self.replan_interval_steps = replan_interval_steps
        self.allow_goal_door_traversal = allow_goal_door_traversal
        self.use_semantic_occupancy = use_semantic_occupancy
        self.use_rgb_occupancy = use_rgb_occupancy
        self.rgb_clears_free_space = rgb_clears_free_space
        self.replan_only_if_blocked = replan_only_if_blocked
        self.replan_lookahead_distance = replan_lookahead_distance
        self.semantic_target = str(semantic_target).strip()
        self.open_vocabulary_perception = open_vocabulary_perception
        self.semantic_detection_mode = SemanticDetectionMode.parse(semantic_detection_mode)
        self.open_vocabulary_startup_error = open_vocabulary_startup_error
        if (
            self.semantic_detection_mode is SemanticDetectionMode.OPEN_VOCABULARY
            and open_vocabulary_perception is None
        ):
            # Hybrid degrades to Isaac-only when no perception is configured,
            # but an explicit open-vocab request must not silently fall back to
            # ground truth: that would corrupt a detection-quality comparison
            # with labels only the simulator can provide.
            raise ValueError(
                'semantic_detection_mode=open_vocab requires an open_vocabulary_perception'
            )
        self.open_vocabulary_frames = 0
        self.open_vocabulary_failures = 0
        self.open_vocabulary_attempts = 0
        self.open_vocabulary_rate_limited_frames = 0
        self.open_vocabulary_empty_frames = 0
        self.open_vocabulary_detections = 0
        self.open_vocabulary_partial_failures = 0
        self.open_vocabulary_consecutive_failures = 0
        self.open_vocabulary_failure_limit = int(open_vocabulary_failure_limit)
        self.open_vocabulary_last_error = None
        self.open_vocabulary_last_partial_errors = []
        self.open_vocabulary_last_labels = []
        self.open_vocabulary_last_detection_step = None
        self.open_vocabulary_last_success_step = None
        self.topology_update_interval = int(topology_update_interval)
        self.semantic_voronoi = None
        self.voronoi_planner = None
        self.voronoi_plan_count = 0
        self.voronoi_plan_fallbacks = 0
        self._last_plan_used_voronoi = False
        if use_semantic_voronoi:
            from grutopia_extension.interactive_navigation.semantic_voronoi import (
                SemanticVoronoiGraph,
                VoronoiPathPlanner,
            )

            self.semantic_voronoi = SemanticVoronoiGraph(
                self.map.occupancy,
                semantic_voronoi_config,
            )
            if prefer_voronoi_paths:
                # Waypoints are sampled directly from the skeleton path at a
                # short spacing, so the robot walks along the max-clearance
                # medial axis instead of driving straight chords towards the
                # goal. Line-of-sight cuts only check the inflated mask,
                # which under-protects thin obstacles such as standing
                # people (observed 0.26 m near-miss with a 2.5 m cap).
                self.voronoi_planner = VoronoiPathPlanner(
                    self.semantic_voronoi,
                    waypoint_spacing=0.30,
                    observed_only=safe_path_tracking,
                )
        self._last_lidar_physics_step = -1
        self.lidar_self_hits_removed = 0
        self._planned_state: Optional[InteractionState] = None
        self._planned_path = None
        self._planned_goal = None
        self._waypoint_index = 0
        self._planned_paths = {}
        self._plan_history = []
        self.planning_failures = 0
        self.replan_count = 0
        self.replan_reasons = {}
        self.last_replan = None
        self._last_path_blockage = None
        self._progress_position = None
        self._progress_step = None
        self._last_plan_step = None
        self._last_safe_position = None
        self._static_obstacle_boxes = []
        self._static_obstacle_violations = 0
        self._violated_static_labels = set()
        self.navigation_headings = {
            InteractionState.NAVIGATE_TO_OBSTACLE: 0.0,
            InteractionState.NAVIGATE_TO_OBJECT: 0.0,
            InteractionState.NAVIGATE_TO_DOOR: 0.0,
        }
        self.final_approach_distance = 0.45
        self.intermediate_waypoint_tolerance = 0.28
        self.calibrated_waypoint_tolerance = 0.08
        self.object_heading_alignment_distance = 0.03
        # Pure-pursuit tracking of planner paths: the carrot slides along the
        # dense skeleton waypoints so velocity commands stay continuous.
        self.velocity_lookahead_distance = 0.55
        self.final_goal_slow_radius = 0.50
        # Blocked-path checks run every step (bounded by this cooldown) so a
        # freshly mapped obstacle triggers a replan within a few steps.
        self.blocked_replan_cooldown_steps = 10
        self._last_blocked_replan_step = None
        # Standing person footprint (0.6 x 0.37 m body circumscribed) stamped
        # into occupancy from camera scene-graph nodes.
        self.person_obstacle_radius = 0.35
        self.person_obstacle_min_observations = 10

    def seed_static_obstacles(self, obstacles: Iterable[dict], mark_occupancy: bool = True):
        """Install immutable metadata obstacles used by the point-navigation stack.

        With ``mark_occupancy=False`` the boxes are only used for collision
        statistics, keeping the online map free of prior knowledge (as needed
        by unknown-environment exploration runs).
        """

        known_labels = {label for label, _, _ in self._static_obstacle_boxes}
        for obstacle in obstacles:
            label = str(obstacle['label'])
            if label in known_labels:
                continue
            minimum = np.asarray(obstacle['minimum_xy'], dtype=np.float64)
            maximum = np.asarray(obstacle['maximum_xy'], dtype=np.float64)
            if mark_occupancy:
                self.map.occupancy.mark_box_occupied(minimum, maximum)
            self._static_obstacle_boxes.append((label, minimum, maximum))
            known_labels.add(label)

    def update(self, step: int, robot_observation: dict):
        if step % self.topology_update_interval == 0:
            # Low-frequency maintenance before new evidence arrives: cells that
            # stayed weak and isolated through the whole interval are noise,
            # while anything hit repeatedly has passed the confirmation level.
            self.map.occupancy.prune_isolated_occupancy()
        sensors = robot_observation.get('sensors', {})
        lidar = sensors.get('lidar', {})
        if step % self.lidar_interval == 0:
            physics_step = int(lidar.get('physics_step', -1))
            points = _points(lidar.get('pointcloud'))
            if physics_step != self._last_lidar_physics_step and len(points) > 0:
                self.lidar_self_hits_removed += int(lidar.get('self_hits_removed', 0))
                origin = _vector3(lidar.get('position'))
                self.map.update_lidar(
                    origin=origin,
                    points=points,
                    step=step,
                    max_range=float(lidar.get('max_range', 8.0)),
                )
                self._last_lidar_physics_step = physics_step

        camera = sensors.get('camera', {})
        if step % self.rgb_interval == 0:
            points, colors, point_image = _camera_cloud(camera)
            if len(points) > 0:
                detections = ()
                if self.use_scene_graph:
                    detections = self._camera_semantic_detections(
                        camera,
                        point_image,
                        step,
                    )
                camera_position = camera.get('position')
                origin = None if camera_position is None else _vector3(camera_position)
                self.map.update_rgb(
                    points,
                    colors,
                    detections,
                    step=step,
                    origin=origin,
                    update_occupancy=self.use_rgb_occupancy,
                    update_semantic_occupancy=self.use_semantic_occupancy,
                    clear_free_space=self.rgb_clears_free_space,
                )

        position = robot_observation.get('position')
        if position is not None:
            position = np.asarray(position, dtype=np.float64)
            self._update_static_obstacle_violations(position)
            self.map.occupancy.mark_free(position[:2], self.map.config.robot_radius)
            navigation_grid = self.navigation_occupancy
            cell = navigation_grid.world_to_cell(position[:2])
            if (
                position[2] >= self.safe_base_height
                and self.map.config.safe_recovery_y_limits[0]
                <= position[1]
                <= self.map.config.safe_recovery_y_limits[1]
                and cell is not None
                and navigation_grid.observed[cell]
                and not navigation_grid.inflated_mask()[cell]
            ):
                self._last_safe_position = tuple(float(value) for value in position[:3])
        if step % self.topology_update_interval == 0:
            self._mark_person_obstacles()
        if self.semantic_voronoi is not None and step % self.topology_update_interval == 0:
            self.semantic_voronoi.update()
            self.semantic_voronoi.attach_semantics(
                self.map.scene_graph.object_nodes()
            )

    def _mark_person_obstacles(self):
        """Stamp well-observed person nodes into the occupancy grid.

        People are thin and often enter the lidar occupancy layer only at
        very close range, while the camera scene graph localizes them within
        centimetres long before that. Marking them explicitly lets the
        planner keep clearance from the start instead of replanning at the
        last moment. The observation threshold skips depth-median ghosts.
        """

        if not self.use_scene_graph:
            return
        for node in self.map.scene_graph.object_nodes():
            if node.label == 'person' and node.observations >= self.person_obstacle_min_observations:
                self.map.occupancy.mark_occupied(
                    node.position[:2],
                    self.person_obstacle_radius,
                )

    def _camera_semantic_detections(self, camera: dict, point_image, step: int):
        mode = self.semantic_detection_mode
        # Ground truth is dropped entirely in open-vocabulary mode so a
        # detection-quality run cannot be silently rescued by simulator labels.
        isaac = (
            tuple(_semantic_detections(camera, point_image, step=step))
            if mode.uses_isaac
            else ()
        )
        if (
            not mode.uses_open_vocabulary
            or self.open_vocabulary_perception is None
            or not self.semantic_target
        ):
            return isaac
        self.open_vocabulary_attempts += 1
        try:
            detections = self.open_vocabulary_perception.perceive(
                rgba=camera.get('rgba'),
                depth=camera.get('depth'),
                point_image=point_image,
                target=self.semantic_target,
                step=step,
            )
            semantic_detections = []
            for detection in detections:
                if isinstance(detection, SemanticDetection):
                    semantic_detections.append(
                        detection
                        if detection.sources
                        else replace(detection, sources=('open_vocabulary',))
                    )
                elif hasattr(detection, 'to_semantic_detection'):
                    semantic_detections.append(detection.to_semantic_detection())
            query_status = getattr(
                self.open_vocabulary_perception,
                'last_query_status',
                'ok',
            )
            if query_status == 'rate_limited':
                self.open_vocabulary_rate_limited_frames += 1
                return isaac
            self.open_vocabulary_frames += 1
            self.open_vocabulary_consecutive_failures = 0
            self.open_vocabulary_last_success_step = int(step)
            item_errors = getattr(
                self.open_vocabulary_perception,
                'last_item_errors',
                (),
            )
            self.open_vocabulary_last_partial_errors = list(item_errors)
            self.open_vocabulary_partial_failures += len(item_errors)
            self.open_vocabulary_detections += len(semantic_detections)
            if semantic_detections:
                self.open_vocabulary_last_labels = sorted(
                    {detection.label for detection in semantic_detections}
                )
                self.open_vocabulary_last_detection_step = int(step)
            else:
                self.open_vocabulary_empty_frames += 1
            # In hybrid mode open-vocabulary detections enrich the camera's
            # ground-truth semantics instead of replacing them: dropping the
            # fallback hid target classes (e.g. refrigerator) from the scene
            # graph whenever GroundingDINO returned any context detection, so
            # lexical target matching could never fire. In open-vocabulary
            # mode `isaac` is empty and the same pass just deduplicates the
            # model's own boxes.
            return _deduplicate_semantic_detections(
                tuple(semantic_detections) + isaac
            )
        except Exception as error:
            self.open_vocabulary_failures += 1
            self.open_vocabulary_consecutive_failures += 1
            item_errors = list(
                getattr(self.open_vocabulary_perception, 'last_item_errors', ())
            )
            self.open_vocabulary_last_partial_errors = item_errors
            self.open_vocabulary_partial_failures += len(item_errors)
            self.open_vocabulary_last_error = {
                'step': int(step),
                'type': type(error).__name__,
                'message': str(error),
            }
            if (
                mode is SemanticDetectionMode.OPEN_VOCABULARY
                and self.open_vocabulary_consecutive_failures
                >= self.open_vocabulary_failure_limit
            ):
                raise RuntimeError(
                    'open-vocabulary perception failed '
                    f'{self.open_vocabulary_consecutive_failures} consecutive frames; '
                    f'last error: {type(error).__name__}: {error}'
                ) from error
            return isaac

    def action_for(
        self,
        decision: StateMachineDecision,
        robot_observation: dict,
        step: Optional[int] = None,
    ) -> dict:
        action = decision.as_action()
        command = decision.command
        if not self.use_planner:
            return action
        if (
            command is not None
            and decision.state == InteractionState.PUSH_OBSTACLE
            and command.name == 'move_by_speed'
        ):
            return self._aligned_push_action(command, robot_observation)
        if (
            command is not None
            and decision.state == InteractionState.RECOVER
            and command.name == 'recover'
            and self._last_safe_position is not None
            and self._planned_state is not None
        ):
            recover_height = float(command.data[0][2])
            recovery_position = self._last_safe_position
            if (
                self._planned_state == InteractionState.NAVIGATE_TO_GOAL
                and 'door' in self.map.traversable_semantic_labels
                and self._planned_path
            ):
                portal = self._planned_path[0]
                if self._last_safe_position[0] <= portal[0] + 0.20:
                    recovery_position = portal
            recover_target = (
                recovery_position[0],
                recovery_position[1],
                recover_height,
            )
            self._planned_state = None
            self._planned_path = None
            self._planned_goal = None
            self._waypoint_index = 0
            return ControllerCommand(command.name, (recover_target, command.data[1])).as_action()
        if decision.state == InteractionState.PUSH_DOOR and 'move_by_speed' in action:
            speed_command = action['move_by_speed']
            action['move_by_speed'] = self._aligned_speed_data(
                forward_speed=float(speed_command[0]),
                orientation=robot_observation['orientation'],
            )
            return action
        if command is None or command.name != 'move_along_path':
            self._planned_state = None
            self._planned_path = None
            self._planned_goal = None
            self._waypoint_index = 0
            self._progress_position = None
            self._progress_step = None
            self._last_plan_step = None
            return action

        if self.map.lidar_frames == 0:
            return ControllerCommand('move_by_speed', (0.0, 0.0, 0.0)).as_action()

        start = _vector3(robot_observation['position'])
        configured_path = command.data[0]
        if (
            decision.state
            in (
                InteractionState.NAVIGATE_TO_OBJECT,
                InteractionState.NAVIGATE_TO_CARRY_GOAL,
            )
            and len(configured_path) > 1
        ):
            return self._calibrated_path_action(
                decision.state,
                command.name,
                configured_path,
                start,
                step,
            )
        goal = tuple(float(value) for value in configured_path[-1])
        plan_reason = None
        if (
            decision.state == InteractionState.NAVIGATE_TO_OBJECT
            and np.linalg.norm(np.asarray(start[:2]) - np.asarray(goal[:2]))
            <= self.object_heading_alignment_distance
            and abs(_yaw(robot_observation['orientation'])) > 0.12
        ):
            return ControllerCommand(
                'move_by_speed',
                tuple(self._aligned_speed_data(0.0, robot_observation['orientation'])),
            ).as_action()
        if self._planned_state != decision.state:
            plan_reason = 'initial' if self._planned_state is None else 'state_changed'
            self._planned_path = None
            self._planned_goal = None
            self._progress_position = np.asarray(start[:2])
            self._progress_step = step
            self._last_plan_step = None
            self._last_blocked_replan_step = None
        elif (
            self._planned_path is not None
            and self._planned_goal is not None
            and np.linalg.norm(
                np.asarray(goal[:2], dtype=np.float64)
                - np.asarray(self._planned_goal[:2], dtype=np.float64)
            )
            > self.map.config.grid_resolution * 0.5
        ):
            self._record_replan('goal_changed', step, start)
            self._progress_position = np.asarray(start[:2])
            self._progress_step = step
            self._last_blocked_replan_step = None
            plan_reason = 'goal_changed'
        elif (
            step is not None
            and self.replan_only_if_blocked
            and self._planned_path is not None
            and (
                self._last_blocked_replan_step is None
                or step - self._last_blocked_replan_step >= self.blocked_replan_cooldown_steps
            )
            and self._remaining_path_blocked(start)
        ):
            # Obstacles confirmed into the map at close range (e.g. a person
            # the lidar only resolves late) must interrupt the plan at once:
            # waiting for the periodic replan timer gave a blind window of
            # ~0.5 m at walking speed and led to a near-collision pass.
            self._last_blocked_replan_step = step
            self._record_replan('blocked', step, start)
            plan_reason = 'blocked'
        elif (
            step is not None
            and self.replan_interval_steps is not None
            and self._last_plan_step is not None
            and step - self._last_plan_step >= self.replan_interval_steps
        ):
            retry_voronoi = (self.voronoi_planner is not None and not self._last_plan_used_voronoi
                             and self.navigation_occupancy is self.map.occupancy)
            if self.replan_only_if_blocked and not retry_voronoi and not self._remaining_path_blocked(start):
                self._last_plan_step = step
            else:
                plan_reason = 'periodic_blocked' if self.replan_only_if_blocked else 'periodic'
                self._record_replan(plan_reason, step, start)
        elif step is not None and self._progress_step is not None and step - self._progress_step >= 240:
            current_position = np.asarray(start[:2])
            if np.linalg.norm(current_position - self._progress_position) < 0.12:
                self._record_replan('stalled', step, start)
                plan_reason = 'stalled'
            self._progress_position = current_position
            self._progress_step = step

        if self._planned_state != decision.state or self._planned_path is None:
            if (
                decision.state == InteractionState.NAVIGATE_TO_GOAL
                and self.use_scene_graph
                and self.allow_goal_door_traversal
            ):
                self.map.set_semantic_labels_traversable(
                    {
                        'door',
                        'door_frame_hinge',
                        'door_frame_latch',
                    }
                )
            try:
                self._planned_path = self._plan_with_final_heading(decision.state, start, goal)
            except PlanningError as error:
                self.last_planning_error = str(error)
                self.planning_failures += 1
                return ControllerCommand('move_by_speed', (0.0, 0.0, 0.0)).as_action()
            self._planned_state = decision.state
            self._planned_goal = goal
            self._waypoint_index = 0
            self._last_plan_step = step
            self._planned_paths[decision.state.value] = self._planned_path
            self._plan_history.append(
                {
                    'step': step,
                    'state': decision.state.value,
                    'reason': plan_reason or 'initial',
                    'goal': list(goal),
                    'path': [list(point) for point in self._planned_path],
                }
            )

        while self._waypoint_index < len(self._planned_path) - 1:
            waypoint = np.asarray(self._planned_path[self._waypoint_index][:2], dtype=np.float64)
            if np.linalg.norm(waypoint - np.asarray(start[:2])) >= self.intermediate_waypoint_tolerance:
                break
            if self.safe_path_tracking and not self._tracking_segment_clear(
                start[:2], self._planned_path[self._waypoint_index + 1][:2]
            ):
                break
            self._waypoint_index += 1
        action[command.name] = [self._planned_path[self._waypoint_index :]]
        return action

    def _calibrated_path_action(self, state, command_name, configured_path, start, step):
        path = tuple(
            tuple(float(value) for value in waypoint)
            for waypoint in configured_path
        )
        if self._planned_state != state or self._planned_path != path:
            self._planned_state = state
            self._planned_path = path
            self._planned_goal = path[-1]
            self._waypoint_index = 0
            self._last_plan_step = step
            self._planned_paths[state.value] = path
            self._plan_history.append(
                {
                    'step': step,
                    'state': state.value,
                    'source': 'calibrated',
                    'path': [list(point) for point in path],
                }
            )
            self.map.plan_count += 1
        while self._waypoint_index < len(path) - 1:
            waypoint = np.asarray(path[self._waypoint_index][:2], dtype=np.float64)
            if (
                np.linalg.norm(waypoint - np.asarray(start[:2]))
                >= self.calibrated_waypoint_tolerance
            ):
                break
            self._waypoint_index += 1
        return ControllerCommand(
            command_name,
            (path[self._waypoint_index :],),
        ).as_action()

    def _remaining_path_blocked(self, start) -> bool:
        if not self._planned_path:
            self._last_path_blockage = {'reason': 'missing_path'}
            return True
        blocked = self.navigation_occupancy.inflated_mask()
        if self.observed_only:
            blocked = blocked | ~self.navigation_occupancy.observed
        start_cell = self.navigation_occupancy.world_to_cell(start[:2])
        start_blocked = start_cell is None or bool(blocked[start_cell])
        points = [np.asarray(start[:2], dtype=np.float64)]
        points.extend(
            np.asarray(point[:2], dtype=np.float64)
            for point in self._planned_path[self._waypoint_index :]
        )
        spacing = self.navigation_occupancy.config.grid_resolution * 0.5
        remaining = self.replan_lookahead_distance
        checked_total = 0.0
        self._last_path_blockage = None
        for segment_index, (left, right) in enumerate(zip(points, points[1:])):
            distance = float(np.linalg.norm(right - left))
            checked_distance = distance if remaining is None else min(distance, remaining)
            if checked_distance <= 0:
                continue
            endpoint = left if distance == 0 else left + (right - left) * (checked_distance / distance)
            samples = max(1, int(np.ceil(checked_distance / spacing)))
            for ratio in np.linspace(0.0, 1.0, samples + 1)[1:]:
                cell = self.navigation_occupancy.world_to_cell(left + (endpoint - left) * ratio)
                if cell is None or blocked[cell]:
                    self._last_path_blockage = {
                        'reason': 'outside_map' if cell is None else 'inflated_occupancy',
                        'segment_index': segment_index,
                        'cell': None if cell is None else list(cell),
                        'distance_along_path': checked_total + checked_distance * float(ratio),
                        'start_cell_blocked': start_blocked,
                    }
                    return True
            checked_total += checked_distance
            if remaining is not None:
                remaining -= checked_distance
                if remaining <= 0:
                    return False
        return False

    def invalidate_path(self):
        """Release an execution goal when its owner completes or abandons it."""
        self._planned_state = None
        self._planned_path = None
        self._planned_goal = None
        self._waypoint_index = 0
        self._progress_position = None
        self._progress_step = None
        self._last_plan_step = None

    def _record_replan(self, reason: str, step: Optional[int], start):
        previous_goal = self._planned_goal
        previous_waypoint_index = self._waypoint_index
        self._planned_path = None
        self._planned_goal = None
        self._waypoint_index = 0
        self.replan_count += 1
        self.replan_reasons[reason] = self.replan_reasons.get(reason, 0) + 1
        self.last_replan = {
            'step': step,
            'reason': reason,
            'position': list(start),
            'previous_goal': None if previous_goal is None else list(previous_goal),
            'previous_waypoint_index': previous_waypoint_index,
            'map_revision': self.map.occupancy.revision,
            'blockage': self._last_path_blockage if 'blocked' in reason else None,
        }

    def velocity_action_for(
        self,
        decision: StateMachineDecision,
        robot_observation: dict,
        step: Optional[int] = None,
        max_forward_speed: float = 0.75,
        max_lateral_speed: float = 0.35,
    ) -> dict:
        fallback_action = self.action_for(decision, robot_observation, step=step)
        if 'move_by_speed' in fallback_action or not self._planned_path:
            return fallback_action
        position = np.asarray(robot_observation['position'][:2], dtype=np.float64)
        # Calibrated approach states must hit every configured waypoint with
        # centimetre tolerance, so they keep targeting the raw waypoint. All
        # other states track a carrot interpolated on the planned path, which
        # keeps the commanded heading (and thus the gait) continuous while
        # dense skeleton waypoints are consumed.
        calibrated = decision.state in (
            InteractionState.NAVIGATE_TO_OBJECT,
            InteractionState.NAVIGATE_TO_CARRY_GOAL,
        )
        if calibrated:
            target = np.asarray(self._planned_path[self._waypoint_index][:2], dtype=np.float64)
        else:
            target = self._lookahead_target(position)
        if self.safe_path_tracking:
            if not self._tracking_segment_clear(position, target):
                # A safe polyline does not imply a safe lookahead chord.
                target = np.asarray(self._planned_path[self._waypoint_index][:2], dtype=np.float64)
                self.tracking_chord_fallbacks += 1
                if not self._tracking_segment_clear(position, target):
                    self.tracking_segment_stops += 1
                    self._record_replan('tracking_segment_blocked', step, position)
                    return {'move_by_speed': [0.0, 0.0, 0.0]}
                if (
                    not calibrated
                    and np.linalg.norm(target - position) <= self.navigation_occupancy.config.grid_resolution * 0.5
                ):
                    # Within half a map cell, chasing this fallback point can
                    # reverse yaw/strafe commands on tiny position changes.
                    # Stop and replan without relaxing the forward-path check.
                    self._record_replan('reached_tracking_waypoint', step, position)
                    return {'move_by_speed': [0.0, 0.0, 0.0]}
        error_world = target - position
        distance = float(np.linalg.norm(error_world))
        if (
            decision.state == InteractionState.NAVIGATE_TO_OBJECT
            and self._waypoint_index == len(self._planned_path) - 1
            and distance <= 0.35
        ):
            return self._pose_target_action(
                target,
                robot_observation,
                desired_yaw=self.navigation_headings[InteractionState.NAVIGATE_TO_OBJECT],
            )
        yaw = _yaw(robot_observation['orientation'])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        forward_error = cos_yaw * error_world[0] + sin_yaw * error_world[1]
        lateral_error = -sin_yaw * error_world[0] + cos_yaw * error_world[1]
        desired_yaw = atan2(error_world[1], error_world[0])
        heading_error = (desired_yaw - yaw + pi) % (2.0 * pi) - pi
        rotation_speed = float(np.clip(1.8 * heading_error, -1.2, 1.2))
        if calibrated:
            forward_speed = float(np.clip(0.9 * forward_error, 0.0, max_forward_speed))
            lateral_speed = float(np.clip(1.2 * lateral_error, -max_lateral_speed, max_lateral_speed))
            if abs(heading_error) > 0.8:
                if (
                    decision.state == InteractionState.NAVIGATE_TO_CARRY_GOAL
                    and abs(heading_error) > 2.0
                ):
                    forward_speed = -0.18
                else:
                    forward_speed = 0.0
                lateral_speed = 0.0
            if distance < 0.30:
                forward_speed *= max(0.25, distance / 0.30)
                lateral_speed *= max(0.25, distance / 0.30)
            return ControllerCommand(
                'move_by_speed',
                (forward_speed, lateral_speed, rotation_speed),
            ).as_action()
        # Smooth speed law: cos^2 has a flat top, so small heading noise does
        # not modulate the forward speed; it fades to zero (turn in place)
        # as the error approaches 90 degrees without any hard threshold.
        heading_factor = max(0.0, cos(heading_error)) ** 2
        final_goal = np.asarray(self._planned_path[-1][:2], dtype=np.float64)
        final_distance = float(np.linalg.norm(final_goal - position))
        goal_factor = min(1.0, final_distance / self.final_goal_slow_radius)
        if self.safe_path_tracking:
            # The checked segment points toward the target, not along body X.
            # Scale both body components together so lateral saturation cannot
            # redirect translation into the inside of a corner while turning.
            velocity = np.array([forward_error, lateral_error]) / max(distance, 1e-9)
            speed = max_forward_speed * goal_factor
            if velocity[0] < -1e-9:
                speed = min(speed, min(max_forward_speed, 0.12) / abs(velocity[0]))
            if abs(velocity[1]) > 1e-9:
                speed = min(speed, max_lateral_speed / abs(velocity[1]))
            return ControllerCommand(
                'move_by_speed',
                (float(speed * velocity[0]), float(speed * velocity[1]), rotation_speed),
            ).as_action()
        forward_speed = float(max_forward_speed * heading_factor * goal_factor)
        lateral_speed = float(
            np.clip(1.2 * lateral_error, -max_lateral_speed, max_lateral_speed) * heading_factor
        )
        return ControllerCommand(
            'move_by_speed',
            (forward_speed, lateral_speed, rotation_speed),
        ).as_action()

    def _tracking_segment_clear(self, start, target):
        occupancy = self.navigation_occupancy
        left, right = occupancy.world_to_cell(start), occupancy.world_to_cell(target)
        return (left is not None and right is not None and self.map.planner.segment_clear(
            left, right, occupancy.inflated_mask() | ~occupancy.observed))

    def update_buffer_recovery(self, step, observation):
        """Temporarily leave a clearance buffer without relaxing normal routes."""
        if not self.safe_path_tracking:
            return False
        from .exploration_frontiers import buffer_exit_goal

        occupancy = self.navigation_occupancy
        position = np.asarray(observation['position'][:2], dtype=float)
        cell = occupancy.world_to_cell(position)
        observed = cell is not None and occupancy.observed[cell]
        in_buffer = observed and occupancy.inflated_mask()[cell]
        if not in_buffer and not self.buffer_recovery_active:
            return False
        if not self.buffer_recovery_active:
            self.buffer_recovery_active = True
            self._buffer_started = step
            self._buffer_exit = None
            self.invalidate_path()
            self.buffer_history.append({'step': step, 'event': 'buffer_exit_started', 'position': position.tolist()})
        if step - self._buffer_started >= self.buffer_recovery_timeout:
            if not self.buffer_recovery_failed:
                self.buffer_history.append({'step': step, 'event': 'buffer_exit_timeout'})
            self.buffer_recovery_failed = True
            return True
        if observed and not in_buffer and (
            self._buffer_exit is None or np.linalg.norm(np.asarray(self._buffer_exit) - position)
            <= max(0.05, occupancy.config.grid_resolution)
        ):
            self.buffer_history.append({'step': step, 'event': 'buffer_exit_completed', 'position': position.tolist()})
            self.buffer_recovery_active = False
            self._buffer_exit = None
            self.invalidate_path()
            return False
        self._buffer_exit = buffer_exit_goal(occupancy, position, self._buffer_exit)
        return True

    def buffer_recovery_action(self, observation, max_forward_speed, max_lateral_speed):
        from .exploration_frontiers import buffer_exit_goal

        stop = {'move_by_speed': [0.0, 0.0, 0.0]}
        if self.buffer_recovery_failed or observation['position'][2] < self.safe_base_height:
            return stop
        position = np.asarray(observation['position'][:2], dtype=float)
        # Revalidate immediately before issuing each motion command.
        target = buffer_exit_goal(self.navigation_occupancy, position, self._buffer_exit)
        if target is None:
            return stop
        self._buffer_exit = target
        delta = np.asarray(target) - position
        world = delta * min(1.0, 0.15 / max(np.linalg.norm(delta), 1e-9))
        yaw = _yaw(observation['orientation'])
        c, s = np.cos(yaw), np.sin(yaw)
        body = np.array([c * world[0] + s * world[1], -s * world[0] + c * world[1]])
        scale = min(1.0, max_forward_speed / max(abs(body[0]), 1e-9),
                    max_lateral_speed / max(abs(body[1]), 1e-9))
        return {'move_by_speed': [float(body[0] * scale), float(body[1] * scale), 0.0]}

    def _lookahead_target(self, position: np.ndarray) -> np.ndarray:
        """Carrot point a fixed arc length ahead on the remaining planned path.

        Walking the polyline robot -> active waypoint -> ... and interpolating
        keeps the target sliding continuously as waypoints are passed, instead
        of jumping 0.3 m sideways whenever the active waypoint switches.
        """

        remaining = float(self.velocity_lookahead_distance)
        current = np.asarray(position, dtype=np.float64)
        for waypoint in self._planned_path[self._waypoint_index :]:
            point = np.asarray(waypoint[:2], dtype=np.float64)
            segment = point - current
            length = float(np.linalg.norm(segment))
            if length > remaining and length > 0.0:
                return current + segment * (remaining / length)
            remaining -= length
            current = point
        return current

    @staticmethod
    def _pose_target_action(target, robot_observation, desired_yaw: float) -> dict:
        position = np.asarray(robot_observation['position'][:2], dtype=np.float64)
        error_world = np.asarray(target[:2], dtype=np.float64) - position
        distance = float(np.linalg.norm(error_world))
        yaw = _yaw(robot_observation['orientation'])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        forward_error = cos_yaw * error_world[0] + sin_yaw * error_world[1]
        lateral_error = -sin_yaw * error_world[0] + cos_yaw * error_world[1]
        heading_error = (desired_yaw - yaw + pi) % (2.0 * pi) - pi
        if abs(heading_error) < 0.25:
            translation_scale = 1.0
        elif abs(heading_error) < 0.60:
            translation_scale = 0.4
        else:
            translation_scale = 0.0
        translation = np.asarray(
            (
                np.clip(0.8 * forward_error, -0.12, 0.12),
                np.clip(0.8 * lateral_error, -0.08, 0.08),
            ),
            dtype=np.float64,
        )
        translation_norm = float(np.linalg.norm(translation))
        if distance > 0.04 and 0.0 < translation_norm < 0.08:
            translation *= 0.08 / translation_norm
        translation *= translation_scale
        return ControllerCommand(
            'move_by_speed',
            (
                float(translation[0]),
                float(translation[1]),
                float(np.clip(2.0 * heading_error, -0.8, 0.8)),
            ),
        ).as_action()

    def point_navigation_action(
        self,
        goal,
        robot_observation: dict,
        step: Optional[int] = None,
        velocity_control: bool = False,
        max_forward_speed: float = 0.75,
        max_lateral_speed: float = 0.35,
    ) -> dict:
        """Plan and track a point-navigation goal without exposing interaction states.

        The interaction demo still uses :meth:`action_for` directly.  This
        adapter keeps pure point-navigation callers independent from the
        deterministic interaction state machine while preserving the same
        tested planning implementation.
        """

        goal = _vector3(goal)
        if self.update_buffer_recovery(step if step is not None else 0, robot_observation):
            return self.buffer_recovery_action(robot_observation, max_forward_speed, max_lateral_speed)
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_GOAL,
            command=ControllerCommand('move_along_path', ((goal,),)),
        )
        if velocity_control:
            return self.velocity_action_for(
                decision,
                robot_observation,
                step=step,
                max_forward_speed=max_forward_speed,
                max_lateral_speed=max_lateral_speed,
            )
        return self.action_for(decision, robot_observation, step=step)

    def _plan_with_final_heading(self, state: InteractionState, start, goal):
        if (
            state == InteractionState.NAVIGATE_TO_GOAL
            and self.use_scene_graph
            and self.allow_goal_door_traversal
        ):
            portal_path = self._plan_through_open_door(goal)
            if portal_path is not None:
                return portal_path
        heading = self.navigation_headings.get(state)
        if heading is None:
            return self._map_plan(start, goal)
        approach = (
            goal[0] - self.final_approach_distance * cos(heading),
            goal[1] - self.final_approach_distance * sin(heading),
            goal[2],
        )
        if state == InteractionState.NAVIGATE_TO_DOOR:
            semantic_path = self._plan_around_pickup_pedestal(start, approach)
            if semantic_path is not None:
                path = list(semantic_path)
                path.append(goal)
                return tuple(path)
        approach_path = self._map_plan(start, approach)
        path = list(approach_path)
        if not path or np.linalg.norm(np.asarray(path[-1][:2]) - np.asarray(approach[:2])) > 1e-6:
            path.append(approach)
        path.append(goal)
        return tuple(path)

    def _plan_around_pickup_pedestal(self, start, approach):
        start_xy = np.asarray(start[:2], dtype=np.float64)
        candidates = [
            node
            for node in self.map.scene_graph.object_nodes()
            if node.label == 'pedestal'
            and node.observations >= 2
            and np.linalg.norm(np.asarray(node.position[:2]) - start_xy) < 3.0
        ]
        if not candidates:
            return None
        pedestal = max(candidates, key=lambda node: node.observations)
        egress = (
            pedestal.position[0] - 0.45,
            pedestal.position[1] + 0.55,
            approach[2],
        )
        clearance = (
            pedestal.position[0] + 0.45,
            pedestal.position[1] + 1.20,
            approach[2],
        )
        try:
            final_leg = self._map_plan(clearance, approach)
        except PlanningError:
            return None
        return (egress, clearance, *final_leg)

    def _plan_through_open_door(self, goal):
        nodes = self.map.scene_graph.object_nodes()
        hinges = [
            node
            for node in nodes
            if node.label == 'door_frame_hinge' and node.observations >= 2 and node.position[2] > 0.3
        ]
        latches = [
            node
            for node in nodes
            if node.label == 'door_frame_latch' and node.observations >= 2 and node.position[2] > 0.3
        ]
        if not hinges or not latches:
            return None
        hinge = max(hinges, key=lambda node: node.observations)
        latch = max(latches, key=lambda node: node.observations)
        center_x = (hinge.position[0] + latch.position[0]) / 2.0
        center_y = (hinge.position[1] + latch.position[1]) / 2.0
        self.map.occupancy.mark_free((center_x, center_y), radius=0.75)
        portal = (center_x + 0.70, center_y, goal[2])
        try:
            final_leg = self._map_plan(portal, goal)
        except PlanningError:
            return None
        path = [portal]
        path.extend(final_leg)
        return self._densify_waypoints(path, max_spacing=0.45)

    @property
    def navigation_occupancy(self):
        return self.map.occupancy


    def _map_plan(self, start, goal):
        self._last_plan_used_voronoi = False
        if self.voronoi_planner is not None:
            try:
                path = self.voronoi_planner.plan(start, goal)
            except PlanningError:
                self.voronoi_plan_fallbacks += 1
            else:
                self.voronoi_plan_count += 1
                self._last_plan_used_voronoi = True
                self.map.plan_count += 1
                return path
        return self.map.plan(start, goal, reinforce_semantics=self.use_semantic_occupancy)

    @staticmethod
    def _densify_waypoints(path, max_spacing: float):
        if len(path) < 2:
            return tuple(path)
        dense_path = [tuple(path[0])]
        for endpoint in path[1:]:
            start = np.asarray(dense_path[-1], dtype=np.float64)
            endpoint_array = np.asarray(endpoint, dtype=np.float64)
            distance = float(np.linalg.norm(endpoint_array[:2] - start[:2]))
            segments = max(1, int(np.ceil(distance / max_spacing)))
            for index in range(1, segments + 1):
                point = start + (endpoint_array - start) * (index / segments)
                dense_path.append(tuple(float(value) for value in point))
        return tuple(dense_path)

    @staticmethod
    def _aligned_push_action(command: ControllerCommand, robot_observation: dict):
        aligned_data = MapNavigationRuntime._aligned_speed_data(
            forward_speed=float(command.data[0]),
            orientation=robot_observation['orientation'],
        )
        return ControllerCommand(command.name, tuple(aligned_data)).as_action()

    @staticmethod
    def _aligned_speed_data(forward_speed: float, orientation):
        yaw = _yaw(orientation)
        heading_error = (0.0 - yaw + pi) % (2.0 * pi) - pi
        forward_speed = forward_speed if abs(heading_error) < 0.12 else 0.0
        rotation_speed = float(np.clip(2.5 * heading_error, -1.2, 1.2))
        return [forward_speed, 0.0, rotation_speed]

    def statistics(self) -> dict:
        if self.use_scene_graph:
            stats = self.map.statistics()
        else:
            stats = {
                'lidar_frames': self.map.lidar_frames,
                'rgb_frames': self.map.rgb_frames,
                'lidar_points': self.map.lidar_points,
                'rgb_points': self.map.rgb_points,
                'voxel_count': self.map.voxels.voxel_count,
                'lidar_voxel_count': self.map.voxels.lidar_voxel_count,
                'colored_voxel_count': self.map.voxels.colored_voxel_count,
                'occupied_cells': int(self.map.occupancy.occupied_mask().sum()),
                'observed_cells': int(self.map.occupancy.observed.sum()),
                'plans': self.map.plan_count,
            }
        stats.update(
            {
                'planning_failures': self.planning_failures,
                'tracking_chord_fallbacks': self.tracking_chord_fallbacks,
                'tracking_segment_stops': self.tracking_segment_stops,
                'buffer_recovery_active': self.buffer_recovery_active,
                'buffer_recovery_failed': self.buffer_recovery_failed,
                'buffer_recovery_history': list(self.buffer_history),
                'replans': self.replan_count,
                'replan_reasons': dict(sorted(self.replan_reasons.items())),
                'last_replan': self.last_replan,
                'planned_states': sorted(self._planned_paths),
                'planned_waypoints': {
                    state: len(path) for state, path in sorted(self._planned_paths.items())
                },
                'active_waypoint_index': self._waypoint_index,
                'active_waypoint': (
                    None
                    if not self._planned_path
                    else list(self._planned_path[min(self._waypoint_index, len(self._planned_path) - 1)])
                ),
                'last_safe_position': self._last_safe_position,
                'static_obstacle_boxes': len(self._static_obstacle_boxes),
                'static_occupied_cells': int(self.map.occupancy.static_occupied.sum()),
                'trajectory_static_obstacle_violations': self._static_obstacle_violations,
                'violated_static_obstacle_labels': sorted(self._violated_static_labels),
                'semantic_target': self.semantic_target,
                'semantic_detection_mode': self.semantic_detection_mode.value,
                'semantic_detection_effective_mode': (
                    SemanticDetectionMode.ISAAC.value
                    if self.semantic_detection_mode is SemanticDetectionMode.HYBRID
                    and self.open_vocabulary_perception is None
                    else self.semantic_detection_mode.value
                ),
                'open_vocabulary_available': self.open_vocabulary_perception is not None,
                'open_vocabulary_startup_error': self.open_vocabulary_startup_error,
                'open_vocabulary_attempts': self.open_vocabulary_attempts,
                'open_vocabulary_frames': self.open_vocabulary_frames,
                'open_vocabulary_failures': self.open_vocabulary_failures,
                'open_vocabulary_rate_limited_frames': self.open_vocabulary_rate_limited_frames,
                'open_vocabulary_empty_frames': self.open_vocabulary_empty_frames,
                'open_vocabulary_detections': self.open_vocabulary_detections,
                'open_vocabulary_partial_failures': self.open_vocabulary_partial_failures,
                'open_vocabulary_consecutive_failures': self.open_vocabulary_consecutive_failures,
                'open_vocabulary_last_error': self.open_vocabulary_last_error,
                'open_vocabulary_last_partial_errors': self.open_vocabulary_last_partial_errors,
                'open_vocabulary_last_labels': self.open_vocabulary_last_labels,
                'open_vocabulary_last_detection_step': self.open_vocabulary_last_detection_step,
                'open_vocabulary_last_success_step': self.open_vocabulary_last_success_step,
                'voronoi_plans': self.voronoi_plan_count,
                'lidar_self_hits_removed': self.lidar_self_hits_removed,
                'voronoi_plan_fallbacks': self.voronoi_plan_fallbacks,
            }
        )
        if self.semantic_voronoi is not None:
            stats['semantic_voronoi'] = self.semantic_voronoi.statistics()
        return stats

    def _update_static_obstacle_violations(self, position):
        xy = np.asarray(position[:2], dtype=np.float64)
        # Interaction standoffs may intentionally sit just outside a support
        # surface's AABB. Count only an actual center-point penetration here;
        # planning still uses the full static-obstacle inflation radius.
        padding = 0.0
        violated = False
        for label, minimum, maximum in self._static_obstacle_boxes:
            if np.all(xy >= minimum - padding) and np.all(xy <= maximum + padding):
                self._violated_static_labels.add(label)
                violated = True
        if violated:
            self._static_obstacle_violations += 1

    def save(self, output_prefix: str):
        if not output_prefix:
            return
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        points = self.map.voxels.points()
        colors = self.map.voxels.colors()
        skeleton = np.zeros_like(self.map.occupancy.observed, dtype=bool)
        if self.semantic_voronoi is not None:
            for cell in self.semantic_voronoi.update(force=True).skeleton_cells:
                skeleton[cell] = True
        np.savez_compressed(
            str(prefix) + '.npz',
            points=points,
            colors=colors,
            occupancy_log_odds=self.map.occupancy.log_odds,
            occupancy_observed=self.map.occupancy.observed,
            occupancy_inflated=self.map.occupancy.inflated_mask(),
            occupancy_static=self.map.occupancy.static_occupied,
            semantic_voronoi_skeleton=skeleton,
        )
        payload = {
            'mapping_config': asdict(self.map.config),
            'statistics': self.statistics(),
            'planned_paths': {
                state: [list(point) for point in path] for state, path in sorted(self._planned_paths.items())
            },
            'plan_history': self._plan_history,
        }
        if self.use_scene_graph:
            graph = self.map.snapshot()
            payload['scene_graph'] = {
                'nodes': [asdict(node) for node in graph.nodes],
                'edges': [asdict(edge) for edge in graph.edges],
                'label_counts': {
                    node.node_id: self.map.scene_graph.label_counts(node.node_id)
                    for node in graph.nodes if node.kind == 'object'
                },
            }
        if self.semantic_voronoi is not None:
            payload['semantic_voronoi'] = self.semantic_voronoi.to_dict()
        with Path(str(prefix) + '.json').open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)


def _camera_cloud(camera: dict):
    points = _points(camera.get('pointcloud'))
    rgba = camera.get('rgba')
    depth = camera.get('depth')
    if len(points) == 0 or rgba is None or depth is None:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            None,
        )

    rgba = np.asarray(rgba)
    depth = np.asarray(depth)
    if rgba.ndim != 3 or depth.ndim != 2 or rgba.shape[:2] != depth.shape:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            None,
        )
    mask = np.isfinite(depth) & (depth > 0.01) & (depth < 10000.0)
    colors = rgba[..., :3][mask]
    if len(points) == depth.size:
        points = points.reshape((*depth.shape, 3))[mask]
    count = min(len(points), len(colors))
    points = points[:count]
    colors = colors[:count]
    point_image = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    valid_indices = np.argwhere(mask)
    valid_indices = valid_indices[:count]
    point_image[valid_indices[:, 0], valid_indices[:, 1]] = points
    return points, colors, point_image


def _semantic_detections(camera: dict, point_image, step: int = 0) -> list:
    if point_image is None:
        return []
    bounding_boxes = camera.get('bounding_box_2d_tight')
    rgba = camera.get('rgba')
    if not isinstance(bounding_boxes, dict) or rgba is None:
        return []
    data = bounding_boxes.get('data', [])
    label_lookup = bounding_boxes.get('info', {}).get('idToLabels', {})
    height, width = point_image.shape[:2]
    detections = []
    for row in data:
        values = tuple(row.tolist()) if hasattr(row, 'tolist') else tuple(row)
        if len(values) < 5:
            continue
        semantic_id, x_min, y_min, x_max, y_max = values[:5]
        label_data = label_lookup.get(str(int(semantic_id)), label_lookup.get(str(semantic_id), {}))
        label = label_data.get('class') if isinstance(label_data, dict) else None
        label = _semantic_category(label)
        if not label or label in _NON_OBJECT_SEMANTIC_LABELS:
            continue
        if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
            continue
        x_min = max(0, min(width - 1, int(x_min)))
        x_max = max(x_min + 1, min(width, int(x_max)))
        y_min = max(0, min(height - 1, int(y_min)))
        y_max = max(y_min + 1, min(height, int(y_max)))
        if (x_max - x_min) * (y_max - y_min) < width * height * 0.001:
            continue
        if len(values) > 5 and float(values[5]) > 0.85:
            continue
        region = point_image[y_min:y_max, x_min:x_max]
        valid = np.isfinite(region).all(axis=2)
        if not valid.any():
            continue
        position = np.median(region[valid], axis=0)
        color_region = np.asarray(rgba)[y_min:y_max, x_min:x_max, :3]
        color = np.median(color_region[valid], axis=0)
        detections.append(
            SemanticDetection(
                label=str(label),
                position=tuple(float(value) for value in position),
                color=tuple(int(value) for value in np.clip(color, 0, 255)),
                step=int(step),
                sources=('isaac',),
            )
        )
    return detections


def _deduplicate_semantic_detections(detections) -> tuple:
    """Fuse duplicate observations produced by multiple sources in one frame."""

    merged = []
    for detection in detections:
        match_index = None
        incoming_embedding = detection.embedding
        for index, candidate in enumerate(merged):
            distance = float(
                np.linalg.norm(
                    np.asarray(candidate.position[:2], dtype=np.float64)
                    - np.asarray(detection.position[:2], dtype=np.float64)
                )
            )
            if distance >= 0.75:
                continue
            same_label = candidate.label.strip().casefold() == detection.label.strip().casefold()
            similar_embedding = False
            if candidate.embedding is not None and incoming_embedding is not None:
                left = np.asarray(candidate.embedding, dtype=np.float64)
                right = np.asarray(incoming_embedding, dtype=np.float64)
                if left.shape == right.shape and left.size:
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    similar_embedding = denominator > 1e-12 and float(np.dot(left, right) / denominator) >= 0.86
            if same_label or similar_embedding:
                match_index = index
                break
        if match_index is None:
            merged.append(detection)
            continue

        previous = merged[match_index]
        # Prefer the geometrically denser observation, while retaining
        # complementary appearance information from either source.
        geometry = detection if detection.point_count > previous.point_count else previous
        appearance = detection if detection.embedding is not None else previous
        color_source = detection if detection.color is not None else previous
        label_source = detection if detection.confidence > previous.confidence else previous
        merged[match_index] = SemanticDetection(
            label=label_source.label,
            position=geometry.position,
            color=color_source.color,
            confidence=max(float(previous.confidence), float(detection.confidence)),
            embedding=appearance.embedding,
            point_count=max(int(previous.point_count), int(detection.point_count)),
            step=max(int(previous.step), int(detection.step)),
            sources=tuple(dict.fromkeys((*previous.sources, *detection.sources))),
            label_evidence=tuple(dict.fromkeys((
                *(previous.label_evidence or (previous.label,)),
                *(detection.label_evidence or (detection.label,)),
            ))),
        )
    return tuple(merged)


def _semantic_category(label) -> str:
    if not label:
        return ''
    return str(label).strip().lower().split('/', 1)[0]


def _points(value) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return array.reshape(-1, 3)


def _vector3(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) < 3:
        raise ValueError('expected a three-dimensional vector')
    return tuple(float(component) for component in array[:3])


def _yaw(quaternion) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


__all__ = [
    'MapNavigationRuntime',
    'SemanticDetectionMode',
]
