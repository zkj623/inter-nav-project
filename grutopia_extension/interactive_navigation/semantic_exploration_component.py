"""Go2-independent orchestration for semantic target exploration."""

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, SceneGraphNode, canonical_semantic_label
from grutopia_extension.interactive_navigation.mapping_runtime import (
    MapNavigationRuntime,
    SemanticDetectionMode,
)
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiConfig
from grutopia_extension.interactive_navigation.semantic_exploration import (
    AdaptiveExplorationPlanner, ExplorationConfig, ExplorationDecision,
)
from grutopia_extension.interactive_navigation.exploration_frontiers import frontier_goals, local_unknown


class SemanticExplorationStatus(str, Enum):
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


@dataclass(frozen=True)
class SemanticExplorationConfig:
    target_query: str
    mapping: MappingConfig = field(default_factory=MappingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    voronoi: SemanticVoronoiConfig = field(default_factory=SemanticVoronoiConfig)
    max_steps: int = 12_000
    target_distance: float = 0.70
    frontier_reached_distance: float = 0.30
    fall_height: float = 0.12
    safe_base_height: float = 0.22
    # `isaac` = simulator ground-truth labels only, `open_vocab` = VLM
    # detections only, `hybrid` = fused. See SemanticDetectionMode.
    semantic_detection_mode: str = 'hybrid'
    open_vocabulary_startup_error: Optional[str] = None
    target_semantic_classifier: str = 'clip'
    target_min_observations: int = 2
    target_embedding_threshold: float = 0.24
    require_lexical_confirmation: bool = True
    target_confirmation_min_label_observations: int = 8
    target_confirmation_min_label_fraction: float = 0.60
    frontier_selection_interval: int = 80
    # Incremental Voronoi updates keep frequent topology refreshes cheap; a
    # short interval also keeps changed-cell windows small and splice-able.
    topology_update_interval: int = 40
    frontier_progress_timeout: int = 600
    frontier_goal_timeout: int = 4000
    frontier_scan_timeout: int = 2400
    frontier_progress_distance: float = 0.15
    max_forward_speed: float = 0.80
    max_lateral_speed: float = 0.20
    enable_target_cues: bool = True
    target_cue_stale_steps: int = 480
    target_cue_navigation_steps: int = 1200
    target_cue_observe_steps: int = 96
    target_cue_cooldown_steps: int = 800

    def __post_init__(self):
        if self.target_semantic_classifier not in ('clip', 'qwen-vl'):
            raise ValueError('target_semantic_classifier must be clip or qwen-vl')
        if not self.target_query.strip():
            raise ValueError('target_query cannot be empty')
        object.__setattr__(
            self,
            'semantic_detection_mode',
            SemanticDetectionMode.parse(self.semantic_detection_mode).value,
        )
        for name in (
            'max_steps',
            'target_min_observations',
            'target_confirmation_min_label_observations',
            'frontier_selection_interval',
            'topology_update_interval',
            'frontier_progress_timeout',
            'frontier_goal_timeout',
            'frontier_scan_timeout',
            'target_cue_stale_steps',
            'target_cue_navigation_steps',
            'target_cue_observe_steps',
            'target_cue_cooldown_steps',
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('target_distance', 'frontier_reached_distance', 'safe_base_height', 'frontier_progress_distance'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0.0 < self.target_confirmation_min_label_fraction <= 1.0:
            raise ValueError('target_confirmation_min_label_fraction must be in (0, 1]')


@dataclass(frozen=True)
class SemanticExplorationResult:
    status: SemanticExplorationStatus
    step: int
    target_query: str
    position: Tuple[float, float, float]
    target_position: Optional[Tuple[float, float, float]]
    failure_reason: Optional[str]
    statistics: dict
    approach_goal: Optional[Tuple[float, float, float]] = None
    goal_distance: Optional[float] = None
    arrival_threshold: float = 0.70

    @property
    def terminal(self) -> bool:
        return self.status != SemanticExplorationStatus.RUNNING

    @property
    def success(self) -> bool:
        return self.status == SemanticExplorationStatus.SUCCEEDED

    def as_event(self) -> dict:
        """Keep terminal stdout small; the map and run summary hold details."""
        return {
            'event': 'semantic_exploration_result',
            'status': self.status.value,
            'success': self.success,
            'step': self.step,
            'target_query': self.target_query,
            'position': [round(value, 3) for value in self.position],
            'goal_distance_m': None if self.goal_distance is None else round(self.goal_distance, 3),
            'arrival_threshold_m': self.arrival_threshold,
            'reason': self.failure_reason,
        }


@dataclass
class _CueTrack:
    cue_id: str
    position: Tuple[float, float, float]
    confidence: float
    last_step: int
    observations: int = 1
    classified_label: str = 'unknown'
    cooldown_until: int = -1


class SemanticExplorationComponent:
    """Fuse observations, select semantic/frontier goals, and drive point navigation."""

    def __init__(self, config: SemanticExplorationConfig, perception=None, scorer=None):
        self.config = config
        self.perception = perception
        self.mapping = MapNavigationRuntime(
            mapping_config=replace(config.mapping, sticky_free_scale=0.25, object_position_window=4),
            safe_base_height=config.safe_base_height,
            replan_interval_steps=120,
            replan_only_if_blocked=True,
            replan_lookahead_distance=2.0,
            allow_goal_door_traversal=False,
            use_scene_graph=True,
            use_semantic_occupancy=False,
            # LiDAR supplies occupancy; well-observed people still receive
            # the runtime's explicit semantic obstacle protection.
            use_rgb_occupancy=False,
            rgb_clears_free_space=False,
            semantic_target=config.target_query,
            open_vocabulary_perception=perception,
            semantic_detection_mode=config.semantic_detection_mode,
            open_vocabulary_startup_error=config.open_vocabulary_startup_error,
            use_semantic_voronoi=True,
            semantic_voronoi_config=config.voronoi,
            topology_update_interval=config.topology_update_interval,
            prefer_voronoi_paths=True,
            safe_path_tracking=True,
        )
        self.planner = AdaptiveExplorationPlanner(config.exploration, scorer=scorer)
        self.last_decision: Optional[ExplorationDecision] = None
        self._last_selection_step = -config.frontier_selection_interval
        self._frontier_phase = 'scan'
        self._frontier_failure = None
        self._frontier_paused = False
        self._frontier_steps = 0
        self._last_update_step = None
        self._last_yaw = None
        self._scan_angle = 0.0
        self._scan_started = 0
        self._scan_frames = self.mapping.map.lidar_frames
        self._full_scan_lidar = False
        self._observed_frontiers = {}
        self._pending_visit = None
        self._minimum_base_up = 1.0
        self._initial_observed_cells = None
        self.current_goal: Optional[Tuple[float, float, float]] = None
        self.current_frontier_id: Optional[str] = None
        self.target_node: Optional[SceneGraphNode] = None
        self.target_navigation_position: Optional[Tuple[float, float, float]] = None
        self.decision_history = []
        self._target_embedding = None
        self._target_embedding_attempted = False
        self._target_lexical = False
        self._rejected_embedding_targets = set()
        self._trajectory = []
        self._cue_tracks = []
        self._next_cue_id = 0
        self.target_cue: Optional[_CueTrack] = None
        self._cue_started_step = None
        self._cue_observe_step = None
        self._cue_goal = None
        self.cue_history = []
        self.frontier_history = []
        self._sync_runtime_state()

    @staticmethod
    def warmup_action() -> dict:
        return {'move_by_speed': [0.0, 0.0, 0.0]}

    @property
    def state(self) -> str:
        if self.mapping.buffer_recovery_active:
            return 'clear_start'
        if self.target_node is not None:
            return 'navigate_to_semantic_target'
        if self.target_cue is not None:
            return 'observe_target_cue' if self._cue_observe_step is not None else 'navigate_to_target_cue'
        return f'explore_{self._frontier_phase}'

    def _node_label_evidence(self, node: SceneGraphNode) -> Tuple[int, int, int]:
        counts = self.mapping.map.scene_graph.label_counts(node.node_id)
        if not counts:
            counts = {node.label: node.observations}
        query = _normalize_label(self.config.target_query)
        matching_counts = [
            count for label, count in counts.items()
            if ((canonical_semantic_label(label) == canonical_semantic_label(query))
                if self.config.target_semantic_classifier == 'qwen-vl'
                else (query in _normalize_label(label) or _normalize_label(label) in query))
        ]
        return sum(matching_counts), sum(counts.values()), max(matching_counts, default=0)

    def _target_label_support(self) -> Tuple[int, int]:
        if self.target_node is None:
            return 0, 0
        matching, total, _ = self._node_label_evidence(self.target_node)
        return matching, total

    @property
    def target_confirmed(self) -> bool:
        if self.target_node is None:
            return False
        if not self.config.require_lexical_confirmation:
            return True
        matching, total, repeated_label = self._node_label_evidence(self.target_node)
        return (
            self._target_lexical
            and matching >= self.config.target_confirmation_min_label_observations
            and total > 0
            and (
                matching / total >= self.config.target_confirmation_min_label_fraction
                # A stable refrigerator can also be called "door" in most
                # frames. Repeated identical target labels are stronger
                # evidence than their fraction of all node observations.
                or repeated_label >= self.config.target_confirmation_min_label_observations
            )
        )

    def update(self, step: int, robot_observation: dict):
        self.mapping.update(step, robot_observation)
        position = _position(robot_observation)
        self._trajectory.append(position)
        delta = 1 if self._last_update_step is None else step - self._last_update_step
        if delta <= 0:
            raise ValueError('update steps must increase strictly')
        self._last_update_step = step
        orientation = np.asarray(robot_observation['orientation'], dtype=float)
        norm = np.linalg.norm(orientation)
        if orientation.shape != (4,) or not np.isfinite(orientation).all() or norm < 1e-9:
            raise ValueError('orientation must be a finite nonzero quaternion')
        orientation = orientation / norm
        w, x, y, z = orientation
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        turn = 0.0 if self._last_yaw is None else (yaw - self._last_yaw + np.pi) % (2 * np.pi) - np.pi
        self._last_yaw = yaw
        self._minimum_base_up = min(self._minimum_base_up, float(1 - 2 * (x*x + y*y)))
        lidar = robot_observation.get('sensors', {}).get('lidar', {})
        self._full_scan_lidar = lidar.get('horizontal_fov', 0) >= 359 and lidar.get('rotation_frequency', -1) == 0
        self._collect_target_cues(step)
        if self.target_node is None:
            self.target_node = self._best_target_node()
        else:
            self.target_node = next(
                (
                    node
                    for node in self.mapping.map.scene_graph.object_nodes()
                    if node.node_id == self.target_node.node_id
                ),
                self.target_node,
            )
            lexical = self._lexical_target_node()
            if lexical is None:
                self._target_lexical = False
            elif lexical.node_id == self.target_node.node_id:
                self._target_lexical = True
            else:
                # An embedding-only lock is provisional. For two lexical
                # nodes, prefer evidence for the requested label before total
                # observations; generic "door" frames must not hold the lock
                # against a better-observed refrigerator.
                if not self._target_lexical or (
                    (self._node_label_evidence(lexical)[0], lexical.observations, lexical.confidence)
                    > (
                        self._node_label_evidence(self.target_node)[0],
                        self.target_node.observations,
                        self.target_node.confidence,
                    )
                ):
                    self.target_node = lexical
                    self._target_lexical = True
                    self.target_navigation_position = None
        if (
            self.config.require_lexical_confirmation
            and self.target_node is not None
            and not self.target_confirmed
            and self.target_navigation_position is not None
            and _distance_xy(position, self.target_navigation_position) <= self.config.target_distance
        ):
            # An embedding match is a place to investigate, not proof that
            # the named object was found. Continue exploring after reaching it.
            self._rejected_embedding_targets.add(self.target_node.node_id)
            self.target_node = None
            self.target_navigation_position = None
            self.current_goal = None
            self.current_frontier_id = None
        if self.target_node is not None:
            if self.target_cue is not None:
                self._finish_target_cue(step, 'semantic_target_selected', cooldown=False)
            if self.target_navigation_position is None:
                self.target_navigation_position = self._target_approach_position(
                    self.target_node,
                    position,
                )
            self._pause_frontier()
            self.current_goal = self.target_navigation_position
            self.current_frontier_id = None
            self.mapping.update_buffer_recovery(step, robot_observation)
            self._sync_runtime_state()
            return

        if self._update_target_cue(step, position):
            self._pause_frontier()
            self.mapping.update_buffer_recovery(step, robot_observation)
            self._sync_runtime_state()
            return

        if self._frontier_paused:
            self._frontier_paused = False
            self.current_goal = self.current_frontier_id = self._pending_visit = None
            self.planner.failure_counts.clear()
            self.planner.clear_blacklist()
            self.mapping.buffer_recovery_active = False
            self.mapping.buffer_recovery_failed = False
            self._begin_frontier_scan()
            self._frontier_event('resumed')
        self._frontier_steps += delta
        self._update_frontier(step, robot_observation, position, turn)
        self._sync_runtime_state()

    def _frontier_event(self, kind, **details):
        self.frontier_history.append(dict(step=self._last_update_step, kind=kind, **details))

    def _pause_frontier(self):
        if not self._frontier_paused:
            self._frontier_paused = True
            self.mapping.invalidate_path()
            self._frontier_event('paused')

    def _begin_frontier_scan(self):
        self._frontier_phase = 'scan'
        self._scan_angle = 0.0
        self._scan_started = self._frontier_steps
        self._scan_frames = self.mapping.map.lidar_frames
        self.mapping.invalidate_path()

    def _fail_frontier(self, reason):
        if self.current_frontier_id is not None:
            key = self.current_frontier_id
            self.planner.record_failure(key)
            self._frontier_event('execution_failed', goal=key, reason=reason,
                                 attempts=self.planner.failure_counts[key])
        self.current_goal = self.current_frontier_id = self._pending_visit = None
        self._begin_frontier_scan()

    def _finish_frontier(self, reason):
        self._frontier_failure = reason
        self._frontier_phase = 'exhausted' if 'exhausted' in reason else 'failed'
        self.current_goal = self.current_frontier_id = None
        self.mapping.invalidate_path()
        self._frontier_event('search_finished', reason=reason)

    def _update_frontier(self, step, observation, position, turn):
        if self._frontier_failure is not None:
            return
        recovering = self.mapping.buffer_recovery_active
        if self.mapping.update_buffer_recovery(step, observation):
            if not recovering:
                self._fail_frontier('start_in_clearance_buffer')
                self._frontier_event('blocked_start', position=list(position))
            self._frontier_phase = 'clear_start'
            if self.mapping.buffer_recovery_failed:
                self._finish_frontier('blocked_start_timeout')
            return
        if recovering:
            self._frontier_event('start_cleared', position=list(position))
            self._begin_frontier_scan()
            return
        if self._frontier_phase == 'scan':
            self._scan_angle += turn
            if self._frontier_steps - self._scan_started > self.config.frontier_scan_timeout:
                self._finish_frontier('scan_timeout')
                return
            fresh_frames = self.mapping.map.lidar_frames - self._scan_frames
            complete = (self._full_scan_lidar and fresh_frames >= 8) or (
                self._scan_angle >= 2 * np.pi - 0.15 and fresh_frames > 0
            )
            if not complete:
                return
            occupancy = self.mapping.map.occupancy
            observed = int(occupancy.observed.sum())
            if self._initial_observed_cells is None:
                self._initial_observed_cells = observed
            self._frontier_event('scan_completed', angle=self._scan_angle, observed_cells=observed)
            if self._pending_visit is not None:
                key, xy = self._pending_visit
                self.planner.record_arrival(key)
                self._observed_frontiers[key] = (xy, local_unknown(occupancy, xy))
                self._pending_visit = None
            self._frontier_phase = 'select'
        if self.current_goal is not None:
            if _distance_xy(position, self.current_goal) <= self.config.frontier_reached_distance:
                self._frontier_event('goal_reached', goal=list(self.current_goal))
                self._pending_visit = (self.current_frontier_id, self.current_goal[:2])
                self.current_goal = self.current_frontier_id = None
                self._begin_frontier_scan()
            else:
                if _distance_xy(position, self._frontier_progress_position) >= self.config.frontier_progress_distance:
                    self._frontier_progress_position = position
                    self._frontier_progress_step = self._frontier_steps
                if self._frontier_steps - self._frontier_progress_step >= self.config.frontier_progress_timeout:
                    self._fail_frontier('no_motion_progress')
                elif self._frontier_steps - self._frontier_goal_step >= self.config.frontier_goal_timeout:
                    self._fail_frontier('goal_timeout')
        elif (self._frontier_phase == 'select'
              and self._frontier_steps - self._last_selection_step >= self.config.frontier_selection_interval):
            self._select_frontier(step, position)

    def _select_frontier(self, step, position):
        occupancy = self.mapping.map.occupancy
        snapshot = self.mapping.semantic_voronoi.update()
        excluded = set(self.planner.blacklist)
        for key, (xy, unknown) in list(self._observed_frontiers.items()):
            if abs(local_unknown(occupancy, xy) - unknown) < 10:
                excluded.add(key)
            else:
                del self._observed_frontiers[key]
        goals = frontier_goals(occupancy, snapshot, position, excluded)
        # Keep the original planner/scorer interface. Only candidate geometry
        # and the geometric score come from planning's reachable-frontier rule.
        frontiers = [dict(id=g.key, position=g.position, path_distance=g.distance,
                          information_gain=g.information_gain, geometric_score=g.score,
                          reachable_boundary=True) for g in goals]
        decision, goal_xy = self.planner.select_goal(
            robot_position=position, frontiers=frontiers, semantic_voronoi=snapshot,
            occupancy=occupancy, target_query=self.config.target_query,
        )
        self.last_decision = decision
        self._last_selection_step = self._frontier_steps
        if goal_xy is None:
            cell = occupancy.world_to_cell(position[:2])
            if cell is None or occupancy.inflated_mask()[cell] or not occupancy.observed[cell]:
                reason = 'robot_outside_traversable_map'
            else:
                reason = ('observation_goals_exhausted_after_failures' if self.planner.failure_counts
                          else 'reachable_observation_goals_exhausted')
            self._finish_frontier(reason)
            return
        self.current_frontier_id = decision.selected_frontier_id
        self.current_goal = (*goal_xy, position[2])
        self._frontier_phase = 'navigate'
        self._frontier_goal_step = self._frontier_progress_step = self._frontier_steps
        self._frontier_progress_position = position
        self.mapping.invalidate_path()
        self.decision_history.append(dict(
            step=step, mode=decision.mode.value, frontier_id=decision.selected_frontier_id,
            reason=decision.reason, used_model=decision.used_model, fell_back=decision.fell_back,
            goal=list(self.current_goal),
            candidates=[dict(id=c.frontier_id, score=c.score, geometric_score=c.geometric_score,
                             semantic_score=c.semantic_score, path_distance=c.path_distance,
                             information_gain=c.information_gain) for c in decision.candidates],
        ))
        self._frontier_event('goal_selected', key=self.current_frontier_id, position=list(goal_xy))

    def action(self, step: int, robot_observation: dict) -> dict:
        exploring = self.target_node is None and self.target_cue is None
        if exploring and self._frontier_failure is not None:
            return self.warmup_action()
        if self.mapping.update_buffer_recovery(step, robot_observation):
            return self.mapping.buffer_recovery_action(
                robot_observation, self.config.max_forward_speed, self.config.max_lateral_speed)
        if self.target_cue is not None and self._cue_observe_step is not None:
            # Face the candidate while normal camera updates collect evidence.
            position = _position(robot_observation)
            delta = np.asarray(self.target_cue.position[:2]) - np.asarray(position[:2])
            w, x, y, z = robot_observation.get('orientation', (1.0, 0.0, 0.0, 0.0))
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            error = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
            return {'move_by_speed': [0.0, 0.0, float(np.clip(error, -0.35, 0.35))]}
        if self.current_goal is None:
            if exploring and self._frontier_phase == 'scan':
                return {'move_by_speed': [0.0, 0.0, 0.0 if self._full_scan_lidar else 0.6]}
            return self.warmup_action()
        # Velocity control with a sliding lookahead target: the discrete
        # move_along_path controller re-targets every dense skeleton waypoint
        # and collapses its forward speed on each heading jump, which showed
        # up as a stuttering gait once waypoints were spaced 0.3 m apart.
        failures = self.mapping.planning_failures
        action = self.mapping.point_navigation_action(
            self.current_goal,
            robot_observation,
            step=step,
            velocity_control=True,
            max_forward_speed=self.config.max_forward_speed,
            max_lateral_speed=self.config.max_lateral_speed,
        )

        if exploring and self.mapping.planning_failures > failures:
            self._fail_frontier('path_unreachable')
            self._sync_runtime_state()
            return self.warmup_action()
        return action

    def evaluate(self, step: int, robot_observation: dict) -> SemanticExplorationResult:
        position = _position(robot_observation)
        approach_goal = self.target_navigation_position
        goal_distance = None if approach_goal is None else _distance_xy(position, approach_goal)
        status = SemanticExplorationStatus.RUNNING
        failure_reason = None
        if self.mapping.buffer_recovery_failed:
            status = SemanticExplorationStatus.FAILED
            failure_reason = 'buffer_exit_timeout'
        elif position[2] < self.config.fall_height:
            status = SemanticExplorationStatus.FAILED
            failure_reason = 'robot_fell'
        elif (
            not self.mapping.buffer_recovery_active
            and self.target_node is not None
            and self.target_confirmed
            and goal_distance is not None
            and goal_distance <= self.config.target_distance
        ):
            status = SemanticExplorationStatus.SUCCEEDED
        elif (self.target_node is None and self.target_cue is None
              and self._frontier_failure is not None):
            status = SemanticExplorationStatus.FAILED
            failure_reason = self._frontier_failure
        elif step + 1 >= self.config.max_steps:
            status = SemanticExplorationStatus.FAILED
            failure_reason = 'global_step_limit'
        terminal = status != SemanticExplorationStatus.RUNNING
        return SemanticExplorationResult(
            status=status,
            step=step,
            target_query=self.config.target_query,
            position=position,
            target_position=(
                None if self.target_node is None else tuple(float(value) for value in self.target_node.position)
            ),
            failure_reason=failure_reason,
            # Full statistics are materialized only for terminal results.
            statistics=self.statistics() if terminal else {},
            approach_goal=approach_goal,
            goal_distance=goal_distance,
            arrival_threshold=self.config.target_distance,
        )

    def progress_event(self, step: int, robot_observation: dict) -> dict:
        position = _position(robot_observation)
        goal_distance = None if self.current_goal is None else _distance_xy(position, self.current_goal)
        return {
            'event': 'semantic_exploration_progress',
            'step': step,
            'state': self.state,
            'position': [round(value, 2) for value in position],
            'target_found': self.target_node is not None,
            'target_confirmed': self.target_confirmed,
            'goal_distance_m': None if goal_distance is None else round(goal_distance, 2),
        }

    def statistics(self) -> dict:
        stats = dict(self.mapping.statistics())
        stats.update(
            {
                'exploration_state': self.state,
                'target_query': self.config.target_query,
                'target_found': self.target_node is not None,
                'target_confirmed': self.target_confirmed,
                'target_label_support': self._target_label_support()[0],
                'rejected_provisional_targets': len(self._rejected_embedding_targets),
                'target_node': None if self.target_node is None else self.target_node.node_id,
                'target_match': (
                    None
                    if self.target_node is None
                    else ('lexical' if self._target_lexical else 'embedding')
                ),
                'target_match_method': (
                    None
                    if self.target_node is None
                    else ('lexical' if self._target_lexical else 'embedding')
                ),
                'target_sources': (
                    [] if self.target_node is None else list(self.target_node.sources)
                ),
                'active_frontier': self.current_frontier_id,
                'active_goal': None if self.current_goal is None else list(self.current_goal),
                'exploration_decisions': len(self.decision_history),
                'frontier_visits': dict(self.planner.visit_counts),
                'frontier_observed_locations': len(self._observed_frontiers),
                'frontier_failures': dict(self.planner.failure_counts),
                'frontier_blacklist': sorted(self.planner.blacklist),
                'trajectory_points': len(self._trajectory),
                'frontier_execution': {
                    'phase': self._frontier_phase, 'failure_reason': self._frontier_failure,
                    'active_steps': self._frontier_steps, 'minimum_base_up': self._minimum_base_up,
                    'initial_observed_cells': self._initial_observed_cells,
                },
                'target_cue_id': None if self.target_cue is None else self.target_cue.cue_id,
                'target_cue_transitions': len(self.cue_history),
            }
        )
        return stats

    def save(self, output_prefix: str):
        self.mapping.save(output_prefix)
        if not output_prefix:
            return
        metadata_path = Path(str(output_prefix) + '.json')
        with metadata_path.open('r', encoding='utf-8') as input_file:
            payload = json.load(input_file)
        payload['semantic_exploration'] = {
            'target_query': self.config.target_query,
            'statistics': self.statistics(),
            'decisions': self.decision_history,
            'trajectory': [list(point) for point in self._trajectory],
            'target_cue_history': self.cue_history,
            'frontier_history': self.frontier_history,
        }
        with metadata_path.open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)

    def _candidate_target_nodes(self) -> list:
        return [
            node
            for node in self.mapping.map.scene_graph.object_nodes()
            if node.observations >= self.config.target_min_observations
        ]

    def _lexical_target_node(self, nodes=None) -> Optional[SceneGraphNode]:
        if nodes is None:
            nodes = self._candidate_target_nodes()
        normalized_target = _normalize_label(self.config.target_query)
        lexical = []
        for node in nodes:
            representative_matches = (
                canonical_semantic_label(node.label) == canonical_semantic_label(normalized_target)
                if self.config.target_semantic_classifier == 'qwen-vl'
                else (normalized_target in _normalize_label(node.label)
                      or _normalize_label(node.label) in normalized_target)
            )
            matching, _, repeated_label = self._node_label_evidence(node)
            if representative_matches or (
                matching >= self.config.target_confirmation_min_label_observations
                and repeated_label >= self.config.target_confirmation_min_label_observations
            ):
                lexical.append(node)
        if not lexical:
            return None
        # Rank target-specific evidence before generic "door" observations.
        return max(
            lexical,
            key=lambda node: (
                self._node_label_evidence(node)[0],
                node.observations,
                node.confidence,
            ),
        )

    def _best_target_node(self) -> Optional[SceneGraphNode]:
        nodes = self._candidate_target_nodes()
        if not nodes:
            return None
        lexical = self._lexical_target_node(nodes)
        if lexical is not None:
            self._target_lexical = True
            return lexical
        if self.config.target_semantic_classifier == 'qwen-vl':
            # Qwen categories are explicit; CLIP similarity is not a class vote.
            return None
        target_embedding = self._text_embedding()
        if target_embedding is None:
            return None
        scored = []
        for node in nodes:
            if node.node_id in self._rejected_embedding_targets or node.embedding is None:
                continue
            embedding = np.asarray(node.embedding, dtype=np.float32)
            if embedding.shape != target_embedding.shape:
                continue
            scored.append((float(np.dot(target_embedding, embedding)), node))
        if not scored:
            return None
        similarity, node = max(scored, key=lambda item: item[0])
        if similarity < self.config.target_embedding_threshold:
            return None
        self._target_lexical = False
        return node

    def _text_embedding(self):
        if self._target_embedding_attempted:
            return self._target_embedding
        self._target_embedding_attempted = True
        if self.perception is None or not hasattr(self.perception, 'embed_text'):
            return None
        try:
            self._target_embedding = np.asarray(
                self.perception.embed_text(self.config.target_query),
                dtype=np.float32,
            )
        except Exception:
            self._target_embedding = None
        return self._target_embedding

    def _target_approach_position(self, node: SceneGraphNode, robot_position):
        return self._approach_position(node.position, robot_position)

    def _approach_position(self, target_position, robot_position):
        target = np.asarray(target_position[:2], dtype=np.float64)
        robot = np.asarray(robot_position[:2], dtype=np.float64)
        direction = robot - target
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            direction = np.array((-1.0, 0.0), dtype=np.float64)
        else:
            direction /= norm
        desired = target + direction * max(
            self.config.frontier_reached_distance,
            self.config.mapping.robot_radius + 0.20,
        )
        occupancy = self.mapping.map.occupancy
        desired_cell = occupancy.world_to_cell(desired)
        blocked = occupancy.inflated_mask()
        if desired_cell is not None:
            candidates = []
            search_radius = max(
                2,
                int(np.ceil(self.config.target_distance / occupancy.config.grid_resolution)),
            )
            for radius in range(search_radius + 1):
                for row in range(desired_cell[0] - radius, desired_cell[0] + radius + 1):
                    for col in range(desired_cell[1] - radius, desired_cell[1] + radius + 1):
                        cell = (row, col)
                        if (
                            occupancy.in_bounds(cell)
                            and occupancy.observed[cell]
                            and not blocked[cell]
                        ):
                            candidates.append(cell)
                if candidates:
                    break
            if candidates:
                best = min(
                    candidates,
                    key=lambda cell: np.linalg.norm(
                        np.asarray(occupancy.cell_to_world(cell)) - desired
                    ),
                )
                desired = np.asarray(occupancy.cell_to_world(best), dtype=np.float64)
        return (float(desired[0]), float(desired[1]), float(robot_position[2]))

    def _collect_target_cues(self, step):
        if not self.config.enable_target_cues or self.config.target_semantic_classifier != 'qwen-vl':
            return
        # A frame can remain available between camera ticks: consume its capture
        # step only once per spatial track, never as new semantic evidence.
        if getattr(self.perception, 'last_query_status', None) != 'ok':
            return
        self._cue_tracks = [
            track for track in self._cue_tracks
            if track is self.target_cue or step <= max(
                track.last_step + self.config.target_cue_stale_steps, track.cooldown_until,
            )
        ]
        for cue in sorted(getattr(self.perception, 'last_target_cues', ()),
                          key=lambda item: item.confidence, reverse=True):
            if (canonical_semantic_label(cue.target_query) != canonical_semantic_label(self.config.target_query)
                    or cue.step > step or step - cue.step > self.config.target_cue_stale_steps
                    or not np.isfinite(cue.position).all()
                    or self.mapping.map.occupancy.world_to_cell(cue.position[:2]) is None):
                continue
            nearby = [track for track in self._cue_tracks if _distance_xy(track.position, cue.position) <= 0.75]
            track = min(nearby, key=lambda item: _distance_xy(item.position, cue.position), default=None)
            if track is None:
                self._next_cue_id += 1
                self._cue_tracks.append(_CueTrack(
                    f'cue:{self._next_cue_id}', tuple(cue.position), cue.confidence,
                    cue.step, classified_label=cue.classified_label,
                ))
            elif cue.step > track.last_step:
                track.position = tuple(float(value) for value in (
                    np.asarray(track.position) * 0.75 + np.asarray(cue.position) * 0.25
                ))
                track.confidence = cue.confidence
                track.last_step = cue.step
                track.classified_label = cue.classified_label
                # Cooldown observations cannot preload a new navigation lock.
                if cue.step >= track.cooldown_until:
                    track.observations += 1

    def _finish_target_cue(self, step, reason, cooldown=True):
        cue = self.target_cue
        if cue is None:
            return
        self.cue_history.append({'step': step, 'event': 'released', 'cue_id': cue.cue_id, 'reason': reason})
        if cooldown:
            cue.cooldown_until = step + self.config.target_cue_cooldown_steps
            cue.observations = 0
        self.target_cue = None
        self._cue_started_step = self._cue_observe_step = self._cue_goal = None
        self.current_goal = None
        self.current_frontier_id = None

    def _update_target_cue(self, step, position):
        if self.target_cue is not None:
            reason = None
            if step - self.target_cue.last_step > self.config.target_cue_stale_steps:
                reason = 'stale'
            elif self._cue_observe_step is not None:
                if step - self._cue_observe_step >= self.config.target_cue_observe_steps:
                    reason = 'unconfirmed_after_observation'
            elif step - self._cue_started_step >= self.config.target_cue_navigation_steps:
                reason = 'navigation_timeout'
            if reason is not None:
                self._finish_target_cue(step, reason)
                # Give exploration a turn before considering another candidate.
                return False
        if self.target_cue is None:
            candidates = [track for track in self._cue_tracks
                          if track.observations >= self.config.target_min_observations
                          and step >= track.cooldown_until
                          and step - track.last_step <= self.config.target_cue_stale_steps]
            if not candidates:
                return False
            cue = max(candidates, key=lambda item: (
                item.observations, item.confidence, -_distance_xy(position, item.position),
            ))
            goal = self._approach_position(cue.position, position)
            occupancy = self.mapping.map.occupancy
            cell = occupancy.world_to_cell(goal[:2])
            if cell is None or not occupancy.observed[cell] or occupancy.inflated_mask()[cell]:
                cue.cooldown_until = step + self.config.target_cue_cooldown_steps
                cue.observations = 0
                return False
            self.target_cue = cue
            self._cue_started_step, self._cue_goal = step, goal
            self.current_frontier_id = None
            self.cue_history.append({
                'step': step, 'event': 'selected', 'cue_id': cue.cue_id,
                'position': list(cue.position), 'classified_label': cue.classified_label,
            })
        self.current_goal = self._cue_goal
        if (self._cue_observe_step is None
                and _distance_xy(position, self._cue_goal) <= self.config.target_distance):
            self._cue_observe_step = step
            self.cue_history.append({'step': step, 'event': 'observing', 'cue_id': self.target_cue.cue_id})
        return True

    def _sync_runtime_state(self):
        self.mapping.exploration_state = self.state
        self.mapping.exploration_target_query = self.config.target_query
        self.mapping.exploration_goal = self.current_goal
        self.mapping.exploration_decision = self.last_decision


def _position(robot_observation: dict) -> Tuple[float, float, float]:
    position = np.asarray(robot_observation['position'], dtype=np.float64).reshape(-1)
    if len(position) < 3:
        raise ValueError('robot position must contain three coordinates')
    return tuple(float(value) for value in position[:3])


def _distance_xy(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left[:2], dtype=np.float64) - np.asarray(right[:2], dtype=np.float64)))


def _normalize_label(value: str) -> str:
    return ' '.join(str(value).casefold().replace('_', ' ').replace('-', ' ').split())


__all__ = [
    'SemanticExplorationComponent',
    'SemanticExplorationConfig',
    'SemanticExplorationResult',
    'SemanticExplorationStatus',
]
