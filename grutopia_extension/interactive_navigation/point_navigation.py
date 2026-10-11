"""Reusable point-navigation orchestration built on the fused mapping stack."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, Vector3
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime


class PointNavigationStatus(str, Enum):
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


@dataclass(frozen=True)
class PointNavigationConfig:
    """Environment-independent navigation and acceptance settings."""

    goal: Vector3
    mapping: MappingConfig = field(default_factory=MappingConfig)
    max_steps: int = 8000
    success_distance: float = 0.30
    fall_height: float = 0.35
    safe_base_height: float = 0.65
    lidar_interval: int = 8
    rgb_interval: int = 24
    replan_interval_steps: Optional[int] = 120
    use_scene_graph: bool = True
    use_semantic_occupancy: bool = False
    use_rgb_occupancy: bool = False
    rgb_clears_free_space: bool = False
    replan_only_if_blocked: bool = True
    replan_lookahead_distance: float = 2.0
    prefer_voronoi_paths: bool = False
    use_planner: bool = True
    velocity_control: bool = False
    max_forward_speed: float = 0.75
    max_lateral_speed: float = 0.35

    def __post_init__(self):
        if len(self.goal) != 3:
            raise ValueError('goal must contain three coordinates')
        if self.max_steps <= 0:
            raise ValueError('max_steps must be positive')
        if self.success_distance <= 0:
            raise ValueError('success_distance must be positive')
        if self.lidar_interval <= 0 or self.rgb_interval <= 0:
            raise ValueError('sensor intervals must be positive')
        if self.replan_interval_steps is not None and self.replan_interval_steps <= 0:
            raise ValueError('replan_interval_steps must be positive when provided')
        if self.replan_lookahead_distance <= 0:
            raise ValueError('replan_lookahead_distance must be positive')


@dataclass(frozen=True)
class PointNavigationResult:
    status: PointNavigationStatus
    step: int
    position: Vector3
    goal: Vector3
    distance: float
    failure_reason: Optional[str]
    statistics: dict

    @property
    def terminal(self) -> bool:
        return self.status != PointNavigationStatus.RUNNING

    @property
    def success(self) -> bool:
        return self.status == PointNavigationStatus.SUCCEEDED

    def as_event(self) -> dict:
        payload = {
            'event': 'navigation_result',
            'success': self.success,
            'step': self.step,
            'position': list(self.position),
            'goal': list(self.goal),
            'distance': self.distance,
            'mapping': self.statistics,
        }
        if self.failure_reason is not None:
            payload['failure_reason'] = self.failure_reason
        return payload


class PointNavigationComponent:
    """Fuse observations, maintain plans, issue actions, and evaluate progress."""

    def __init__(self, config: PointNavigationConfig):
        self.config = config
        self.mapping = MapNavigationRuntime(
            mapping_config=config.mapping,
            lidar_interval=config.lidar_interval,
            rgb_interval=config.rgb_interval,
            use_planner=config.use_planner,
            use_semantic_voronoi=config.prefer_voronoi_paths,
            prefer_voronoi_paths=config.prefer_voronoi_paths,
            safe_path_tracking=config.prefer_voronoi_paths,
            use_scene_graph=config.use_scene_graph,
            safe_base_height=config.safe_base_height,
            replan_interval_steps=config.replan_interval_steps,
            allow_goal_door_traversal=False,
            use_semantic_occupancy=config.use_semantic_occupancy,
            use_rgb_occupancy=config.use_rgb_occupancy,
            rgb_clears_free_space=config.rgb_clears_free_space,
            replan_only_if_blocked=config.replan_only_if_blocked,
            replan_lookahead_distance=config.replan_lookahead_distance,
        )
        self._trajectory = []
        self._minimum_clearance = None
        self._static_obstacle_boxes = []
        self._static_obstacle_violations = 0
        self._violated_static_labels = set()

    @property
    def goal(self) -> Vector3:
        return tuple(float(value) for value in self.config.goal)

    @property
    def trajectory(self) -> Tuple[Vector3, ...]:
        return tuple(self._trajectory)

    @staticmethod
    def warmup_action() -> dict:
        return {'move_by_speed': [0.0, 0.0, 0.0]}

    def update(self, step: int, robot_observation: dict):
        self.mapping.update(step, robot_observation)
        position = _position(robot_observation)
        self._trajectory.append(position)
        self._update_obstacle_clearance(position)
        self._update_static_obstacle_violations(position)

    def seed_static_obstacles(self, obstacles: Iterable[dict]):
        for obstacle in obstacles:
            label = str(obstacle['label'])
            minimum = np.asarray(obstacle['minimum_xy'], dtype=np.float64)
            maximum = np.asarray(obstacle['maximum_xy'], dtype=np.float64)
            self.mapping.map.occupancy.mark_box_occupied(minimum, maximum)
            self._static_obstacle_boxes.append((label, minimum, maximum))

    def action(self, step: int, robot_observation: dict) -> dict:
        return self.mapping.point_navigation_action(
            self.goal,
            robot_observation,
            step=step,
            velocity_control=self.config.velocity_control,
            max_forward_speed=self.config.max_forward_speed,
            max_lateral_speed=self.config.max_lateral_speed,
        )

    def evaluate(self, step: int, robot_observation: dict) -> PointNavigationResult:
        position = _position(robot_observation)
        distance = float(np.linalg.norm(np.asarray(position[:2]) - np.asarray(self.goal[:2])))
        status = PointNavigationStatus.RUNNING
        failure_reason = None
        if distance <= self.config.success_distance and not self.mapping.buffer_recovery_active:
            status = PointNavigationStatus.SUCCEEDED
        elif position[2] < self.config.fall_height:
            status = PointNavigationStatus.FAILED
            failure_reason = 'robot_fell'
        elif self.mapping.buffer_recovery_failed:
            status = PointNavigationStatus.FAILED
            failure_reason = 'buffer_exit_timeout'
        elif step + 1 >= self.config.max_steps:
            status = PointNavigationStatus.FAILED
            failure_reason = 'global_step_limit'
        statistics = self.statistics() if status != PointNavigationStatus.RUNNING else {}
        return PointNavigationResult(
            status=status,
            step=step,
            position=position,
            goal=self.goal,
            distance=distance,
            failure_reason=failure_reason,
            statistics=statistics,
        )

    def progress_event(self, step: int, robot_observation: dict) -> dict:
        result = self.evaluate(step, robot_observation)
        return {
            'event': 'navigation_progress',
            'step': step,
            'position': list(result.position),
            'goal': list(result.goal),
            'distance': result.distance,
            'mapping': self.statistics(),
        }

    def statistics(self) -> dict:
        stats = dict(self.mapping.statistics())
        stats['trajectory_points'] = len(self._trajectory)
        stats['trajectory_length'] = self._trajectory_length()
        stats['minimum_mapped_obstacle_clearance'] = self._minimum_obstacle_clearance()
        stats['static_obstacle_boxes'] = len(self._static_obstacle_boxes)
        stats['trajectory_static_obstacle_violations'] = self._static_obstacle_violations
        stats['violated_static_obstacle_labels'] = sorted(self._violated_static_labels)
        return _json_statistics(stats)

    def save(self, output_prefix: str):
        self.mapping.save(output_prefix)
        if not output_prefix:
            return
        metadata_path = Path(str(output_prefix) + '.json')
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            payload = json.load(metadata_file)
        payload['point_navigation'] = {
            'goal': list(self.goal),
            'statistics': self.statistics(),
            'trajectory': [list(point) for point in self._trajectory],
        }
        with metadata_path.open('w', encoding='utf-8') as metadata_file:
            json.dump(payload, metadata_file, indent=2)

    def _trajectory_length(self) -> float:
        if len(self._trajectory) < 2:
            return 0.0
        points = np.asarray(self._trajectory, dtype=np.float64)
        return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())

    def _minimum_obstacle_clearance(self) -> Optional[float]:
        return self._minimum_clearance

    def _update_obstacle_clearance(self, position: Vector3):
        occupancy = self.mapping.map.occupancy
        occupied_cells = np.argwhere(occupancy.occupied_mask())
        if len(occupied_cells) == 0:
            return
        occupied_xy = np.asarray(
            [occupancy.cell_to_world(tuple(int(value) for value in cell)) for cell in occupied_cells],
            dtype=np.float64,
        )
        minimum = float(np.linalg.norm(occupied_xy - np.asarray(position[:2]), axis=1).min())
        if np.isfinite(minimum):
            self._minimum_clearance = minimum if self._minimum_clearance is None else min(
                self._minimum_clearance,
                minimum,
            )

    def _update_static_obstacle_violations(self, position: Vector3):
        xy = np.asarray(position[:2], dtype=np.float64)
        padding = self.config.mapping.robot_radius
        violated = False
        for label, minimum, maximum in self._static_obstacle_boxes:
            if np.all(xy >= minimum - padding) and np.all(xy <= maximum + padding):
                self._violated_static_labels.add(label)
                violated = True
        if violated:
            self._static_obstacle_violations += 1


def _position(robot_observation: dict) -> Vector3:
    values = np.asarray(robot_observation['position'], dtype=np.float64).reshape(-1)
    if len(values) < 3:
        raise ValueError('robot observation position must contain three coordinates')
    return tuple(float(value) for value in values[:3])


def _json_statistics(value):
    if isinstance(value, dict):
        return {str(key): _json_statistics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_statistics(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
