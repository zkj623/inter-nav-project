"""Upstream task contracts and independent geometric clearance evaluation."""

from dataclasses import asdict, dataclass, field
import itertools
import math

import numpy as np


@dataclass(frozen=True)
class Region:
    lower: tuple
    upper: tuple

    def validate(self):
        if len(self.lower) != 2 or len(self.upper) != 2:
            raise ValueError('Regions require lower and upper XY coordinates')
        values = np.asarray((self.lower, self.upper), dtype=float)
        if not np.isfinite(values).all() or np.any(values[0] >= values[1]):
            raise ValueError('Region bounds must be finite and strictly ordered')


@dataclass(frozen=True)
class SceneObstacle:
    name: str
    position: tuple
    dimensions: tuple

    def validate(self):
        if not self.name.isidentifier() or self.name.startswith('_'):
            raise ValueError('Obstacle names must be public identifiers')
        if len(self.position) != 3 or len(self.dimensions) != 3:
            raise ValueError('Scene obstacles require three position and size coordinates')
        if not np.isfinite((*self.position, *self.dimensions)).all() or min(self.dimensions) <= 0:
            raise ValueError('Scene obstacle geometry must be finite and positive')


def default_scene_obstacles():
    return (SceneObstacle('NorthWall', (1.90, 0.55, 0.40), (1.55, 0.10, 0.80)),
            SceneObstacle('SouthWall', (1.90, -0.55, 0.40), (1.55, 0.10, 0.80)))


@dataclass(frozen=True)
class ClearanceTask:
    task_id: str = 'box_clearance_v0'
    object_id: str = 'box_0'
    object_xy: tuple = (0.65, 0.0)
    dimensions: tuple = (0.40, 0.50, 0.28)
    mass_kg: float = 1.5
    place_xy: tuple = (0.65, 0.70)
    passage: Region = field(default_factory=lambda: Region((0.45, -0.35), (1.8, 0.35)))
    placement: Region = field(default_factory=lambda: Region((0.35, 0.40), (0.98, 1.05)))
    required_clearance_m: float = 0.03
    required_stable_seconds: float = 1.0
    seed: int = 0
    scene_obstacles: tuple = field(default_factory=default_scene_obstacles)
    robot_xy: tuple = (0.0, 0.0)
    robot_yaw_rad: float = 0.0
    object_yaw_rad: float = 0.0
    grasp_distance_m: float = 0.65
    return_to_start: bool = False

    def validate(self):
        self.passage.validate()
        self.placement.validate()
        if not isinstance(self.return_to_start, bool):
            raise ValueError('return_to_start must be a boolean')
        values = (*self.object_xy, *self.place_xy, *self.dimensions, self.mass_kg,
                  self.required_clearance_m, self.required_stable_seconds, *self.robot_xy,
                  self.robot_yaw_rad, self.object_yaw_rad, self.grasp_distance_m)
        if len(self.object_xy) != 2 or len(self.place_xy) != 2 or len(self.dimensions) != 3 or len(self.robot_xy) != 2:
            raise ValueError('Task requires XY targets and three object dimensions')
        if not np.isfinite(values).all() or min(self.dimensions) <= 0 or self.mass_kg <= 0:
            raise ValueError('Task geometry and mass must be finite and positive')
        if self.required_clearance_m < 0.03 or self.required_stable_seconds < 1.0:
            raise ValueError('Clearance and stability thresholds cannot be weakened below v0 limits')
        if abs(self.robot_yaw_rad) > math.pi or abs(self.object_yaw_rad) > math.pi or not 0.58 <= self.grasp_distance_m <= 0.72:
            raise ValueError('Yaw must be within [-pi, pi] and grasp distance within 0.58..0.72 m')
        object_half = np.abs(yaw_rotation(self.object_yaw_rad)[:2, :2]) @ (np.asarray(self.dimensions[:2]) / 2)
        if not self.scene_obstacles or len(self.scene_obstacles) > 12:
            raise ValueError('A clearance scene requires between 1 and 12 obstacles')
        names = set()
        for obstacle in self.scene_obstacles:
            obstacle.validate()
            if obstacle.name in names:
                raise ValueError('Scene obstacle names must be unique')
            names.add(obstacle.name)
            half = np.asarray(obstacle.dimensions) / 2
            lower = np.asarray(obstacle.position) - half
            upper = np.asarray(obstacle.position) + half
            if lower[2] < self.dimensions[2] and upper[2] > 0:
                if np.all(lower[:2] < self.passage.upper) and np.all(upper[:2] > self.passage.lower):
                    raise ValueError('A static obstacle cannot occupy the required passage')
                for center in (self.object_xy, self.place_xy):
                    extent = object_half + 0.005
                    if np.all(lower[:2] < np.asarray(center) + extent) and np.all(upper[:2] > np.asarray(center) - extent):
                        raise ValueError('A static obstacle intersects the initial or target box footprint')
        half_extent = object_half
        destination = np.asarray(self.place_xy)
        if np.any(destination - half_extent < self.placement.lower) or np.any(destination + half_extent > self.placement.upper):
            raise ValueError('The target object footprint does not fit the allowed placement region')
        initial = evaluate_clearance(self, (*self.object_xy, 0.0), yaw_rotation(self.object_yaw_rad), 0, 0, 0, True, 0)
        if initial['road_clear']:
            raise ValueError('Upstream clearance task must initially contain a blocking object')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        for name in ('passage', 'placement'):
            if name in values:
                values[name] = Region(tuple(values[name]['lower']), tuple(values[name]['upper']))
        for name in ('object_xy', 'place_xy', 'dimensions', 'robot_xy'):
            if name in values:
                values[name] = tuple(values[name])
        if 'scene_obstacles' in values:
            values['scene_obstacles'] = tuple(SceneObstacle(item['name'], tuple(item['position']), tuple(item['dimensions']))
                                              for item in values['scene_obstacles'])
        task = cls(**values)
        task.validate()
        return task


def yaw_rotation(yaw):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def evaluate_clearance(task, position, rotation, linear_speed, angular_speed, contact_force,
                       safety_ok, stable_seconds):
    length, width, height = task.dimensions
    corners = np.array(list(itertools.product((-length / 2, length / 2),
                                              (-width / 2, width / 2), (0.0, height))))
    world = corners @ np.asarray(rotation).T + np.asarray(position)
    lower = world.min(axis=0)
    upper = world.max(axis=0)
    separation = max(lower[0] - task.passage.upper[0], task.passage.lower[0] - upper[0],
                     lower[1] - task.passage.upper[1], task.passage.lower[1] - upper[1])
    road_clear = bool(separation >= task.required_clearance_m)
    in_placement = bool(np.all(lower[:2] >= np.asarray(task.placement.lower) - 1e-6)
                        and np.all(upper[:2] <= np.asarray(task.placement.upper) + 1e-6))
    supported = bool(-0.002 <= lower[2] <= 0.015)
    stationary = bool(linear_speed <= 0.015 and angular_speed <= 0.08)
    released = bool(contact_force < 1.0)
    physically_complete = road_clear and in_placement and supported and stationary and released and bool(safety_ok)
    return {'success': bool(physically_complete and stable_seconds >= task.required_stable_seconds),
            'physically_complete': bool(physically_complete), 'road_clear': road_clear,
            'passage_clearance_m': float(separation), 'in_allowed_placement': in_placement,
            'object_supported': supported, 'object_stationary': stationary, 'hands_released': released,
            'object_bottom_m': float(lower[2]), 'object_footprint_lower': lower[:2].tolist(),
            'object_footprint_upper': upper[:2].tolist(), 'stable_seconds': float(stable_seconds),
            'safety_ok': bool(safety_ok)}
