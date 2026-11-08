"""Constrained placement sampling and robot-footprint clearance checks."""

import itertools

import numpy as np

from grutopia_extension.interaction_clearance.task import evaluate_clearance, yaw_rotation
from grutopia_extension.interaction_clearance.motion_geometry import angle_error


def carrying_base_pose(start, object_position, destination, yaw_delta=0.0):
    start = np.asarray(start, dtype=float)
    object_position = np.asarray(object_position, dtype=float)
    destination = np.asarray(destination, dtype=float)
    if (start.shape != (3,) or object_position.shape not in ((2,), (3,)) or destination.shape != (2,)
            or not np.isfinite(np.r_[start, object_position, destination, yaw_delta]).all()):
        raise ValueError('Carrying target requires finite base pose, object position, destination and yaw')
    offset = yaw_rotation(yaw_delta)[:2, :2] @ (object_position[:2] - start[:2])
    return np.r_[destination - offset, angle_error(start[2] + yaw_delta, 0)]


def placement_candidates(task, rotation, margin=0.025):
    half_extent = np.abs(np.asarray(rotation)[:2, :2]) @ (np.asarray(task.dimensions[:2]) / 2)
    lower = np.asarray(task.placement.lower) + half_extent + margin
    upper = np.asarray(task.placement.upper) - half_extent - margin
    if np.any(lower > upper):
        return []
    preferred = np.clip(np.asarray(task.place_xy), lower, upper)
    coordinates = [np.unique(np.r_[preferred[axis], np.linspace(lower[axis], upper[axis], 5)]) for axis in range(2)]
    candidates = []
    for position in itertools.product(*coordinates):
        metrics = evaluate_clearance(task, (*position, 0.0001), rotation, 0, 0, 0, True, 0)
        if metrics['physically_complete']:
            candidates.append(np.asarray(position))
    return sorted(candidates, key=lambda position: float(np.linalg.norm(position - preferred)))


def robot_passage_metrics(task, model, transforms):
    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    for name, vertices in model.vertices.items():
        transform = transforms[name]
        world = np.asarray(vertices) @ transform[:3, :3].T + transform[:3, 3]
        lower = np.minimum(lower, world[:, :2].min(axis=0))
        upper = np.maximum(upper, world[:, :2].max(axis=0))
    separation = max(lower[0] - task.passage.upper[0], task.passage.lower[0] - upper[0],
                     lower[1] - task.passage.upper[1], task.passage.lower[1] - upper[1])
    return {'robot_clear_of_passage': bool(separation >= task.required_clearance_m),
            'robot_passage_clearance_m': float(separation)}
