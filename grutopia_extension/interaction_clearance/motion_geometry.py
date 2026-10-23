"""Robot/object frame geometry shared by physical primitives and regression tests."""

import math

import numpy as np

from grutopia_extension.interaction_clearance.task import yaw_rotation


def angle_error(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def quaternion_yaw(quaternion):
    scalar, first, second, third = np.asarray(quaternion, dtype=float)
    return math.atan2(2 * (scalar * third + first * second), 1 - 2 * (second ** 2 + third ** 2))


def planar_transform(pose, height=0.0):
    transform = np.eye(4)
    transform[:3, :3] = yaw_rotation(pose[2])
    transform[:3, 3] = (*pose[:2], height)
    return transform


def grasp_base_pose(object_position, object_yaw, distance):
    offset = yaw_rotation(object_yaw) @ np.array((distance, 0.0, 0.0))
    return np.r_[np.asarray(object_position)[:2] - offset[:2], object_yaw]


def grasp_frames(center, yaw, half_width, height):
    rotation = yaw_rotation(yaw)
    roll = np.diag((1.0, -1.0, -1.0))
    left = np.asarray(center) + rotation @ np.array((0.0, half_width, height))
    right = np.asarray(center) + rotation @ np.array((0.0, -half_width, height))
    return left, right, rotation @ yaw_rotation(-math.pi / 2) @ roll, rotation @ yaw_rotation(math.pi / 2) @ roll


def opposed_object_contact(forces, object_yaw, minimum_force=0.5):
    forces = np.asarray(forces, dtype=float)
    if forces.shape != (2, 3) or not np.isfinite(forces).all():
        return False
    local = forces @ yaw_rotation(object_yaw)
    return bool(local[0, 1] > minimum_force and local[1, 1] < -minimum_force)


def bounded_swerve_targets(command, current_angles, module_positions, wheel_radius=0.07, max_wheel_speed=2.0):
    command = np.asarray(command, dtype=float)
    current = np.asarray(current_angles, dtype=float)
    positions = np.asarray(module_positions, dtype=float)
    if (command.shape != (3,) or current.shape != (3,) or positions.shape != (3, 2)
            or not np.isfinite(np.r_[command, current, positions.ravel()]).all()):
        raise ValueError('Swerve inputs must be finite twist, three steering angles and three module positions')
    if not np.isfinite((wheel_radius, max_wheel_speed)).all() or wheel_radius <= 0 or max_wheel_speed <= 0:
        raise ValueError('Wheel radius and speed limit must be positive and finite')
    if np.any(np.abs(current) > 2 * math.pi):
        raise ValueError('Measured steering state exceeds the PhysX revolute-joint range')
    vectors = np.column_stack((command[0] - command[2] * positions[:, 1],
                               command[1] + command[2] * positions[:, 0]))
    speed = np.linalg.norm(vectors, axis=1)
    desired = np.arctan2(vectors[:, 1], vectors[:, 0])
    shifts = np.arange(-2, 3)
    candidates = desired[:, None] + shifts * math.pi
    limit = 2 * math.pi - 0.05
    costs = np.where(np.abs(candidates) <= limit, np.abs(candidates - current[:, None]), np.inf)
    selected = np.argmin(costs, axis=1)
    targets = candidates[np.arange(3), selected]
    targets = np.where((speed < 1e-6) & (np.abs(current) <= limit), current, targets)
    directions = np.where(shifts[selected] % 2 == 0, 1.0, -1.0)
    velocities = directions * speed / wheel_radius * np.maximum(np.cos(targets - current), 0.0)
    velocities *= min(1.0, max_wheel_speed / max(float(np.max(np.abs(velocities))), 1e-9))
    return targets, velocities


def base_waypoints(start, target, route):
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if route not in ('direct', 'retreat', 'x_first', 'y_first'):
        raise ValueError('Unknown base route')
    points = [start.copy()]
    pivot = start.copy()
    if route == 'retreat':
        pivot[:2] -= (yaw_rotation(start[2]) @ np.array((0.20, 0.0, 0.0)))[:2]
        points.append(pivot.copy())
    aligned = np.r_[pivot[:2], pivot[2] + angle_error(target[2], pivot[2])]
    points.append(aligned)
    if route == 'x_first':
        points.append(np.array((target[0], aligned[1], aligned[2])))
    if route == 'y_first':
        points.append(np.array((aligned[0], target[1], aligned[2])))
    points.append(np.r_[target[:2], aligned[2]])
    return points


def sample_base_path(points, spacing=0.01, angular_spacing=math.radians(1)):
    samples = []
    for start, target in zip(points, points[1:]):
        steps = max(1, math.ceil(np.linalg.norm(target[:2] - start[:2]) / spacing),
                    math.ceil(abs(angle_error(target[2], start[2])) / angular_spacing))
        for progress in np.linspace(0.0, 1.0, steps + 1):
            samples.append(start + progress * (target - start))
    return samples
