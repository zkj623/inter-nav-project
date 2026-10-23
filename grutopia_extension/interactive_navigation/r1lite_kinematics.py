"""Three-module swerve inverse kinematics; x forward, y left, z up."""

import numpy as np

# Keep a small margin for conversion to the PhysX float drive target.
STEERING_TARGET_LIMIT = 2 * np.pi - 1e-5


def swerve_targets(command, current_angles, module_positions, wheel_radius=0.07, max_wheel_speed=12.0):
    c = np.asarray(command, dtype=float)
    current = np.asarray(current_angles, dtype=float)
    xy = np.asarray(module_positions, dtype=float)
    if (
        c.shape != (3,)
        or current.shape != (3,)
        or xy.shape != (3, 2)
        or not np.isfinite(np.concatenate((c, current, xy.ravel()))).all()
    ):
        raise ValueError('Swerve inputs must be finite vx/vy/yaw, three angles and three XY positions')
    if not np.isfinite([wheel_radius, max_wheel_speed]).all() or wheel_radius <= 0 or max_wheel_speed <= 0:
        raise ValueError('Wheel radius and limit must be finite and positive')
    vx, vy, w = c
    vectors = np.column_stack((vx - w * xy[:, 1], vy + w * xy[:, 0]))
    speed = np.linalg.norm(vectors, axis=1)
    desired = np.arctan2(vectors[:, 1], vectors[:, 0])
    # Both wheel headings produce the same chassis velocity when the rolling
    # direction is reversed. Choose the nearest *valid absolute* target among
    # their equivalent turns; current + wrapped_delta can exceed PhysX +/-2pi.
    opposite = (desired + 2 * np.pi) % (2 * np.pi) - np.pi
    headings = np.column_stack((desired, opposite))
    candidates = (headings[:, :, None] + np.array([-2 * np.pi, 0, 2 * np.pi])).reshape(3, 6)
    valid = np.abs(candidates) <= STEERING_TARGET_LIMIT
    distances = np.where(valid, np.abs(candidates - current[:, None]), np.inf)
    choices = np.argmin(distances, axis=1)
    target = candidates[np.arange(3), choices]
    reverse = choices >= 3
    target = np.where(speed < 1e-6, np.clip(current, -STEERING_TARGET_LIMIT, STEERING_TARGET_LIMIT), target)
    delta = target - current
    wheels = np.where(reverse, -speed, speed) / wheel_radius
    # Suppress rolling while a module turns, avoiding sideways tire scrub.
    # Near an absolute range boundary the shortest valid turn can exceed 90
    # degrees. Keep that wheel stationary until its heading catches up.
    wheels *= np.where(np.abs(delta) <= np.pi / 2, np.maximum(np.cos(delta), 0.0), 0.0)
    peak = np.max(np.abs(wheels))
    if peak > max_wheel_speed:
        wheels *= max_wheel_speed / peak
    return target, wheels
