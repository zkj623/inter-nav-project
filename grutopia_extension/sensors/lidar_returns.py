"""Identify robot self-returns without hiding nearby scene obstacles."""

import numpy as np


def freeze_lidar_scan(points, hit_paths, position, orientation, physics_step, time):
    """Own one matched scan before either backend buffers or the robot advance.

    Point coordinates, hit identities and the sensor pose must refer to the
    same acquisition callback. Reading identities or pose later pairs cached
    leg points with newer floor hits while the robot is moving.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3).copy()
    paths = np.asarray(hit_paths, dtype=str).reshape(-1).copy()
    if len(points) != len(paths):
        raise ValueError('LiDAR point and hit-path counts differ')
    return {
        'point_cloud': points,
        'hit_paths': paths,
        'position': np.asarray(position, dtype=np.float32).reshape(3).copy(),
        'orientation': np.asarray(orientation, dtype=np.float32).reshape(4).copy(),
        'physics_step': int(physics_step),
        'time': float(time),
    }


def exclude_robot_returns(points, hit_paths, robot_path):
    points = np.asarray(points).reshape(-1, 3)
    paths = np.asarray(hit_paths, dtype=str).reshape(-1)
    if len(points) != len(paths):
        raise ValueError('LiDAR point and hit-path counts differ')
    root = robot_path.rstrip('/')
    own = (paths == root) | np.char.startswith(paths, root + '/')
    return points[~own], int(own.sum())
