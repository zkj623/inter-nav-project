"""Identify robot self-returns without hiding nearby scene obstacles."""

import numpy as np


def exclude_robot_returns(points, hit_paths, robot_path):
    points = np.asarray(points).reshape(-1, 3)
    paths = np.asarray(hit_paths, dtype=str).reshape(-1)
    if len(points) != len(paths):
        raise ValueError('LiDAR point and hit-path counts differ')
    root = robot_path.rstrip('/')
    own = (paths == root) | np.char.startswith(paths, root + '/')
    return points[~own], int(own.sum())
