"""Validate and normalize sensor packets before existing navigation sees them."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .localization import quaternion


@dataclass
class Snapshot:
    observation: dict
    captured_at: float
    pose_at: float
    lidar_at: float
    camera_at: float | None
    velocity: tuple
    sequence: int
    geometry_calibrated: bool = True
    planning_frame: str = "unknown"

    def require_fresh(self, now, max_age, camera=False):
        times = [('odom', self.pose_at), ('lidar', self.lidar_at)]
        if camera:
            if self.camera_at is None:
                raise ValueError('RGB-D observation is missing')
            times.append(('camera', self.camera_at))
        for name, stamp in times:
            age = now-stamp
            if not np.isfinite(age):
                raise ValueError(f'{name} observation has an invalid timestamp')
            if age > max_age:
                raise ValueError(f'{name} observation stale: age={age*1000:.1f} ms, limit={max_age*1000:.0f} ms')
            if age < -.05:
                raise ValueError(f'{name} observation future: ahead={-age*1000:.1f} ms, limit=50 ms')


def normalize(observation, ground_z=0.0):
    """Return a copy; ground-relative Z preserves existing flat-floor checks."""
    position = np.asarray(observation['position'], dtype=float).copy()
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Invalid robot position')
    position[2] -= ground_z
    result = {'position': position, 'orientation': quaternion(observation['orientation']),
              'sensors': {}}
    lidar = dict(observation['sensors']['lidar'])
    points = np.asarray(lidar['pointcloud'], dtype=np.float32).copy()
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('LiDAR requires nonempty finite N x 3 map-frame points')
    points[:, 2] -= ground_z
    origin = np.asarray(lidar['position'], dtype=float).copy()
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Invalid LiDAR origin')
    origin[2] -= ground_z
    result['sensors']['lidar'] = {**lidar, 'pointcloud': points, 'position': origin}
    camera = observation['sensors'].get('camera')
    if camera is not None:
        camera = dict(camera)
        depth = np.asarray(camera['depth'], dtype=np.float32).copy()
        rgba = np.asarray(camera['rgba'])
        xyz = np.asarray(camera['pointcloud'], dtype=np.float32).copy()
        if (depth.ndim != 2 or rgba.shape[:2] != depth.shape or rgba.ndim != 3
                or rgba.shape[2] not in (3, 4) or xyz.shape != (depth.size, 3)):
            raise ValueError('RGB-D and dense XYZ must align pixel for pixel')
        # Preserve invalid pixels as NaNs, never compact XYZ independently of RGB.
        valid = np.isfinite(depth) & (depth > .01) & np.isfinite(xyz.reshape(*depth.shape, 3)).all(axis=2)
        depth[~valid] = np.nan
        xyz[~valid.ravel()] = np.nan
        xyz[:, 2] -= ground_z
        origin = np.asarray(camera['position'], dtype=float).copy()
        origin[2] -= ground_z
        # Only whitelisted fields survive; no simulator ground-truth labels.
        result['sensors']['camera'] = {
            'rgba': rgba.copy(), 'depth': depth, 'pointcloud': xyz, 'position': origin,
            'orientation': quaternion(camera['orientation']),
            'horizontal_fov_degrees': float(camera['horizontal_fov_degrees']),
            'capture_step': int(camera['capture_step']),
        }
    return result
