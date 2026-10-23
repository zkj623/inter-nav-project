"""Geometry shared by ROS adapters and offline tests; quaternions are wxyz."""
from collections import deque
import numpy as np


def quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    return q / np.linalg.norm(q)


def rotation(value):
    w, x, y, z = quaternion(value)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform(points, position, orientation):
    return np.asarray(points) @ rotation(orientation).T + np.asarray(position)


def camera_cloud(depth, intrinsics, position, orientation):
    """Rectified, RGB-aligned optical depth in metres to dense map-frame XYZ."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Depth must be H x W')
    k = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    v, u = np.indices(depth.shape)
    points = np.stack(((u-k[0, 2])*depth/k[0, 0],
                       (v-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    points = transform(points, position, orientation)
    points[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return points.reshape(-1, 3).astype(np.float32)


class PoseContinuity:
    """Latch jumps: a fixed fused map cannot absorb SLAM relocalization silently."""
    def __init__(self, distance, angle):
        self.distance, self.angle = distance, angle
        self.previous = None
        self.fault = None

    def check(self, position, orientation):
        p, q = np.asarray(position), quaternion(orientation)
        if self.previous is not None:
            old_p, old_q = self.previous
            turn = 2*np.arccos(np.clip(abs(np.dot(q, old_q)), 0, 1))
            if np.linalg.norm(p-old_p) > self.distance or turn > self.angle:
                self.fault = 'Localization jump: stop and restart with a new map'
        self.previous = p.copy(), q.copy()
        if self.fault:
            raise ValueError(self.fault)


def multiply(left, right):
    w, x, y, z = quaternion(left)
    a, b, c, d = quaternion(right)
    return quaternion((w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                       w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a))


class PoseHistory:
    """Time-align a range origin with world-frame points, never with world origin.

    Interpolate only nearby samples; allow nearest sample within the same tolerance
    at the stream edge. No unbounded extrapolation during odometry outages.
    """
    def __init__(self, tolerance_s):
        self.tolerance_ns = tolerance_s * 1e9
        self.samples = deque(maxlen=512)

    def add(self, stamp, position, orientation):
        if self.samples and stamp <= self.samples[-1][0]:
            raise ValueError('Non-increasing odometry timestamp')
        self.samples.append((stamp, np.asarray(position).copy(), quaternion(orientation)))

    def at(self, stamp):
        if not self.samples:
            raise ValueError('Waiting for odometry at point-cloud time')
        nearest = min(self.samples, key=lambda sample: abs(sample[0]-stamp))
        if abs(nearest[0]-stamp) > self.tolerance_ns:
            raise ValueError('Point-cloud / odometry timestamp mismatch')
        for left, right in zip(self.samples, list(self.samples)[1:]):
            if left[0] <= stamp <= right[0] and right[0]-left[0] <= 2*self.tolerance_ns:
                alpha = (stamp-left[0])/(right[0]-left[0])
                q0, q1 = left[2], right[2]
                dot = np.dot(q0, q1)
                if dot < 0:
                    q1, dot = -q1, -dot
                theta = np.arccos(np.clip(dot, 0, 1))
                q = ((1-alpha)*q0+alpha*q1 if theta < 1e-5 else
                     (np.sin((1-alpha)*theta)*q0+np.sin(alpha*theta)*q1)/np.sin(theta))
                return (1-alpha)*left[1]+alpha*right[1], quaternion(q)
        return nearest[1].copy(), nearest[2].copy()
