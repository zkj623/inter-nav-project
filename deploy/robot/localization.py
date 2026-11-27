"""Geometry shared by ROS adapters and offline tests; quaternions are wxyz."""
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


def registered_floor_height(points, position):
    """Estimate local floor Z once in a gravity-aligned world, near the start pose."""
    xyz = np.asarray(points) - position
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    xyz = xyz[(radius >= .65) & (radius <= 4.) & (xyz[:, 2] < -.08) & (xyz[:, 2] > -1.3)]
    if len(xyz) < 60:
        return None
    # Give each 25 cm square one vote. Repeated hits on one wall or leg
    # should not outvote a floor observed across several square metres.
    _, cells = np.unique(np.floor(xyz[:, :2] / .25), axis=0, return_inverse=True)
    groups = np.split(xyz[np.argsort(cells)], np.cumsum(np.bincount(cells))[:-1])
    lows = np.array([[*np.median(g[:, :2], axis=0), np.percentile(g[:, 2], 15)]
                     for g in groups if len(g) >= 3])
    if len(lows) < 16:
        return None
    levels = np.arange(-1.3, -.08, .02)
    votes = np.abs(lows[:, 2, None] - levels) < .04
    inliers = lows[votes[:, votes.sum(axis=0).argmax()]]
    if len(inliers) < 16 or np.linalg.svd(inliers[:, :2] - inliers[:, :2].mean(axis=0),
                                          compute_uv=False)[-1] < .45 * np.sqrt(len(inliers)):
        return None
    plane = np.column_stack((inliers[:, :2], np.ones(len(inliers))))
    slope_x, slope_y, intercept = np.linalg.lstsq(plane, inliers[:, 2], rcond=None)[0]
    if np.hypot(slope_x, slope_y) > .06:
        return None
    return float(position[2] + intercept)


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


def odometry_pose(message):
    pose = message.pose.pose
    p, q = pose.position, pose.orientation
    position = np.array([p.x, p.y, p.z])
    if not np.isfinite(position).all():
        raise ValueError('Invalid odometry position')
    return position, (q.w, q.x, q.y, q.z)


class LioMount:
    """Relate the MID-360S IMU reported by Swarm-LIO to the Go2 body."""
    def __init__(self, mount):
        pitch = np.deg2rad(mount['pitch_deg'])
        self.body_from_imu = rotation([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
        self.imu_from_body_q = (np.cos(pitch/2), 0., -np.sin(pitch/2), 0.)
        self.imu_in_body = np.array(mount['imu_in_base_m'])

    def body_state(self, imu_p, imu_q, world_linear):
        body_q = multiply(imu_q, self.imu_from_body_q)
        world_from_body = rotation(body_q)
        body_p = imu_p - world_from_body @ self.imu_in_body
        return body_p, body_q, world_from_body.T @ world_linear
