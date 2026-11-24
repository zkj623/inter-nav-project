"""Reproject an already deskewed Go2 scan using the MID-360S trajectory."""
from collections import deque

import numpy as np

from .localization import PoseHistory, multiply, rotation


def odometry_pose(message):
    pose = message.pose.pose
    p, q = pose.position, pose.orientation
    position = np.array([p.x, p.y, p.z])
    if not np.isfinite(position).all():
        raise ValueError('Invalid odometry position')
    return position, (q.w, q.x, q.y, q.z)


class MixedCloud:
    def __init__(self, config):
        self.unitree = PoseHistory(config['pose_cloud_tolerance_s'])
        self.lio = PoseHistory(config['pose_cloud_tolerance_s'])
        self.pending = deque(maxlen=8)
        self.ready = None
        mount = config['mid360_mount']
        pitch = np.deg2rad(mount['pitch_deg'])
        self.body_from_imu = rotation([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
        self.imu_from_body_q = (np.cos(pitch/2), 0., -np.sin(pitch/2), 0.)
        self.imu_in_body = np.array(mount['imu_in_base_m'])
        self.max_range = config['lidar_max_range']

    def project(self, unitree_stamp, lio_stamp, points):
        old_p, old_q = self.unitree.at(unitree_stamp, bracketed=True)
        state, imu_q = self.lio.at(lio_stamp, bracketed=True)
        body_q = multiply(imu_q, self.imu_from_body_q)
        world_from_body = rotation(body_q)
        body_p = state[:3] - world_from_body @ self.imu_in_body
        body_points = (points - old_p) @ rotation(old_q)
        world_points = (body_points @ world_from_body.T + body_p).astype(np.float32)
        # Linear velocity at the IMU, in body axes. latest() shifts it to
        # the body origin with the synchronized gyro.
        linear = world_from_body.T @ state[3:]
        return body_p, body_q, linear, world_points

    def peek(self):
        if self.ready is None:
            self.ready = self._next_scan()
        return self.ready

    def take(self):
        scan = self.peek()
        self.ready = None
        return scan

    def _next_scan(self):
        if not self.unitree.samples or not self.lio.samples:
            return None
        while self.pending:
            raw_stamp, stamp, captured_at, points = self.pending[0]
            if self.lio.samples[-1][0] < stamp or self.unitree.samples[-1][0] < raw_stamp:
                return None
            self.pending.popleft()
            try:
                p, q, linear, xyz = self.project(raw_stamp, stamp, points)
            except ValueError:  # Old scan predates a buffer, or crosses an odometry gap.
                continue
            pose = (stamp, captured_at, p, q, (linear[0], linear[1], 0.))
            scan = (captured_at, dict(pointcloud=xyz, physics_step=stamp,
                                      source_stamp_ns=raw_stamp, max_range=self.max_range))
            return pose, scan, (linear, (0., 0., 0.))
        return None
