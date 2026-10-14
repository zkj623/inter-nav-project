"""Camera surface memory and bounded, cost-aware look directions.

This is a 2D visibility heuristic, not semantic certainty or a complete 3D
inspection planner. Only actual RGB-D returns add observation evidence.
"""
from dataclasses import dataclass

import numpy as np


def angle_error(target, current):
    return (target - current + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class LookDirection:
    yaw: float
    gain: float


class VisualObservationMemory:
    def __init__(self, occupancy, max_range=4.0):
        self.occupancy = occupancy
        self.max_range = max_range
        # Eight approach directions distinguish opposite sides of a wall.
        self.seen = np.zeros(occupancy.observed.shape, dtype=np.uint8)
        self.last_frame = None
        self.camera_yaw = None
        self.camera_position = None
        self.fov = None
        self.frames = 0

    def _cells(self, points):
        cfg = self.occupancy.config
        col = np.floor((points[..., 0] - cfg.x_limits[0]) / cfg.grid_resolution).astype(int)
        row = np.floor((points[..., 1] - cfg.y_limits[0]) / cfg.grid_resolution).astype(int)
        valid = (row >= 0) & (col >= 0) & (row < self.seen.shape[0]) & (col < self.seen.shape[1])
        return row, col, valid

    @staticmethod
    def _sectors(angles):
        return np.floor((angles + np.pi / 8) / (np.pi / 4)).astype(int) % 8

    def observe(self, camera):
        frame = camera.get('rendering_frame')
        if frame is None:
            frame = camera.get('capture_step')
        if frame is None or frame == self.last_frame:
            return False
        orientation = np.asarray(camera.get('orientation', ()), dtype=float)
        origin = np.asarray(camera.get('position', ()), dtype=float)
        points = np.asarray(camera.get('pointcloud', ()), dtype=float)
        fov = float(camera.get('horizontal_fov_degrees', 0.0))
        if (orientation.shape != (4,) or origin.shape != (3,) or points.ndim != 2
                or points.shape[1:] != (3,) or not len(points)
                or not np.isfinite(orientation).all() or not np.isfinite(origin).all()
                or np.linalg.norm(orientation) < 1e-9 or not 20 < fov <= 180):
            return False
        w, x, y, z = orientation / np.linalg.norm(orientation)
        yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        # Dense endpoints already encode depth occlusion. Never fill an RGB
        # frustum using LiDAR or declare everything behind a hit observed.
        points = points[::max(1, len(points) // 12000)]
        delta = points[:, :2] - origin[:2]
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        cfg = self.occupancy.config
        valid = (np.isfinite(points).all(axis=1)
                 & (points[:, 2] >= cfg.obstacle_height[0])
                 & (points[:, 2] <= cfg.obstacle_height[1])
                 & (np.linalg.norm(delta, axis=1) <= self.max_range)
                 & (np.abs(angle_error(angles, yaw)) <= np.deg2rad(fov / 2 - 10)))
        points, angles = points[valid], angles[valid]
        row, col, valid = self._cells(points)
        sector = self._sectors(angles[valid])
        bits = ((1 << sector) | (1 << ((sector + 1) % 8)) | (1 << ((sector - 1) % 8))).astype(np.uint8)
        # One-cell tolerance covers RGB/LiDAR quantization, but opposite view
        # directions are still separate and no ray is extended through walls.
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = row[valid] + dr, col[valid] + dc
            inside = (rr >= 0) & (cc >= 0) & (rr < self.seen.shape[0]) & (cc < self.seen.shape[1])
            np.bitwise_or.at(self.seen, (rr[inside], cc[inside]), bits[inside])
        self.last_frame = frame
        self.camera_yaw, self.camera_position, self.fov = float(yaw), origin, fov
        self.frames += 1
        return True

    def best_direction(self, position, min_gain=0.30):
        if self.camera_yaw is None or np.linalg.norm(np.asarray(position[:2]) - self.camera_position[:2]) > .8:
            return None
        cfg = self.occupancy.config
        # Trace each ray only to its first occupied or unknown cell. Unknown
        # space belongs to the geometric frontier planner, not a fictitious
        # visible surface on the far side of an occluder.
        angles = np.arange(180) * (2 * np.pi / 180)
        distance = np.arange(.15, self.max_range, cfg.grid_resolution / 2)
        xy = self.camera_position[:2] + np.stack((np.cos(angles), np.sin(angles)), axis=-1)[:, None, :] * distance[None, :, None]
        row, col, valid = self._cells(xy)
        rr, cc = np.clip(row, 0, self.seen.shape[0]-1), np.clip(col, 0, self.seen.shape[1]-1)
        occupied = self.occupancy.occupied_mask()
        stop = ~valid | ~self.occupancy.observed[rr, cc] | occupied[rr, cc]
        first = np.argmax(stop, axis=1)
        ray = np.arange(len(angles))
        r, c = rr[ray, first], cc[ray, first]
        hits = stop.any(axis=1) & valid[ray, first] & occupied[r, c]
        cells = np.unique(np.column_stack((r[hits], c[hits])), axis=0)
        if not len(cells):
            return None
        xy = np.array([self.occupancy.cell_to_world(tuple(cell)) for cell in cells])
        delta = xy - self.camera_position[:2]
        bearings = np.arctan2(delta[:, 1], delta[:, 0])
        unseen = (self.seen[cells[:, 0], cells[:, 1]] & (1 << self._sectors(bearings))) == 0
        best, best_score = None, 0.0
        for turn in np.deg2rad((-90, -75, -60, -45, -30, 30, 45, 60, 75, 90)):
            yaw = self.camera_yaw + turn
            visible = unseen & (np.abs(angle_error(bearings, yaw)) <= np.deg2rad(self.fov / 2 - 10))
            # Metres of new projected boundary, weighted against distant hits.
            gain = float(np.sum(cfg.grid_resolution / (1 + np.linalg.norm(delta[visible], axis=1) / self.max_range)))
            score = gain / (1 + abs(turn) / .5)
            if gain >= min_gain and score > best_score:
                best, best_score = LookDirection(float(yaw), gain), score
        return best
