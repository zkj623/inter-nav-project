"""Bounded collision-checked A* for a holonomic base in XY or SE(2)."""

from dataclasses import dataclass
import heapq
import itertools
import math
import time

import numpy as np

from grutopia_extension.interaction_clearance.motion_geometry import angle_error


class PlanningError(RuntimeError):
    pass


@dataclass
class BasePath:
    waypoints: list
    expanded_nodes: int
    collision_queries: int
    search_seconds: float
    used_search: bool


def pose_samples(start, target, spatial_step=0.01, angular_step=math.radians(1)):
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    delta = np.r_[target[:2] - start[:2], angle_error(target[2], start[2])]
    count = max(1, math.ceil(np.linalg.norm(delta[:2]) / spatial_step), math.ceil(abs(delta[2]) / angular_step))
    return [start + progress * delta for progress in np.linspace(0, 1, count + 1)]


class SE2GridPlanner:
    def __init__(self, pose_is_free, bounds, fixed_yaw=False, resolution=0.10, yaw_step_deg=10,
                 max_expansions=12000, max_seconds=30.0, yaw_offsets=None, yaw_cost_m_per_rad=0.35):
        self.pose_is_free = pose_is_free
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (2, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[0] >= self.bounds[1]):
            raise ValueError('Planner bounds must be finite ordered XY corners')
        if not 0.02 <= resolution <= 0.25 or not 2 <= yaw_step_deg <= 30 or max_expansions < 1 or max_seconds <= 0:
            raise ValueError('Invalid planner resolution or budget')
        if not math.isfinite(yaw_cost_m_per_rad) or yaw_cost_m_per_rad <= 0:
            raise ValueError('Yaw motion cost must be finite and positive')
        self.fixed_yaw = fixed_yaw
        self.resolution = resolution
        self.yaw_bins = round(360 / yaw_step_deg)
        self.yaw_step = 2 * math.pi / self.yaw_bins
        self.yaw_offsets = None
        self.yaw_cost = yaw_cost_m_per_rad
        if yaw_offsets is not None:
            offsets = np.asarray(yaw_offsets, dtype=float)
            if (fixed_yaw or offsets.ndim != 1 or not 2 <= len(offsets) <= 8 or not np.isfinite(offsets).all()
                    or abs(offsets[0]) > 1e-8):
                raise ValueError('Limited headings require 2..8 finite offsets starting at zero')
            self.yaw_offsets = [angle_error(float(offset), 0) for offset in offsets]
            if len(set(np.round(self.yaw_offsets, 7))) != len(offsets):
                raise ValueError('Heading offsets must be distinct')
            self.yaw_bins = len(offsets)
        self.max_expansions = max_expansions
        self.max_seconds = max_seconds
        self.cache = {}
        self.expanded = 0

    def key(self, pose):
        return tuple(np.round(np.r_[pose[:2], angle_error(pose[2], 0.0)], 7))

    def free(self, pose):
        if time.monotonic() - self.started > self.max_seconds:
            raise PlanningError('PLANNING_TIME_LIMIT')
        if np.any(pose[:2] < self.bounds[0]) or np.any(pose[:2] > self.bounds[1]):
            return False
        key = self.key(pose)
        if key not in self.cache:
            self.cache[key] = bool(self.pose_is_free(pose))
        return self.cache[key]

    def edge_free(self, start, target, fine=True):
        spacing = 0.01 if fine else min(0.04, self.resolution / 2)
        angular = math.radians(1) if fine else min(math.radians(3), self.yaw_step / 3)
        return all(self.free(pose) for pose in pose_samples(start, target, spacing, angular))

    def connection(self, start, target):
        if self.fixed_yaw:
            candidates = [[start, target]]
        else:
            candidates = [[start, np.r_[start[:2], target[2]], target],
                          [start, np.r_[target[:2], start[2]], target]]
        for points in candidates:
            if all(self.edge_free(first, second) for first, second in zip(points, points[1:])):
                return points
        return None

    def pose(self, state):
        yaw_offset = state[2] * self.yaw_step if self.yaw_offsets is None else self.yaw_offsets[state[2]]
        return np.r_[self.start[:2] + self.resolution * np.asarray(state[:2]),
                     angle_error(self.start[2] + yaw_offset, 0)]

    def heuristic(self, pose):
        return float(np.linalg.norm(pose[:2] - self.goal[:2]) + self.yaw_cost * abs(angle_error(self.goal[2], pose[2])))

    def search(self, forbidden):
        initial = (0, 0, 0)
        serial = itertools.count()
        queue = [(self.heuristic(self.start), next(serial), 0.0, initial)]
        costs = {initial: 0.0}
        parents = {}
        translations = [(horizontal, vertical, 0) for horizontal in (-1, 0, 1) for vertical in (-1, 0, 1)
                        if horizontal or vertical]
        turns = [(0, 0, jump) for jump in range(1, self.yaw_bins)] if self.yaw_offsets is not None else [(0, 0, -1), (0, 0, 1)]
        motions = translations if self.fixed_yaw else translations + turns
        while queue:
            _, _, cost, state = heapq.heappop(queue)
            if cost > costs[state] + 1e-9:
                continue
            self.expanded += 1
            if self.expanded > self.max_expansions:
                raise PlanningError('PLANNING_EXPANSION_LIMIT')
            current = self.pose(state)
            if (np.linalg.norm(current[:2] - self.goal[:2]) <= 1.5 * self.resolution
                    and abs(angle_error(self.goal[2], current[2])) <= self.yaw_step + 1e-8):
                connection = self.connection(current, self.goal)
                if connection is not None:
                    chain = [state]
                    while chain[-1] != initial:
                        chain.append(parents[chain[-1]])
                    return [self.pose(item) for item in reversed(chain)] + connection[1:]
            for horizontal, vertical, turn in motions:
                child = (state[0] + horizontal, state[1] + vertical, (state[2] + turn) % self.yaw_bins)
                target = self.pose(child)
                edge = (self.key(current), self.key(target))
                step_cost = self.resolution * math.hypot(horizontal, vertical) + self.yaw_cost * abs(angle_error(target[2], current[2]))
                proposed = cost + step_cost
                if proposed >= costs.get(child, math.inf) - 1e-9 or edge in forbidden:
                    continue
                if not self.free(target) or not self.edge_free(current, target, fine=False):
                    continue
                costs[child] = proposed
                parents[child] = state
                heapq.heappush(queue, (proposed + self.heuristic(target), next(serial), proposed, child))
        raise PlanningError('NO_PATH_IN_SEARCH_BOUNDS')

    def simplify(self, points):
        distinct = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point[:2] - distinct[-1][:2]) > 1e-8 or abs(angle_error(point[2], distinct[-1][2])) > 1e-8:
                distinct.append(point)
        result = [distinct[0]]
        index = 0
        while index < len(distinct) - 1:
            chosen = index + 1
            for candidate in range(len(distinct) - 1, index, -1):
                target = distinct[candidate]
                pure_translation = abs(angle_error(target[2], distinct[index][2])) < 1e-7
                pure_rotation = np.linalg.norm(target[:2] - distinct[index][:2]) < 1e-7
                if (pure_translation or pure_rotation) and self.edge_free(distinct[index], target):
                    chosen = candidate
                    break
            result.append(distinct[chosen])
            index = chosen
        bounded = [result[0]]
        for first, second in zip(result, result[1:]):
            count = max(1, math.ceil(np.linalg.norm(second[:2] - first[:2]) / 1.2))
            delta = np.r_[second[:2] - first[:2], angle_error(second[2], first[2])]
            bounded.extend(first + progress * delta for progress in np.linspace(0, 1, count + 1)[1:])
        return bounded

    def plan(self, start, goal):
        self.start, self.goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float).copy()
        if self.start.shape != (3,) or self.goal.shape != (3,) or not np.isfinite(np.r_[self.start, self.goal]).all():
            raise ValueError('Planner requires finite XY/yaw poses')
        if self.fixed_yaw:
            if abs(angle_error(self.goal[2], self.start[2])) > 0.01:
                raise PlanningError('CARRYING_YAW_ROTATION_UNSUPPORTED')
            self.goal[2] = self.start[2]
        self.started = time.monotonic()
        self.cache = {}
        self.expanded = 0
        if not self.free(self.start) or not self.free(self.goal):
            raise PlanningError('START_OR_GOAL_IN_COLLISION')
        path = self.connection(self.start, self.goal)
        used_search = path is None
        forbidden = set()
        if path is None:
            for _attempt in range(6):
                path = self.search(forbidden)
                invalid = next(((first, second) for first, second in zip(path, path[1:])
                                if not self.edge_free(first, second)), None)
                if invalid is None:
                    break
                edge = tuple(self.key(pose) for pose in invalid)
                forbidden.add(edge)
                forbidden.add(edge[::-1])
            else:
                raise PlanningError('PATH_CERTIFICATION_FAILED')
        path = self.simplify(path)
        if not all(self.edge_free(first, second) for first, second in zip(path, path[1:])):
            raise PlanningError('SMOOTHED_PATH_CERTIFICATION_FAILED')
        return BasePath(path, self.expanded, len(self.cache), time.monotonic() - self.started, used_search)
