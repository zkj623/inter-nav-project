"""Find a straight passage in the observed grid and traverse it at a held heading."""
from dataclasses import dataclass, replace
import math
from copy import copy

import numpy as np
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_footprint_samples
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner, OccupancyGridMap, PlanningError


def angle_error(target, heading):
    return (target - heading + math.pi) % (2 * math.pi) - math.pi


class PassageGrid:
    """Test the swept rectangle against entire occupied cells, including their edges."""

    def __init__(self, planner, free):
        self.free = free
        self.origin = np.array([axis[0] for axis in planner.limits])
        self.resolution = planner.resolution
        x, y = navigation_footprint_samples(planner)
        self.front, self.back = float(x.max()), float(-x.min())
        self.half_width = float(np.abs(y).max())
        self.radius = math.hypot(max(self.front, self.back), self.half_width)

    def clear(self, start, end, heading, margin=.02):
        start, end = np.asarray(start), np.asarray(end)
        forward = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-forward[1], forward[0]])
        length = (self.front + self.back) / 2 + margin
        width = self.half_width + margin
        travel = (end - start) / 2
        center = (start + end) / 2 + (self.front - self.back) / 2 * forward
        extent = abs(forward) * length + abs(left) * width + abs(travel)
        lo, hi = np.floor((np.array([center - extent, center + extent]) - self.origin) /
                          self.resolution).astype(int)
        if np.any(lo < 0) or np.any(hi >= self.free.shape[::-1]):
            return False
        rows, cols = np.nonzero(~self.free[lo[1]:hi[1]+1, lo[0]:hi[0]+1])
        if not len(rows):
            return True
        offsets = self.origin + (np.column_stack((cols, rows)) + lo + .5) * self.resolution - center
        # Separating axes of a cell and a rectangle extruded along the translation.
        axes = [forward, left]
        if np.linalg.norm(travel) > 1e-8:
            axes.append(np.array([-travel[1], travel[0]]) / np.linalg.norm(travel))
        for axis in axes:
            reach = (length * abs(forward @ axis) + width * abs(left @ axis)
                     + abs(travel @ axis) + .5 * self.resolution * abs(axis).sum())
            offsets = offsets[abs(offsets @ axis) <= reach + 1e-10]
            if not len(offsets):
                return True
        return False

    def can_turn(self, point, heading, target):
        delta = angle_error(target, heading)
        steps = max(1, math.ceil(abs(delta) / .10))
        # Enclose corner travel between two sampled headings.
        margin = .02 + self.radius * abs(delta) / (2 * steps)
        return all(self.clear(point, point, theta, margin)
                   for theta in np.linspace(heading, heading + delta, steps + 1))


@dataclass
class StraightPassage:
    entry: np.ndarray
    exit: np.ndarray
    heading: float
    approach: tuple


def find_passage(planner, grid, blocked, start, heading, goal):
    """Use a width-only path to locate a bottleneck. Only a swept-body proof admits it."""
    start, goal = np.asarray(start[:2]), np.asarray(goal[:2])
    r = planner.resolution
    # The guide may cut corners. It is never returned as a motion path.
    radius = grid.half_width
    guide = OccupancyGridMap(replace(planner.config, obstacle_inflation_radius=radius,
                                     static_obstacle_inflation_radius=radius))
    guide.observed[:] = True
    guide.static_occupied[:] = ~grid.free
    try:
        path = AStarMapPlanner(guide, observed_only=True).plan((*start, 0.), (*goal, 0.))
    except PlanningError:
        return None
    samples = [start]
    for waypoint in path:
        a, b = samples[-1], np.array(waypoint[:2])
        steps = max(1, math.ceil(np.linalg.norm(b-a) / (.5*r)))
        samples.extend(np.linspace(a, b, steps+1)[1:])
    points = np.array(samples)
    cells = tuple(np.array([planner.cell(xy) for xy in points]).T)
    pinched = np.flatnonzero(blocked[cells])
    if not len(pinched):
        return None
    # Handle one throat at a time. Continue normal planning in the room beyond.
    clearance = distance_transform_edt(np.pad(grid.free, 1))[1:-1, 1:-1] * r
    spacious = clearance[cells] > grid.radius + .5 * math.sqrt(2) * r + .02
    first = pinched[0]
    before = np.flatnonzero(spacious[:first])
    after = np.flatnonzero(spacious[first+1:]) + first + 1
    i = int(before[-1]) if len(before) else 0
    j = int(after[0]) if len(after) else len(points)-1
    a, b = points[i], points[j]
    axis = b-a
    if np.linalg.norm(axis) < r:
        return None
    theta = math.atan2(axis[1], axis[0])
    # Corner rounding on a coarse guide can skew the chord through a straight door.
    snapped = round(theta / (math.pi/4)) * math.pi/4
    for course in dict.fromkeys([snapped, theta]):
        direction = np.array([math.cos(course), math.sin(course)])
        normal = np.array([-direction[1], direction[0]])
        midpoint = np.median(points[i:j+1] @ normal)
        for shift in (0., -.5*r, .5*r, -r, r):
            offset = midpoint + shift
            entry = a + (offset - a @ normal) * normal
            exit_point = b + (offset - b @ normal) * normal
            if (i == 0 and np.linalg.norm(start-entry) > .18
                    or j == len(points)-1 and np.linalg.norm(goal-exit_point) > .12):
                continue
            if not grid.clear(entry, exit_point, course):
                continue
            if i == 0:
                if not grid.can_turn(start, heading, course) or not grid.clear(start, entry, course):
                    continue
                approach = ()
            else:
                # Allow an incoming tracker heading at the staging pose.
                if clearance[planner.cell(entry)] < grid.radius + r / math.sqrt(2) + .02:
                    continue
                view = copy(planner)
                view.free = grid.free
                approach = view.path(start, entry, ())
                if not approach:
                    continue
            return StraightPassage(entry, exit_point, course, tuple(approach))
    return None
