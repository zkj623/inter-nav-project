"""Locate two bounded perpendicular faces in a window of experiment-frame returns."""
from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np


RESOLUTION = .02


@dataclass
class WindowFit:
    center: np.ndarray
    face_angle: float  # radians modulo pi/2; no tag-to-face identity is inferred
    support_index: int  # earlier of the most recent supporting scan on each face
    residual: float
    long_axis_rad: Optional[float] = None  # radians modulo pi for distinguishable sides

    def angle_to(self, other):
        if self.long_axis_rad is None or other.long_axis_rad is None:
            return face_angle_difference(self.face_angle, other.face_angle)
        return abs((self.long_axis_rad-other.long_axis_rad+math.pi/2) % math.pi-math.pi/2)


def inward_axes(angle, corner, origin):
    c, s = math.cos(angle), math.sin(angle)
    axes = np.array([[c, -s], [s, c]])
    return axes * np.where((corner - origin[:2]) @ axes >= 0, 1., -1.)


def face_angle_difference(first, second):
    return abs((first - second + math.pi/4) % (math.pi/2) - math.pi/4)


def proposals(points, origin, lengths, region):
    """Search a distance image for finite L shapes; include space outside the registry."""
    if not len(points):
        return []
    x0, x1, y0, y1 = region
    radius = .5 * math.hypot(*lengths) + .04
    padding = radius + max(lengths) + .28
    anchor = np.array([x0 - padding, y0 - padding])
    width, height = np.ceil((np.array([x1, y1]) + padding - anchor) / RESOLUTION).astype(int) + 1
    image = np.ones((height, width), np.uint8)
    cells = np.rint((points - anchor) / RESOLUTION).astype(int)
    x, y = cells.T
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not inside.any():
        return []
    image[y[inside], x[inside]] = 0
    distance = cv2.distanceTransform(image, cv2.DIST_L2, 5) * RESOLUTION
    iy, ix = np.where(distance <= .055)
    xy = anchor + RESOLUTION * np.column_stack((ix, iy))
    keep = ((xy[:, 0] >= x0-radius) & (xy[:, 0] <= x1+radius) &
            (xy[:, 1] >= y0-radius) & (xy[:, 1] <= y1+radius))
    ix, iy, xy = ix[keep], iy[keep], xy[keep]
    delta = xy - origin[:2]
    hypotheses = []
    for angle in np.deg2rad(np.arange(0., 90., 3.)):
        c, s = math.cos(angle), math.sin(angle)
        u, v = np.array([c, s]), np.array([-s, c])
        # Two components per point. Avoid dispatching these tiny projections
        # to a threaded BLAS for every orientation on the onboard computer.
        pu = delta[:, 0]*c + delta[:, 1]*s
        pv = delta[:, 1]*c - delta[:, 0]*s
        su, sv = np.where(pu >= 0, 1, -1), np.where(pv >= 0, 1, -1)
        centers = xy + .5 * (lengths[0]*su[:, None]*u + lengths[1]*sv[:, None]*v)
        ix_ok = np.flatnonzero((centers[:, 0] >= x0) & (centers[:, 0] <= x1) &
                               (centers[:, 1] >= y0) & (centers[:, 1] <= y1) &
                               (np.abs(pu) > .04) & (np.abs(pv) > .04))
        def sample(axis, sign, length):
            offsets = np.linspace(.025, length-.02, 10)
            dx, dy = np.rint(axis[:, None]*offsets/RESOLUTION).astype(int)
            return distance[iy[ix_ok, None] + sign[ix_ok, None]*dy, ix[ix_ok, None] + sign[ix_ok, None]*dx]
        du, dv = sample(u, su, lengths[0]), sample(v, sv, lengths[1])
        ok = ((du <= .045).sum(axis=1) >= 7) & ((dv <= .045).sum(axis=1) >= 7)
        costs = np.minimum(du, .10).mean(axis=1) + np.minimum(dv, .10).mean(axis=1)
        for k in np.flatnonzero(ok):
            i = ix_ok[k]
            hypotheses.append((costs[k], xy[i], angle, centers[i]))
    selected = []
    for item in sorted(hypotheses, key=lambda value: value[0]):
        _, _, angle, center = item
        if not any(np.linalg.norm(center - old[3]) < .075 and
                   face_angle_difference(angle, old[2]) < .13 for old in selected):
            selected.append(item)
    return selected


def verify(points, sources, origin, lengths, height, guess, theta, config):
    tol = config['residual_threshold_m']
    axes = inward_axes(theta, guess, origin)
    center = guess + axes @ lengths / 2
    subset = (np.abs(points[:, :2]-center) <= .5*math.hypot(*lengths)+.24).all(axis=1)
    local, scan_ids = points[subset], sources[subset]
    middle = (local[:, 2] >= .20*height) & (local[:, 2] <= .85*height)
    xy = local[middle, :2]
    uv = (xy-guess) @ axes
    faces = [(np.abs(uv[:, i]) <= .055) & (uv[:, 1-i] > .035) & (uv[:, 1-i] < lengths[1-i]-.015)
             for i in (0, 1)]
    minimum = config['min_face_points']
    if any(mask.sum() < minimum for mask in faces):
        return None
    best_score = math.inf
    for angle in theta + np.deg2rad(np.arange(-5., 6.)):
        rotation = inward_axes(angle, guess, origin)
        q = (xy-guess) @ rotation
        offsets = np.array([np.median(q[mask, i]) for i, mask in enumerate(faces)])
        score = sum(np.mean(np.minimum(np.abs(q[mask, i]-offsets[i]), .045)**2)
                    for i, mask in enumerate(faces))
        if score < best_score:
            best_score, best_angle = score, angle
            axes, corner = rotation, guess + rotation @ offsets
    uv = (local[:, :2]-corner) @ axes
    support = []
    for axis in (0, 1):
        across, along = uv[:, axis], uv[:, 1-axis]
        length = lengths[1-axis]
        face = middle & (np.abs(across) <= tol)
        fitted = face & (along >= -.025) & (along <= length+.04)
        if fitted.sum() < minimum:
            return None
        scans = np.unique(scan_ids[fitted & (along > .045)])
        if len(scans) < 2:
            return None
        trace = along[fitted]
        lo, hi = np.quantile(trace, [.02, .98])
        if hi-lo < .75*length or hi < .82*length or lo > .065:
            return None
        if len(np.unique(np.clip((trace/length*5).astype(int), 0, 4))) < 4:
            return None
        for beyond in (face & (along < -.065) & (along > -.20),
                       face & (along > length+.065) & (along < length+.20)):
            if beyond.sum() >= max(minimum, .16*fitted.sum()) and np.ptp(along[beyond]) > .055:
                return None
        support.append(scans[-1])
    inside = (uv >= -tol).all(axis=1) & (uv <= lengths+tol).all(axis=1)
    height_margin = config['height_tolerance_m']
    if (inside & (local[:, 2] > height+height_margin)).sum() >= minimum:
        return None
    wall = (np.min(np.abs(uv), axis=1) <= tol) & inside
    zs = local[wall, 2]
    zs = zs[(zs >= config['floor_clearance_m']) & (zs <= height+height_margin)]
    if len(zs) < config['min_points'] or np.quantile(zs, .01) > height_margin or np.quantile(zs, .99) < height-height_margin:
        return None
    if (inside & middle & ~wall).sum() > .20 * max(1, (inside & middle).sum()):
        return None
    if np.any((origin[:2]-corner) @ axes >= -tol):
        return None
    residual = np.min(np.abs(uv[wall & middle]), axis=1)
    long_axis = None
    if lengths[0] != lengths[1]:
        long_axis = (best_angle + (math.pi/2 if lengths[1] > lengths[0] else 0.)) % math.pi
    return WindowFit(corner + axes @ lengths/2, best_angle % (math.pi/2),
                     int(min(support)), float(np.sqrt(np.mean(residual**2))), long_axis)


def find_window_boxes(points, sources, origin, dimensions, region, config):
    height = dimensions[2]
    sides = np.asarray(dimensions[:2], float)
    x0, x1, y0, y1 = region
    # Keep the entire verification patch of every allowed center, including
    # the continuation of both planes beyond the expected ends.
    halo = .5 * math.hypot(*sides) + .40
    nearby = ((points[:, 0] >= x0-halo) & (points[:, 0] <= x1+halo) &
              (points[:, 1] >= y0-halo) & (points[:, 1] <= y1+halo))
    points, sources = points[nearby], sources[nearby]
    middle = (points[:, 2] >= .20*height) & (points[:, 2] <= .85*height)
    # Below the depth resolution the long/short axes are interchangeable.
    assignments = ([np.full(2, sides.mean())] if abs(sides[0]-sides[1]) <= config['residual_threshold_m']
                   else [sides, sides[::-1]])
    found = []
    for lengths in assignments:
        for _, corner, angle, center in proposals(points[middle, :2], origin, lengths, region):
            if any(np.linalg.norm(center-fit.center) < .055 for fit in found):
                continue
            fit = verify(points, sources, origin, lengths, height, corner, angle, config)
            if fit is None:
                continue
            x, y = fit.center
            if x0 <= x <= x1 and y0 <= y <= y1 and not any(np.linalg.norm(fit.center-v.center) < .075 for v in found):
                found.append(fit)
    return found


def refit_window_box(points, sources, origin, dimensions, config, previous, reach, turn, *, search_region=True):
    """Measure fresh faces around a previous pose within the association gate."""
    sides = np.array(sorted(dimensions[:2], reverse=True))
    angle = previous.long_axis_rad
    if angle is None:
        sides[:] = sides.mean()
        angle = previous.face_angle
    # A box contains a circle of diameter min(sides). Test whether a second
    # disjoint footprint could fit in the gate beside the measured center.
    separation = min(sides)-2*config['residual_threshold_m']
    if reach < separation:
        axes = inward_axes(angle, previous.center, origin)
        corner = previous.center - axes @ sides/2
        fit = verify(points, sources, origin, sides, dimensions[2], corner, angle, config)
        if fit is not None:
            offset = np.linalg.norm(fit.center-previous.center)
            if offset <= reach and offset+reach < separation and fit.angle_to(previous) <= turn:
                return [fit]
    if not search_region:
        return []
    x, y = previous.center
    region = (x-reach, x+reach, y-reach, y+reach)
    return [fit for fit in find_window_boxes(points, sources, origin, dimensions, region, config)
            if np.linalg.norm(fit.center-previous.center) <= reach and fit.angle_to(previous) <= turn]
