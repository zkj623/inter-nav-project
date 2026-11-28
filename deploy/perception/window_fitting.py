"""Locate a registered box from measured side faces or a bounded lid."""
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
    surfaces: str = 'two_sides'

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


def _top_support_for_partial_sides(uv, z, sources, lengths, height, partial, config):
    """Use a measured top to check dimensions left uncertain by two visible side planes."""
    tol, minimum = config['residual_threshold_m'], config['min_face_points']
    inside = np.all((uv >= -tol) & (uv <= lengths+tol), axis=1)
    interior = np.all((uv > 1.5*tol) & (uv < lengths-1.5*tol), axis=1)
    band = min(config['height_tolerance_m'], 1.75*tol)
    core = interior & (abs(z-height) <= band)
    if core.sum() < minimum:
        return None
    level = np.median(z[core])
    plane = abs(z-level) <= tol
    inner = plane & interior
    if inner.sum() < minimum or len(np.unique(sources[inner])) < 2:
        return None
    positions, heights = uv[inner], z[inner]
    tiles = np.floor(positions/lengths*3).astype(int)
    if len(np.unique(tiles, axis=0)) < 5 or np.any(np.ptp(positions, axis=0) < .40*lengths):
        return None
    slope = np.linalg.lstsq(positions-positions.mean(axis=0), heights-heights.mean(), rcond=None)[0]
    if np.linalg.norm(slope) > .10 or np.std(heights) > .6*tol:
        return None
    margin = config['isolation_margin_m']+.04
    ring = np.all((uv >= -margin) & (uv <= lengths+margin), axis=1)
    extends = np.any((uv < -1.5*tol) | (uv > lengths+1.5*tol), axis=1)
    if (ring & extends & plane).sum() >= max(minimum, .05*inner.sum()):
        return None
    support = int(sources[inner].max())
    for axis in partial:
        margin = min(config['extent_tolerance_m'], .18*lengths[axis])
        for edge in (0., lengths[axis]):
            measured = sources[plane & inside & (abs(uv[:, axis]-edge) <= margin)]
            if len(np.unique(measured)) < 2:
                return None
            support = min(support, int(measured.max()))
    return support


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


def verify(points, sources, origin, lengths, height, guess, theta, config, *, confirmed=False):
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
    support, partial = [], []
    wide_face = False
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
        bins = len(np.unique(np.clip((trace/length*5).astype(int), 0, 4)))
        span = hi-lo
        if span >= .65*length and bins >= 4:
            wide_face = True
        if span < .75*length or hi < .82*length or lo > .065 or bins < 4:
            # After acquisition the ends can be hidden. A short, measured
            # perpendicular face locates this box only beside a broad face.
            min_span = max(3*tol, .25*length) if confirmed else .40*length
            if (origin[2] <= height+tol or span < min_span or bins < (2 if confirmed else 3)
                    or np.ptp(local[fitted, 2]) < .30*height):
                return None
            partial.append(1-axis)
        for beyond in (face & (along < -.065) & (along > -.20),
                       face & (along > length+.065) & (along < length+.20)):
            if beyond.sum() >= max(minimum, .16*fitted.sum()) and np.ptp(along[beyond]) > .055:
                return None
        support.append(scans[-1])
    if partial and confirmed and not wide_face:
        return None
    # Once both dimensions have been confirmed, two intersecting measured
    # sides can locate the same box again without remeasuring its hidden ends.
    use_top = bool(partial) and not confirmed
    if use_top:
        top_scan = _top_support_for_partial_sides(uv, local[:, 2], scan_ids, lengths, height, partial, config)
        if top_scan is None:
            return None
        support.append(top_scan)
    inside = (uv >= -tol).all(axis=1) & (uv <= lengths+tol).all(axis=1)
    height_margin = config['height_tolerance_m']
    if (inside & (local[:, 2] > height+height_margin)).sum() >= minimum:
        return None
    wall = (np.min(np.abs(uv), axis=1) <= tol) & inside
    zs = local[wall, 2]
    zs = zs[(zs >= config['floor_clearance_m']) & (zs <= height+height_margin)]
    if (len(zs) < config['min_points'] or (not partial and np.quantile(zs, .01) > height_margin)
            or (partial and confirmed and np.ptp(zs) < .60*height)
            or np.quantile(zs, .99) < height-height_margin):
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
                     int(min(support)), float(np.sqrt(np.mean(residual**2))), long_axis,
                     'top_and_side' if use_top else 'two_sides')


def top_side_boxes(points, sources, origin, dimensions, region, config):
    """Find a bounded horizontal patch and verify its footprint and support."""
    height = dimensions[2]
    tol = config['residual_threshold_m']
    if origin[2] <= height+tol:
        return []
    lengths = np.sort(dimensions[:2])[::-1]
    minimum = config['min_face_points']
    height_band = min(config['height_tolerance_m'], 1.75*tol)
    top_ids = np.flatnonzero(np.abs(points[:, 2]-height) <= height_band)
    if len(top_ids) < minimum:
        return []
    top = points[top_ids]
    # Form components before checking the center against the search region.
    # The halo contains the full footprint plus continuation outside its edges.
    xmin, xmax, ymin, ymax = region
    halo = .5*np.linalg.norm(lengths)+.25
    base = np.array([xmin-halo, ymin-halo])
    step = .035
    cols, rows = np.ceil((np.array([xmax+halo, ymax+halo])-base)/step).astype(int)+1
    cells = np.floor((top[:, :2]-base)/step).astype(int)
    x, y = cells.T
    inside = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
    x, y, top_ids = x[inside], y[inside], top_ids[inside]
    raster = np.zeros((rows, cols), np.uint8)
    raster[y, x] = 1
    kernel = np.ones((3, 3), np.uint8)
    filled = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, kernel)
    patches = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(patches)
    membership = labels[y, x]
    found = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA]*step**2 < .45*np.prod(lengths):
            continue
        indices = top_ids[membership == label]
        if len(indices) < minimum:
            continue
        xy = np.ascontiguousarray(points[indices, :2], np.float32)
        center, size, angle = cv2.minAreaRect(xy)
        if (not xmin <= center[0] <= xmax or not ymin <= center[1] <= ymax
                or np.any(abs(np.sort(size)[::-1]-lengths) > config['extent_tolerance_m'])):
            continue
        angle = math.radians(angle + (90. if size[1] > size[0] else 0.)) % math.pi
        fit = verify_top_side(points, sources, origin, lengths, height, np.array(center), angle, config)
        if fit is not None and xmin <= fit.center[0] <= xmax and ymin <= fit.center[1] <= ymax:
            found.append(fit)
    return found


def verify_top_side(points, sources, origin, lengths, height, center, angle, config):
    tol, minimum = config['residual_threshold_m'], config['min_face_points']
    if origin[2] <= height+tol:
        return None
    c, s = math.cos(angle), math.sin(angle)
    axes = np.array([[c, -s], [s, c]])
    nearby = np.all(abs(points[:, :2]-center) <= .5*np.linalg.norm(lengths)+.18, axis=1)
    xyz, ids = points[nearby], sources[nearby]
    local = (xyz[:, :2]-center) @ axes
    half = lengths/2
    z = xyz[:, 2]
    # Measure all four top edges on every refit. A prior locates the patch
    # but cannot supply a hidden edge, center, or orientation update.
    height_band = min(config['height_tolerance_m'], 1.75*tol)
    core = (abs(z-height) <= height_band) & np.all(abs(local) < half-2*tol, axis=1)
    if core.sum() < minimum:
        return None
    level = float(np.median(z[core]))
    if origin[2] <= level+tol:
        return None
    top = (abs(z-level) <= tol) & np.all(abs(local) <= half+.07, axis=1)
    if top.sum() < minimum:
        return None
    lower, upper = np.quantile(local[top], [.01, .99], axis=0)
    extents = upper-lower
    if (np.any(abs(extents-lengths) > np.minimum(config['extent_tolerance_m'], .18*lengths))
            or np.any(extents < config['min_face_coverage']*lengths)):
        return None
    shift = (upper+lower)/2
    center = center + axes @ shift
    local -= shift
    footprint = np.all(abs(local) <= half+tol, axis=1)
    # Use interior returns, away from the upper edges of vertical walls.
    inner = top & np.all(abs(local) < half-1.5*tol, axis=1)
    uv, zz = local[inner], z[inner]
    if len(zz) < minimum or len(np.unique(ids[inner])) < 2:
        return None
    tiles = np.floor((uv/lengths+.5)*[4, 3]).astype(int)
    if len(np.unique(tiles[:, 0]*3+tiles[:, 1])) < 9:
        return None
    slopes = np.linalg.lstsq(uv-uv.mean(axis=0), zz-zz.mean(), rcond=None)[0]
    if np.linalg.norm(slopes) > .10 or np.std(zz) > .6*tol:
        return None
    ring = np.all(abs(local) <= half+config['isolation_margin_m']+.04, axis=1)
    extension = ring & ~np.all(abs(local) <= half+tol*1.5, axis=1) & (abs(z-level) <= tol)
    if extension.sum() >= max(minimum, .05*inner.sum()) or (footprint & (z > height+config['height_tolerance_m'])).sum() >= minimum:
        return None
    middle = (z >= .16*height) & (z <= level-2*tol)
    if (middle & np.all(abs(local) < half-2*tol, axis=1)).sum() >= max(minimum, .12*(middle & footprint).sum()):
        return None
    sensor = (origin[:2]-center) @ axes
    side_support = []
    for i in range(2):
        if abs(sensor[i]) <= half[i]+tol:
            continue
        j = 1-i
        side = middle & (abs(local[:, i]-np.sign(sensor[i])*half[i]) <= tol) & (abs(local[:, j]) < half[j])
        if side.sum() < minimum or len(np.unique(ids[side])) < 2:
            continue
        span = np.quantile(local[side, j], .98)-np.quantile(local[side, j], .02)
        if span >= .65*lengths[j] and np.ptp(z[side]) >= .35*height:
            side_support.append(ids[side].max())
    if not side_support:
        if config.get('use_box_top') and lengths[0] != lengths[1]:
            support = int(ids[inner].max())
            for axis in range(2):
                for edge in (-half[axis], half[axis]):
                    edge_ids = ids[top & (abs(local[:, axis]-edge) < min(.06, .18*lengths[axis]))]
                    if len(np.unique(edge_ids)) < 2:
                        return None
                    support = min(support, int(edge_ids.max()))
            return WindowFit(center, angle % (math.pi/2), support, float(np.std(zz)), angle, 'top')
        return None
    support = int(min(ids[inner].max(), max(side_support)))
    return WindowFit(center, angle % (math.pi/2), support, float(np.sqrt(np.mean((zz-level)**2))),
                     angle if lengths[0] != lengths[1] else None, 'top_and_side')


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
    found = top_side_boxes(points, sources, origin, dimensions, region, config)
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


def refit_window_box(points, sources, origin, dimensions, config, previous, reach, turn, *,
                     search_region=True, confirmed=False):
    """Measure fresh faces around a previous pose within the association gate."""
    sides = np.array(sorted(dimensions[:2], reverse=True))
    angle = previous.long_axis_rad
    if angle is None:
        sides[:] = sides.mean()
        angle = previous.face_angle
    # A box contains a circle of diameter min(sides). Test whether a second
    # disjoint footprint could fit in the gate beside the measured center.
    separation = min(sides)-2*config['residual_threshold_m']
    fit = None
    if reach < separation or (search_region and confirmed):
        axes = inward_axes(angle, previous.center, origin)
        corner = previous.center - axes @ sides/2
        fit = verify(points, sources, origin, sides, dimensions[2], corner, angle, config, confirmed=confirmed)
        if fit is None and reach < separation:
            x, y = previous.center
            # Refit a measured rectangle, including yaw, within the local gate.
            tops = top_side_boxes(points, sources, origin, dimensions, (x-reach, x+reach, y-reach, y+reach), config)
            fit = tops[0] if len(tops) == 1 else None
        if fit is not None:
            offset = np.linalg.norm(fit.center-previous.center)
            if offset > reach or fit.angle_to(previous) > turn:
                fit = None
            elif offset+reach < separation:
                return [fit]
    if not search_region:
        return []
    x, y = previous.center
    region = (x-reach, x+reach, y-reach, y+reach)
    candidates = [other for other in find_window_boxes(points, sources, origin, dimensions, region, config)
                  if np.linalg.norm(other.center-previous.center) <= reach and other.angle_to(previous) <= turn]
    # A wide gate can contain another box. Keep both when the other position
    # is distinct, so the observer reports the ambiguity instead of guessing.
    if confirmed and fit is not None:
        for index, other in enumerate(candidates):
            if np.linalg.norm(other.center-fit.center) < .075:
                if fit.support_index > other.support_index:
                    candidates[index] = fit
                break
        else:
            candidates.append(fit)
    return candidates
