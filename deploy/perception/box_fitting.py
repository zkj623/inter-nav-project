"""Measured, upright cuboid poses from one LiDAR scan (no temporal cloud fusion).

The full size is registered; pose is constrained by two visible, perpendicular
side faces with observed end boundaries. A confirmed track may subsequently
use one complete face only with independent free-ray proof outside both edges.
A partial or unbounded plane is deliberately not a box estimate. The floor is experiment z=0, already calibrated upstream.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class BoxFit:
    center: tuple
    yaw_rad: float
    dimensions: tuple
    residual_m: float
    visible_faces: tuple
    face_spans: tuple
    point_count: int
    uncertainty_m: float
    indices: tuple = ()
    geometry_source: str = 'current_lidar_two_side_faces'
    boundary_rays: tuple = ()
    measured_face_axis: int | None = None


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _clusters(points, cell_size, min_points):
    """Bounded 2-D occupied-cell components; vertical returns stay together."""
    cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    buckets = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(index)
    while buckets:
        seed, indices = buckets.popitem()
        pending = [seed]
        while pending:
            x, y = pending.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (x + dx, y + dy)
                    values = buckets.pop(neighbour, None)
                    if values is not None:
                        indices.extend(values)
                        pending.append(neighbour)
        if len(indices) >= min_points:
            yield np.asarray(indices, dtype=int)


def fit_upright_box(points, dimensions, origin, config):
    """Return a BoxFit or None; the supplied component must be untruncated.

    Pose is fitted in the current scan. Registered height fixes center_z on the
    calibrated floor; observed vertical extent must independently support that
    upright model. No mass, unseen edges or scan time is invented.
    """
    p = np.asarray(points, dtype=float)
    dims = np.asarray(dimensions, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < config['min_points']
            or not np.isfinite(p).all() or dims.shape != (3,)
            or not np.isfinite(dims).all() or np.any(dims <= 0)
            or origin.shape != (3,) or not np.isfinite(origin).all()):
        return None
    height = dims[2]
    zlow, zhigh = np.quantile(p[:, 2], [.01, .99])
    tolerance = config['height_tolerance_m']
    if (zlow > tolerance or zhigh < height - tolerance
            or zhigh > height + tolerance or zlow < -tolerance
            or zhigh-zlow < config['min_height_fraction']*height):
        return None
    # Top returns do not masquerade as lateral-face evidence.
    sides = p[(p[:, 2] >= config['floor_clearance_m']) & (p[:, 2] <= height-.025)]
    if len(sides) < config['min_points']:
        return None
    xy = sides[:, :2]
    half = dims[:2]/2

    def evaluate(yaw, detailed=False):
        rotation = _rotation(yaw)
        local = xy @ rotation
        lower, upper = np.quantile(local, [.005, .995], axis=0)
        center = (lower+upper)/2
        centered = local-center
        extent_error = np.abs((upper-lower)-dims[:2])
        distances = np.abs(np.abs(centered)-half)
        residual = np.min(distances, axis=1)
        score = float(np.mean(np.minimum(residual, .10)**2) + np.sum(extent_error**2))
        if detailed:
            return rotation, center, centered, extent_error, residual
        return score

    angles = np.arange(0., math.pi, math.radians(config['angle_step_deg']))
    scores = np.array([evaluate(a) for a in angles])
    index = int(np.argmin(scores))
    step = math.radians(config['angle_step_deg'])
    optimum = minimize_scalar(evaluate, bounds=(angles[index]-step, angles[index]+step),
                              method='bounded', options={'xatol': 1e-5, 'maxiter': 25})
    yaw = float((optimum.x+math.pi/2) % math.pi-math.pi/2)
    rotation, center, local, extent_error, residual = evaluate(yaw, True)
    if np.any(extent_error > config['extent_tolerance_m']):
        return None
    threshold = config['residual_threshold_m']
    if (np.mean(residual <= threshold) < config['min_inlier_fraction']
            or np.sqrt(np.mean(residual**2)) > threshold):
        return None
    sensor = origin[:2] @ rotation-center
    faces, spans = [], []
    # Two adjacent side faces must both face the LiDAR, and have independent
    # observed extent. A large wall with no box boundaries fails this check.
    for axis in (0, 1):
        if abs(sensor[axis]) <= half[axis] + threshold:
            return None
        sign = 1 if sensor[axis] > 0 else -1
        at_face = np.abs(local[:, axis]-sign*half[axis]) <= threshold
        values = local[at_face, 1-axis]
        if len(values) < config['min_face_points']:
            return None
        lo, hi = np.quantile(values, [.02, .98])
        span = float(hi-lo)
        expected = dims[1-axis]
        if (span < config['min_face_coverage']*expected
                or abs(lo+half[1-axis]) > config['extent_tolerance_m']
                or abs(hi-half[1-axis]) > config['extent_tolerance_m']):
            return None
        faces.append(('+' if sign > 0 else '-') + ('x' if axis == 0 else 'y'))
        spans.append(span)
    # Returns inside the model or on its occluded back faces do not support a
    # visible-surface model. This rejects combined objects and through-box clouds.
    visible_distances = np.stack([
        np.abs(local[:, axis] - (1 if sensor[axis] > 0 else -1)*half[axis])
        for axis in (0, 1)], axis=1)
    if np.mean(np.min(visible_distances, axis=1) <= threshold) < config['min_inlier_fraction']:
        return None
    center_xy = center @ rotation.T
    rms = float(np.sqrt(np.mean(residual**2)))
    uncertainty = max(config['geometry_margin_m'], rms*3, float(extent_error.max())/2)
    return BoxFit((float(center_xy[0]), float(center_xy[1]), float(height/2)), yaw,
                  tuple(map(float, dims)), rms, tuple(faces), tuple(spans), len(p), uncertainty)


def scan_candidates(points, dimensions, origin, config):
    """Fit complete scan components before applying registration search regions.

    Never crop a wall to a registration ROI: that could manufacture box edges.
    Neighbouring surfaces in the clearance shell invalidate an otherwise good fit.
    """
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or len(points) > config['max_scan_points']):
        raise ValueError('LiDAR requires finite N x 3 within max_scan_points')
    height = float(dimensions[2])
    valid = ((points[:, 2] >= config['floor_clearance_m'])
             & (points[:, 2] <= height+config['height_tolerance_m']))
    original = np.flatnonzero(valid)
    p = points[valid]
    results = []
    for indices in _clusters(p, config['cluster_cell_m'], config['min_points']):
        component = p[indices]
        # Rotation-independent diameter bound avoids expensive fits on walls.
        if np.any(np.ptp(component[:, :2], axis=0) > math.hypot(*dimensions[:2])+2*config['extent_tolerance_m']):
            continue
        fit = fit_upright_box(component, dimensions, origin, config)
        if fit is None:
            continue
        # Height gating is only for component extraction. Check the uncropped
        # source too, so a tall pillar cannot acquire a manufactured top edge.
        all_local = (points[:, :2]-np.asarray(fit.center[:2])) @ _rotation(fit.yaw_rad)
        above = (np.all(np.abs(all_local) <= np.asarray(dimensions[:2])/2 + config['residual_threshold_m'], axis=1)
                 & (points[:, 2] > height+config['height_tolerance_m']))
        if np.count_nonzero(above) >= config['min_face_points']:
            continue
        local = all_local[valid]
        shell = np.all(np.abs(local) <= np.asarray(dimensions[:2])/2 + config['isolation_margin_m'], axis=1)
        shell[indices] = False
        if np.count_nonzero(shell) >= config['min_face_points']:
            continue
        results.append(BoxFit(**{**fit.__dict__, 'indices': tuple(map(int, original[indices]))}))
        if len(results) > config['max_candidates']:
            raise ValueError('Too many candidate boxes; identity is ambiguous')
    return tuple(results)


def _bounded_face_rays(all_points, origin, normal, tangent, plane_offset, edges, height, depth, config):
    """Prove free space just outside BOTH observed side edges using this scan.

    The rays must cross the face's plane at box height and continue behind the
    registered far face. A return on the plane is not free evidence. Independent
    vertical samples prevent one duplicate ray from manufacturing an edge.
    """
    rays = all_points-origin
    denominator = rays[:, :2] @ normal
    numerator = plane_offset-float(origin[:2] @ normal)
    fraction = np.divide(numerator, denominator, out=np.full(len(rays), np.nan), where=denominator < -1e-8)
    crossing = origin+fraction[:, None]*rays
    crossing_u = crossing[:, :2] @ tangent
    far_fraction = np.divide(numerator-depth, denominator, out=np.full(len(rays), np.nan), where=denominator < -1e-8)
    far_u = (origin+far_fraction[:, None]*rays)[:, :2] @ tangent
    behind = plane_offset-all_points[:, :2] @ normal
    valid = (np.isfinite(fraction) & (fraction > 0.) & (fraction < 1.)
             & (behind >= depth+config['boundary_ray_depth_margin_m'])
             & (crossing[:, 2] >= config['floor_clearance_m']+.02)
             & (crossing[:, 2] <= height-.025) & (all_points[:, 2] >= -.025))
    counts = []
    for sign, edge in ((-1, edges[0]), (1, edges[1])):
        gap = sign*(crossing_u-edge)
        far_gap = sign*(far_u-edge)
        selected = (valid & (gap >= config['boundary_ray_min_gap_m'])
                    & (gap <= config['boundary_ray_max_gap_m'])
                    & (far_gap >= config['boundary_ray_min_gap_m']))
        crossings = crossing[selected]
        # Quantize geometry rather than count repeated packet coordinates.
        independent = np.unique(np.floor(crossings/.01).astype(np.int64), axis=0)
        if (len(independent) < config['min_boundary_rays']
                or len(crossings) == 0
                or np.ptp(crossings[:, 2]) < config['boundary_ray_height_fraction']*height):
            return None
        counts.append(len(independent))
    return tuple(counts)


def fit_tracked_face(points, all_points, dimensions, origin, config, previous):
    """Fresh bounded single-face pose, only for an already identified cuboid.

    The previous fit selects a consistent local face identity, never its current
    position. Plane normal/offset, both lateral endpoints and vertical span are
    measured again. Only hidden thickness and floor-supported height come from
    registration. Both lateral edges need independent current free-ray proof.
    """
    p, all_points = np.asarray(points, float), np.asarray(all_points, float)
    dims, origin = np.asarray(dimensions, float), np.asarray(origin, float)
    if len(p) < config['min_points']:
        return None
    height = dims[2]
    zlow, zhigh = np.quantile(p[:, 2], [.005, .995])
    if (zlow > config['floor_clearance_m']+.025 or zhigh < height-config['extent_tolerance_m']
            or zhigh > height+config['height_tolerance_m']
            or zhigh-zlow < max(.8, config['min_height_fraction'])*height):
        return None
    xy = p[:, :2]
    centroid = np.mean(xy, axis=0)
    _, vectors = np.linalg.eigh(np.cov(xy.T))
    normal = vectors[:, 0]
    if np.dot(origin[:2]-centroid, normal) < 0:
        normal = -normal
    offset = float(np.median(xy @ normal))
    residual = np.abs(xy @ normal-offset)
    threshold = config['residual_threshold_m']
    if (np.sqrt(np.mean(residual**2)) > threshold
            or np.mean(residual <= threshold) < config['min_inlier_fraction']
            or np.dot(origin[:2], normal)-offset <= threshold):
        return None
    old_axes = _rotation(previous.yaw_rad)
    alignment = normal @ old_axes
    axis = int(np.argmax(abs(alignment)))
    # A history hint can select the equivalent face, not override a measured
    # large turn. The tighter frame-to-frame angular gate remains in the owner.
    if abs(alignment[axis]) < math.cos(math.radians(25.)):
        return None
    sign = 1 if alignment[axis] > 0 else -1
    if axis == 0:
        xaxis = sign*normal
    else:
        yaxis = sign*normal
        xaxis = np.array([yaxis[1], -yaxis[0]])
    yaw = math.atan2(xaxis[1], xaxis[0])
    tangent = _rotation(yaw)[:, 1-axis]
    edges = np.quantile(xy @ tangent, [.005, .995])
    span = float(edges[1]-edges[0])
    width, depth = dims[1-axis], dims[axis]
    if abs(span-width) > config['extent_tolerance_m'] or span < config['min_face_coverage']*width:
        return None
    plane_center = offset*normal + float(np.mean(edges))*tangent
    center_xy = plane_center-normal*depth/2
    # Require each half of the face to have real near-top and near-bottom
    # returns, rather than combining disjoint short patches into full height.
    for low, high in ((edges[0], np.mean(edges)), (np.mean(edges), edges[1])):
        subset = p[((xy @ tangent) >= low) & ((xy @ tangent) <= high)]
        if len(subset) < config['min_face_points'] or np.ptp(subset[:, 2]) < .8*height:
            return None
    proof = _bounded_face_rays(all_points, origin, normal, tangent, offset, edges, height, depth, config)
    if proof is None:
        return None
    local_all = (all_points[:, :2]-center_xy) @ _rotation(yaw)
    # Neither a tall pillar nor a second surface hidden in this rectangle can
    # be removed by clipping the fit component's Z or XY coordinates.
    footprint = np.all(np.abs(local_all) <= dims[:2]/2+threshold, axis=1)
    above = footprint & (all_points[:, 2] > height+config['height_tolerance_m'])
    if np.count_nonzero(above) >= config['min_face_points']:
        return None
    rms = float(np.sqrt(np.mean(residual**2)))
    uncertainty = max(config['geometry_margin_m'], 3*rms, abs(span-width)/2)
    return BoxFit((float(center_xy[0]), float(center_xy[1]), float(height/2)), yaw,
        tuple(map(float, dims)), rms, (('+' if sign > 0 else '-')+('x' if axis == 0 else 'y'),),
        (span,), len(p), uncertainty, geometry_source='current_lidar_bounded_single_face',
        boundary_rays=proof, measured_face_axis=axis)


def scan_tracked_candidates(points, dimensions, origin, config, previous):
    """Single-face candidates supplement two-face fits only for a fresh hint."""
    points = np.asarray(points, float)
    height = float(dimensions[2])
    valid = ((points[:, 2] >= config['floor_clearance_m'])
             & (points[:, 2] <= height+config['height_tolerance_m']))
    original, p = np.flatnonzero(valid), points[valid]
    results = []
    for indices in _clusters(p, config['cluster_cell_m'], config['min_points']):
        component = p[indices]
        if np.any(np.ptp(component[:, :2], axis=0) > math.hypot(*dimensions[:2])+2*config['extent_tolerance_m']):
            continue
        fit = fit_tracked_face(component, points, dimensions, origin, config, previous)
        if fit is None:
            continue
        local = (p[:, :2]-np.asarray(fit.center[:2])) @ _rotation(fit.yaw_rad)
        nearby = np.all(np.abs(local) <= np.asarray(dimensions[:2])/2+config['isolation_margin_m'], axis=1)
        nearby[indices] = False
        if np.count_nonzero(nearby) >= config['min_face_points']:
            continue
        results.append(BoxFit(**{**fit.__dict__, 'indices': tuple(map(int, original[indices]))}))
        if len(results) > config['max_candidates']:
            raise ValueError('Too many bounded-face candidates; identity is ambiguous')
    return tuple(results)
