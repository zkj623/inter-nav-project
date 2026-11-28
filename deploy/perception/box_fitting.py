"""Fit registered upright boxes on the calibrated experiment floor.

Side-face fits use one scan. The upper-short-end field preset also tracks a
known box by its lid, using recent points for orientation and current returns
on the lid and its near short edge for position along the push direction.
"""
from __future__ import annotations

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
    top_end: str | None = None

    def sees_push_face(self, direction, tolerance):
        """Check the planned push against the measured side face or short lid edge."""
        local_push = np.asarray(direction) @ _rotation(self.yaw_rad)
        length = math.hypot(*local_push)
        inward = {'-x': local_push[0], '+x': -local_push[0],
                  '-y': local_push[1], '+y': -local_push[1]}
        ends = (self.top_end,) if self.top_end is not None else self.visible_faces
        return length > 0 and any(inward[face] >= length*math.cos(tolerance) for face in ends)


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def refit_box_top(points, dimensions, origin, config, previous, recent=()):
    """Track the registered rectangle from its lid and the near short top edge."""
    dims = np.asarray(dimensions)
    height, half = dims[2], dims[:2]/2
    long = int(np.argmax(half))
    across = 1-long
    tol = config['residual_threshold_m']
    if origin[2] <= height+tol or half[long]-half[across] <= tol:
        return None
    anchor = np.asarray(previous.center[:2])
    axes = _rotation(previous.yaw_rad)
    current_count = len(points)
    if recent:
        points = np.concatenate((points, *recent))
    local = (points[:, :2]-anchor) @ axes
    sensor = (origin[:2]-anchor) @ axes
    if abs(sensor[long]) <= half[long]+tol:
        return None
    sign = np.sign(sensor[long])
    z = points[:, 2]
    near = (abs(local) <= half+config['isolation_margin_m']+.05).all(axis=1)
    band = near & (abs(z-height) <= min(config['height_tolerance_m'], 1.75*tol))
    minimum = max(6, config['min_face_points']-2)
    if band.sum() < minimum:
        return None
    level = np.median(z[band])
    ids = np.flatnonzero(band & (abs(z-level) <= tol))
    outside = (abs(local[ids]) > half+2*tol).any(axis=1)
    newest = ids < current_count
    if (outside[newest].sum() >= max(2, math.ceil(.10*newest.sum()))
            or outside.sum() >= max(2, math.ceil(.10*len(ids)))):
        return None
    ids = ids[~outside]
    uv = local[ids]
    if len(ids) < minimum:
        return None
    inside = (abs(local) <= half-tol).all(axis=1)
    if np.count_nonzero(inside & (z > height+config['height_tolerance_m'])) >= minimum:
        return None
    span = np.ptp(uv, axis=0)
    if span[across] < .60*dims[across] or span[long] < .35*dims[long]:
        return None
    # Reject lines, tilted surfaces and the upper rims of vertical walls.
    inner = (abs(uv) < half-tol).all(axis=1)
    if inner.sum() < 3 or np.linalg.svd(uv-uv.mean(0), compute_uv=False)[1] < .25*dims[across]:
        return None
    slopes = np.linalg.lstsq(uv-uv.mean(0), z[ids]-z[ids].mean(), rcond=None)[0]
    if np.linalg.norm(slopes) > .16 or np.std(z[ids]) > .6*tol:
        return None
    # Measure the near rim on both halves of the lid. The registered depth
    # supplies the far edge when the grazing beam misses it.
    along = sign*uv[:, long]
    leading = np.quantile(along, .98)
    if abs(leading-half[long]) > config['extent_tolerance_m']:
        return None
    rim = along >= leading-tol
    current = ids < current_count
    if current.sum() < 4 or not (rim & current).any() or not (inner & current).any():
        return None
    if np.ptp(uv[current, across]) < .40*dims[across] or np.ptp(uv[current, long]) < .25*dims[long]:
        return None
    rim_points = uv[rim]
    low, high = rim_points[:, across].min(), rim_points[:, across].max()
    if high-low < .36*dims[across] or abs((high+low)/2) > .25*dims[across]:
        return None
    ends = []
    for group in (rim_points[:, across] <= low+.15*(high-low),
                  rim_points[:, across] >= high-.15*(high-low)):
        part = rim_points[group]
        ends.append(part[np.argmax(sign*part[:, long])])
    edge = ends[1]-ends[0]
    if edge[across] < .36*dims[across]:
        return None
    angle = math.atan2(edge[long], edge[across]) * (1 if across == 0 else -1)
    if abs(angle) > math.radians(2*config['association_yaw_margin_deg']):
        return None
    yaw = previous.yaw_rad + .25*angle
    refined = _rotation(yaw)
    projected = (points[ids, :2]-anchor) @ refined
    bounds = np.quantile(projected, [.02, .98], axis=0)
    offset = bounds.mean(axis=0)
    # Old points stabilize the edge angle and width. Measure progress along
    # the push axis from the current scan, including when the box is moving.
    offset[long] = sign*(np.quantile(sign*projected[rim & current, long], .9)-half[long])
    if np.linalg.norm(offset) > config['extent_tolerance_m']:
        return None
    center = anchor + refined @ offset
    end = ('+' if sign > 0 else '-')+'xy'[long]
    return BoxFit((*center, height/2), yaw, tuple(dims), float(np.std(z[ids])), ('top',),
                  tuple(span), int(current.sum()), config['geometry_margin_m'], tuple(map(int, ids[current])),
                  geometry_source='current_lidar_registered_top', top_end=end)


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
    # The XY bounding diagonal bounds every projected span, at any yaw.
    if np.linalg.norm(np.ptp(xy, axis=0)) < dims[:2].max()-config['extent_tolerance_m']:
        return None

    def evaluate_rotation(rotation, detailed=False):
        local = xy @ rotation
        lower, upper = np.quantile(local, [.005, .995], axis=-2)
        center = (lower+upper)/2
        centered = local-center[..., None, :]
        extent_error = np.abs((upper-lower)-dims[:2])
        distances = np.abs(np.abs(centered)-half)
        residual = np.min(distances, axis=-1)
        if detailed:
            return rotation, center, centered, extent_error, residual
        return np.mean(np.minimum(residual, .10)**2, axis=-1) + np.sum(extent_error**2, axis=-1)

    def evaluate(yaw):
        return float(evaluate_rotation(_rotation(yaw)))

    # Score the same trial angles together; bound temporary memory on large clouds.
    angles = np.arange(0., math.pi, math.radians(config['angle_step_deg']))
    scores = np.empty(len(angles))
    batch_size = max(1, 32768//len(xy))
    for start in range(0, len(angles), batch_size):
        batch = slice(start, start+batch_size)
        rotations = np.stack([_rotation(angle) for angle in angles[batch]])
        scores[batch] = evaluate_rotation(rotations)
    index = int(np.argmin(scores))
    step = math.radians(config['angle_step_deg'])
    optimum = minimize_scalar(evaluate, bounds=(angles[index]-step, angles[index]+step),
                              method='bounded', options={'xatol': 1e-5, 'maxiter': 25})
    yaw = float((optimum.x+math.pi/2) % math.pi-math.pi/2)
    rotation, center, local, extent_error, residual = evaluate_rotation(_rotation(yaw), detailed=True)
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

    Use rays behind the far face, or rays to exposed floor outside a low
    carton's ends when the floor cuts off the far-face vertical span.
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
    floor_tolerance = config['floor_clearance_m']
    upper_crossing = height-.025
    required_span = config['boundary_ray_height_fraction']*height
    valid = (np.isfinite(fraction) & (fraction > 0.) & (fraction < 1.)
             & (crossing[:, 2] >= floor_tolerance+.02)
             & (crossing[:, 2] <= upper_crossing) & (all_points[:, 2] >= -.025))
    depth_margin = config['boundary_ray_depth_margin_m']
    proof_depth = depth + depth_margin
    past_box = behind >= proof_depth
    distance = -numerator
    allow_floor = False
    if distance > 0. and origin[2] > 0. and edges[0] < origin[:2] @ tangent < edges[1]:
        # Height at the front face of a ray to the floor at the required depth.
        # A high scanner can run out of vertical span past a low box even
        # when it is more than one box depth away from its front face.
        floor_crossing = floor_tolerance + (origin[2]-floor_tolerance)*proof_depth/(distance+proof_depth)
        allow_floor = distance <= proof_depth or (origin[2] > height and upper_crossing-floor_crossing <= required_span)
    endpoint_u = all_points[:, :2] @ tangent
    counts = []
    for sign, edge in ((-1, edges[0]), (1, edges[1])):
        gap = sign*(crossing_u-edge)
        far_gap = sign*(far_u-edge)
        outside = past_box & (far_gap >= config['boundary_ray_min_gap_m'])
        if allow_floor:
            # The measured segment from crossing to ground remains outside
            # the side plane. It establishes an edge, not free space under
            # or behind the box; motion still uses independent sweep evidence.
            floor = (np.abs(all_points[:, 2]) <= floor_tolerance)
            outside |= floor & (behind >= 2*depth_margin) & (sign*(endpoint_u-edge) >= config['boundary_ray_min_gap_m'])
        selected = valid & (gap >= config['boundary_ray_min_gap_m']) & (gap <= config['boundary_ray_max_gap_m']) & outside
        crossings = crossing[selected]
        # Quantize geometry rather than count repeated packet coordinates.
        independent = np.unique(np.floor(crossings/.01).astype(np.int64), axis=0)
        if (len(independent) < config['min_boundary_rays']
                or len(crossings) == 0
                or np.ptp(crossings[:, 2]) < required_span):
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


def refit_registered_surfaces(points, dimensions, origin, config, previous):
    """Measure pose from one sparse scan near an already identified box.

    Two nonparallel side planes constrain both center coordinates. When only
    one plane is visible, its two observed ends and rays past both ends must
    constrain the other coordinate. The map supplies only the search pose.
    """
    dims = np.asarray(dimensions, float)
    anchor = np.asarray(previous.center[:2])
    height, half = dims[2], dims[:2]/2
    tolerance = config['residual_threshold_m']
    axes = _rotation(previous.yaw_rad)
    side = np.sign((origin[:2]-anchor) @ axes)
    local = (points[:, :2]-anchor) @ axes
    z = points[:, 2]
    vertical = (z > max(.035, config['floor_clearance_m'])) & (z < height-.018)
    faces = {}
    min_points = config['min_face_points']
    for axis in range(2):
        other = 1-axis
        if side[axis]*((origin[:2]-anchor) @ axes[:, axis]) <= half[axis]+tolerance:
            continue
        # Exclude the shared corner so the two constraints use distinct returns.
        band = (vertical & (abs(local[:, axis]-side[axis]*half[axis]) <= 2*tolerance)
                & (abs(local[:, other]) < half[other]-.035))
        indices = np.flatnonzero(band)
        if len(indices) < min_points or np.ptp(z[indices]) < .32*height:
            continue
        xy = points[indices, :2]
        along = local[indices, other]
        span = float(np.ptp(along))
        if span < min(.14, .40*dims[other]):
            continue
        _, vectors = np.linalg.eigh(np.cov(xy.T))
        normal = vectors[:, 0]
        if normal @ axes[:, axis] < 0:
            normal = -normal
        delta = math.atan2(axes[0, axis]*normal[1]-axes[1, axis]*normal[0], normal @ axes[:, axis])
        if abs(delta) > math.radians(10.):
            continue
        faces[axis] = (indices, delta, span)
    if not faces:
        return None
    if len(faces) == 2 and abs(faces[0][1]-faces[1][1]) > math.radians(10.):
        return None
    weights = [len(idx)*span**2 for idx, _, span in faces.values()]
    yaw = previous.yaw_rad + float(np.average([delta for _,delta,_ in faces.values()], weights=weights))
    axes = _rotation(yaw)
    projected = (points[:, :2]-anchor) @ axes
    offset = np.zeros(2)
    proof = ()
    spans, names, owned, errors = [], [], [], []
    for axis, (indices, _, _) in faces.items():
        other = 1-axis
        plane = float(np.median(projected[indices, axis]))
        offset[axis] = plane-side[axis]*half[axis]
        residuals = projected[indices, axis]-plane
        rms = float(np.sqrt(np.mean(residuals**2)))
        if rms > tolerance*.65:
            return None
        errors.append(rms)
        on_plane = vertical & (abs(projected[:, axis]-plane) < tolerance)
        beyond = abs(projected[:, other])-half[other]
        extension = on_plane & (beyond > .055) & (beyond < .20)
        if np.count_nonzero(extension) >= min_points:
            return None
        trace = np.flatnonzero(on_plane & (beyond <= tolerance))
        owned.extend(map(int, trace))
        low, high = np.quantile(projected[trace, other], [.01, .99])
        spans.append(float(high-low))
        names.append(('+' if side[axis] > 0 else '-')+'xy'[axis])
        if len(faces) == 1:
            if high-low < dims[other]-config['extent_tolerance_m'] or high-low > dims[other]+2*tolerance:
                return None
            middle = (high+low)/2
            for half_trace in (trace[projected[trace, other] <= middle], trace[projected[trace, other] > middle]):
                if len(half_trace) < min_points or np.ptp(z[half_trace]) < .6*height:
                    return None
            tangent = axes[:, other]
            normal = side[axis]*axes[:, axis]
            bounds = np.array([low, high])+anchor @ tangent
            proof = _bounded_face_rays(points, origin, normal, tangent, side[axis]*plane+anchor @ normal,
                                       bounds, height, dims[axis], config)
            if proof is None:
                return None
            offset[other] = (high+low)/2
    if np.linalg.norm(offset) > 2*tolerance:
        return None
    interior = (abs(projected-offset) < half-2*tolerance).all(axis=1)
    if np.count_nonzero(vertical & interior) >= min_points:
        return None
    inside = (abs(projected-offset) <= half+tolerance).all(axis=1)
    if np.count_nonzero(inside & (z > height+config['height_tolerance_m'])) >= min_points:
        return None
    center = anchor+axes @ offset
    owned = tuple(sorted(set(owned)))
    heights = z[list(owned)]
    if np.ptp(heights) < .60*height or heights.max() < height-config['height_tolerance_m']:
        return None
    return BoxFit((*map(float, center), height/2), yaw, tuple(dimensions), max(errors), tuple(names),
                  tuple(spans), len(owned), max(config['geometry_margin_m'], 3*max(errors)), owned,
                  'current_lidar_registered_planes' if len(faces)==2 else 'current_lidar_bounded_single_face',
                  tuple(proof), next(iter(faces)) if len(faces)==1 else None)
