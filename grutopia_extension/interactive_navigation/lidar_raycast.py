"""Measured LiDAR rays through a bounded grid; no angular cone approximation."""
import numpy as np


def clearing_endpoints(origin, points, limit, phase=0):
    """Keep the nearest return on duplicate beams, then rotate a bounded sample.

    Distinct elevations must not hide each other's visibility. Rounding unit
    directions to six decimals only merges numerical duplicates (micrometres
    at our ranges), rather than the former five-degree vertical buckets.
    All obstacle hits are handled separately, before this clearing-only budget.
    """
    vectors = np.asarray(points, dtype=np.float64) - origin
    ranges = np.linalg.norm(vectors, axis=1)
    order = np.argsort(ranges, kind='stable')
    directions = np.round(vectors[order] / ranges[order, None], 6)
    _, first = np.unique(directions, axis=0, return_index=True)
    indices = order[first]
    if len(indices) > limit:
        indices = indices[(np.arange(limit)*len(indices)//limit + phase) % len(indices)]
    return points[indices]


def ray_cells(origin, endpoints, grid_origin, cell_size, lower, upper):
    """Return unique XYZ cells intersected by measured segments inside a box.

    Batched 3D DDA traverses actual grid-plane crossings. A cell touched only
    at an edge/corner has no free segment and is excluded. Clip to the map and
    useful height band, without extending rays beyond their measured return.
    Negative Z indices can describe near-ground free coverage below the
    obstacle band; callers decide which cells carry obstacle evidence.
    """
    origin = np.asarray(origin, dtype=np.float64)
    delta = np.asarray(endpoints, dtype=np.float64) - origin
    grid_origin, cell_size = np.asarray(grid_origin), np.asarray(cell_size)
    if not len(delta):
        return np.empty((0, 3), dtype=np.int64)
    moving = delta != 0
    inverse = np.divide(1., delta, out=np.full_like(delta, np.inf), where=moving)
    # Slab clipping handles parallel rays explicitly, including box faces.
    near = np.full_like(delta, -np.inf)
    far = np.full_like(delta, np.inf)
    np.multiply(np.asarray(lower)-origin, inverse, out=near, where=moving)
    np.multiply(np.asarray(upper)-origin, inverse, out=far, where=moving)
    entry = np.maximum(np.minimum(near, far).max(axis=1), 0.)
    exit_at = np.minimum(np.maximum(near, far).min(axis=1), 1.)
    parallel_outside = (~moving & ((origin < lower) | (origin >= upper))).any(axis=1)
    keep = (exit_at-entry > 1e-12) & ~parallel_outside & moving.any(axis=1)
    delta, inverse, entry, exit_at = delta[keep], inverse[keep], entry[keep], exit_at[keep]
    if not len(delta):
        return np.empty((0, 3), dtype=np.int64)
    step = np.sign(delta).astype(np.int64)
    # Move just inside the segment to assign a boundary start to its travel side.
    start = origin + (entry + np.minimum(1e-10, (exit_at-entry)/2))[:, None]*delta
    cells = np.floor((start-grid_origin)/cell_size).astype(np.int64)
    crossing = grid_origin + (cells + (step > 0))*cell_size
    next_at = np.full_like(delta, np.inf)
    np.multiply(crossing-origin, inverse, out=next_at, where=step != 0)
    increment = np.abs(cell_size*inverse)
    visited = []
    while len(cells):
        next_crossing = next_at.min(axis=1)
        end = np.minimum(next_crossing, exit_at)
        visited.append(cells[end-entry > 1e-12].copy())
        active = next_crossing < exit_at-1e-12
        if not active.any():
            break
        cells, step = cells[active], step[active]
        next_at, increment = next_at[active], increment[active]
        entry, exit_at = next_crossing[active], exit_at[active]
        crossed = next_at <= entry[:, None]+1e-12
        cells += crossed*step
        # Avoid 0*inf for axes parallel to a grid plane.
        np.add(next_at, increment, out=next_at, where=crossed)
    return np.unique(np.concatenate(visited), axis=0)
