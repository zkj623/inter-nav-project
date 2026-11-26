"""Exact ray evidence for unknown-map navigation.

Floor and ceiling returns only clear cells where the measured beam crosses
the obstacle-height band. The nearest in-band hit clips its azimuth bin.
"""
import numpy as np


def _mark_cells(mask, cols, rows):
    inside = (cols >= 0) & (rows >= 0) & (cols < mask.shape[1]) & (rows < mask.shape[0])
    mask[rows[inside], cols[inside]] = True


def _mark_segments(mask, starts, ends):
    """Traverse measured segments exactly, retiring short rays from the batch."""
    if not len(starts):
        return
    cells = np.floor(starts).astype(np.intp)
    targets = np.floor(ends).astype(np.intp)
    # Each traversal step crosses at least one row or column until the ray
    # reaches its endpoint. Sorting by that bound lets us skip finished rays.
    lengths = np.abs(targets - cells).sum(axis=1)
    order = np.argsort(lengths)
    lengths = lengths[order]
    starts, ends = starts[order], ends[order]
    cols, rows = cells[order].T
    end_cols, end_rows = targets[order].T
    dx, dy = (ends - starts).T
    sx, sy = np.sign(dx).astype(np.intp), np.sign(dy).astype(np.intp)
    dx = np.where(sx == 0, 1., dx)
    dy = np.where(sy == 0, 1., dy)
    tx = (cols + (sx > 0) - starts[:, 0]) / dx
    ty = (rows + (sy > 0) - starts[:, 1]) / dy
    dtx, dty = np.abs(1. / dx), np.abs(1. / dy)
    # The traversal stays within the rectangle between its two endpoints.
    shape_xy = mask.shape[::-1]
    inside = ((cells >= 0).all() and (targets >= 0).all()
              and (cells < shape_xy).all() and (targets < shape_xy).all())
    for step in range(lengths[-1] + 1):
        live = slice(lengths.searchsorted(step), None)
        col, row = cols[live], rows[live]
        if inside:
            mask[row, col] = True
        else:
            _mark_cells(mask, col, row)
        more_x, more_y = col != end_cols[live], row != end_rows[live]
        x_time, y_time = tx[live], ty[live]
        # Corner touches enter only the diagonal cell. Stop each coordinate
        # at its measured endpoint, even within the corner tolerance.
        corner = np.abs(x_time - y_time) < 1e-10
        step_x = more_x & (~more_y | (x_time < y_time) | corner)
        step_y = more_y & (~more_x | (y_time < x_time) | corner)
        col += sx[live] * step_x
        row += sy[live] * step_y
        x_time += dtx[live] * step_x
        y_time += dty[live] * step_y


def _first_in_bins(indices, bins, *keys):
    """First beam per bin, with the last key primary and input order breaking ties."""
    order = indices[np.lexsort((*keys, bins[indices]))]
    return order[np.r_[True, bins[order[1:]] != bins[order[:-1]]]]


def online_lidar_masks(config, origin, points, max_range, min_range=.25):
    resolution = config.grid_resolution
    shape = (int(np.ceil((config.y_limits[1] - config.y_limits[0]) / resolution)),
             int(np.ceil((config.x_limits[1] - config.x_limits[0]) / resolution)))
    offset = np.array([config.x_limits[0], config.y_limits[0]])
    hit, free = np.zeros(shape, bool), np.zeros(shape, bool)
    points = np.asarray(points, float).reshape(-1, 3)
    origin = np.asarray(origin, float)
    delta = points - origin
    distance = np.linalg.norm(delta, axis=1)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    valid = (np.isfinite(points).all(axis=1) & (distance >= min_range)
             & (distance <= max_range + min_range + .01) & (horizontal > 1e-8))
    delta, points, distance, horizontal = delta[valid], points[valid], distance[valid], horizontal[valid]
    stats = {'rays': len(points), 'range_rays': 0, 'free_segments': 0}
    if not len(points):
        return hit, free, stats
    bins = min(config.lidar_scan_bins, config.max_lidar_rays)
    azimuth = ((np.arctan2(delta[:, 1], delta[:, 0]) + np.pi) / (2*np.pi) * bins).astype(int) % bins
    lower, upper = config.obstacle_height
    measured = np.flatnonzero((distance < max_range*.99) & (points[:, 2] >= lower) & (points[:, 2] <= upper))
    limits = np.full(bins, np.inf)
    if measured.size:
        nearest = _first_in_bins(measured, azimuth, horizontal[measured])
        limits[azimuth[nearest]] = horizontal[nearest]
        cols, rows = np.floor((points[nearest, :2] - offset) / resolution).astype(np.intp).T
        _mark_cells(hit, cols, rows)

    end = np.minimum(np.minimum(1., max_range*.99 / distance), limits[azimuth] / horizontal)
    dz = delta[:, 2]
    flat = np.abs(dz) < 1e-9
    slope = np.where(flat, 1., dz)
    t0, t1 = (lower - origin[2]) / slope, (upper - origin[2]) / slope
    first = np.where(flat, 0., np.maximum(0., np.minimum(t0, t1)))
    last = np.where(flat, end, np.minimum(end, np.maximum(t0, t1)))
    candidates = np.flatnonzero((~flat | (lower <= origin[2] <= upper)) & (last - first >= 1e-8))
    if candidates.size:
        # Retain two actual beams per azimuth, the longest reach and the
        # nearest start (breaking ties by longest reach). A shared beam counts once.
        reach = horizontal[candidates] * last[candidates]
        near = horizontal[candidates] * first[candidates]
        chosen = np.union1d(_first_in_bins(candidates, azimuth, -reach),
                            _first_in_bins(candidates, azimuth, -reach, near))
        starts = origin[:2] + first[chosen, None] * delta[chosen, :2]
        ends = origin[:2] + last[chosen, None] * delta[chosen, :2]
        _mark_segments(free, (starts - offset) / resolution, (ends - offset) / resolution)
        stats['free_segments'] = len(chosen)
        stats['range_rays'] = int(np.count_nonzero(distance[chosen] >= max_range*.99))
    free &= ~hit
    return hit, free, stats
