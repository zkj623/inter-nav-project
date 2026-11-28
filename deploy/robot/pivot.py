"""Continuous-pose clearance for a Go2 turning in place on its navigation map."""
import math

import numpy as np

from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_footprint_samples


PIVOT_CLEARANCE_M = .025


def measured_pivot_clear(planner, boxes, position, heading, end_heading, drift_m=0.):
    """Test the padded body against entire blocked cell squares through the turn."""
    step = planner.resolution
    xx, yy = navigation_footprint_samples(planner, PIVOT_CLEARANCE_M)
    x_min, x_max = xx.min(), xx.max()
    length, width = (x_max - x_min) / 2, np.max(np.abs(yy))
    offset = (x_min + x_max) / 2
    radius = math.hypot(max(-x_min, x_max), width)
    # Between consecutive angles every body point is within this distance
    # of its nearest sampled pose. Enlarge the cell squares by that bound.
    sweep = end_heading - heading
    intervals = max(1, math.ceil(abs(sweep) / math.radians(.5)))
    padding = drift_m + 2 * radius * math.sin(abs(sweep) / (4 * intervals))
    xy = np.asarray(position[:2], float)
    origin = np.array([axis[0] for axis in planner.limits])
    lo = np.floor((xy - radius - padding - origin) / step).astype(int)
    hi = np.floor((xy + radius + padding - origin) / step).astype(int)
    cols, rows = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    inside = (rows >= 0) & (rows < planner.free.shape[0]) & (cols >= 0) & (cols < planner.free.shape[1])
    blocked = ~inside
    blocked[inside] = ~planner.free[rows[inside], cols[inside]]
    for box in boxes:
        # Include cells touching the fitted box as well as its measured returns.
        mask = planner.box_mask(box, padding=step / math.sqrt(2))
        if mask is None:
            return False
        blocked[inside] |= mask[rows[inside], cols[inside]]
    if not blocked.any():
        return True
    dx = origin[0] + (cols[blocked] + .5) * step - xy[0]
    dy = origin[1] + (rows[blocked] + .5) * step - xy[1]
    angles = np.linspace(heading, end_heading, intervals + 1)[:, None]
    c, s = np.cos(angles), np.sin(angles)
    cell = step / 2 + padding
    # Separating axes are world X/Y and the body's forward/lateral axes.
    body_cell = cell * (abs(c) + abs(s))
    overlap = ((abs(dx*c + dy*s - offset) <= length + body_cell)
               & (abs(-dx*s + dy*c) <= width + body_cell)
               & (abs(dx - offset*c) <= cell + length*abs(c) + width*abs(s))
               & (abs(dy - offset*s) <= cell + length*abs(s) + width*abs(c)))
    return not overlap.any()
