"""Measured field map on the navigation grid's axes, including both sides of the fence."""
from dataclasses import replace
import math

import numpy as np

from grutopia_extension.interactive_navigation.mapping import OccupancyGridMap
from grutopia_extension.interactive_navigation.online_lidar_evidence import online_lidar_masks


class FieldSurvey:
    def __init__(self, config, navigation_shape):
        self.config = config
        self.navigation_shape = navigation_shape
        self.grid = None
        self.padding = 0

    def _ensure_range(self, max_range):
        # The scanner operates inside the fence. One sensor range beyond each
        # edge contains every usable return without enlarging the planner.
        cell_size = self.config.grid_resolution
        padding = math.ceil(max_range / cell_size) + 1
        if self.grid is not None and padding <= self.padding:
            return
        def limits(start, count):
            return (start-padding*cell_size, start+(count+padding)*cell_size)
        rows, cols = self.navigation_shape
        cfg = replace(self.config, x_limits=limits(self.config.x_limits[0], cols),
                      y_limits=limits(self.config.y_limits[0], rows))
        grid = OccupancyGridMap(cfg)
        if self.grid is not None:
            offset = padding-self.padding
            h, w = self.grid.observed.shape
            region = np.s_[offset:offset+h, offset:offset+w]
            grid.observed[region] = self.grid.observed
            grid.log_odds[region] = self.grid.log_odds
        self.padding, self.grid = padding, grid

    def scan_masks(self, origin, points, max_range, min_range):
        self._ensure_range(max_range)
        grid = self.grid
        hit, free, stats = online_lidar_masks(grid.config, origin, points, max_range, min_range)
        grid.observed |= hit | free
        grid.log_odds[hit] += grid.config.occupied_update
        grid.log_odds[free] -= grid.config.free_update
        np.clip(grid.log_odds, -4., 4., out=grid.log_odds)
        r, (h, w) = self.padding, self.navigation_shape
        return hit[r:r+h, r:r+w], free[r:r+h, r:r+w], stats

    def snapshot(self, navigation):
        """Trim unused margins. Real exterior evidence surrounds the bounded navigation layers."""
        if self.grid is None:
            return None
        r, (h, w) = self.padding, self.navigation_shape
        observed = self.grid.observed
        rows = np.flatnonzero(observed.any(axis=1))
        cols = np.flatnonzero(observed.any(axis=0))
        y0, y1 = (min(r, rows[0]), max(r+h, rows[-1]+1)) if rows.size else (r, r+h)
        x0, x1 = (min(r, cols[0]), max(r+w, cols[-1]+1)) if cols.size else (r, r+w)
        extent = np.s_[y0:y1, x0:x1]
        seen = observed[extent].copy()
        occupied = self.grid.log_odds[extent] >= self.config.occupied_threshold
        y, x = int(r-y0), int(r-x0)
        where = navigation.interior | navigation.boundary_wall
        np.copyto(seen[y:y+h, x:x+w], navigation.observed, where=where)
        np.copyto(occupied[y:y+h, x:x+w], navigation.local_occupied, where=where)
        step = self.config.grid_resolution
        xmin = self.config.x_limits[0] - x*step
        ymin = self.config.y_limits[0] - y*step
        return dict(observed=seen, occupied=occupied, navigation_offset_rc=(y, x),
                    x_limits=(xmin, xmin+seen.shape[1]*step), y_limits=(ymin, ymin+seen.shape[0]*step))
