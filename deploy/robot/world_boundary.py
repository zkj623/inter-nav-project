"""Rectangular operating area in the fixed experiment frame, in metres."""
from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class WorldBoundary:
    x: tuple
    y: tuple

    @classmethod
    def from_navigation(cls, nav):
        enabled = nav.get('world_boundary', False)
        if type(enabled) is not bool:
            raise ValueError('navigation.world_boundary must be boolean')
        if not enabled:
            return None
        mapping = nav['mapping']
        axes = []
        for key in ('x_limits', 'y_limits'):
            limits = mapping.get(key)
            if (not isinstance(limits, (list, tuple)) or len(limits) != 2
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in limits)
                    or not limits[0] < limits[1]):
                raise ValueError(f'mapping.{key} must contain increasing finite limits in metres')
            axes.append(tuple(limits))
        return cls(*axes)

    def grid_config(self, mapping):
        # One cell for a permanent wall, one for visible unknown space beyond.
        r = mapping.grid_resolution
        def padded(limits):
            lo, hi = limits
            cells = math.ceil(round((hi-lo)/r, 12)) + 4
            # Exact grid lines such as x=1 must land in the same cell as
            # x=1.001 despite subtraction and integer truncation by the planner.
            start = lo - (2 + 1e-12)*r
            end = start + cells*r
            # OccupancyGridMap uses ceil. Avoid an extra column from roundoff.
            if (end-start)/r > cells:
                end = np.nextafter(end, start)
            return (start, float(end))
        return replace(mapping, x_limits=padded(self.x), y_limits=padded(self.y),
                       safe_recovery_y_limits=self.y)

    def interior_mask(self, config, shape):
        # Only full interior cells can be traversed for non-grid-aligned bounds.
        r = config.grid_resolution
        def interior(count, start, limits):
            edge = start + np.arange(count)*r
            return (edge >= limits[0]-1e-10*r) & (edge+r <= limits[1]+1e-10*r)
        return (interior(shape[0], config.y_limits[0], self.y)[:, None]
                & interior(shape[1], config.x_limits[0], self.x)[None, :])

    def contains(self, xy):
        return self.x[0] < xy[0] < self.x[1] and self.y[0] < xy[1] < self.y[1]

    def point_mask(self, points):
        return ((points[:, 0] >= self.x[0]) & (points[:, 0] <= self.x[1])
                & (points[:, 1] >= self.y[0]) & (points[:, 1] <= self.y[1]))

    def contains_box(self, xy, dimensions, yaw, margin=0.):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        a, b = dimensions[0]/2, dimensions[1]/2
        dx, dy = a*c+b*s+margin, a*s+b*c+margin
        return self.contains((xy[0]-dx, xy[1]-dy)) and self.contains((xy[0]+dx, xy[1]+dy))

    def contains_motion(self, xy, yaw, velocity, horizon, corners):
        """Enclose every body corner over a constant body-frame twist."""
        vx, vy, wz = velocity
        speed = math.hypot(vx, vy)
        radius = float(np.hypot(corners[:, 0], corners[:, 1]).max())
        clearance = min(xy[0]-self.x[0], self.x[1]-xy[0], xy[1]-self.y[0], self.y[1]-xy[1])
        if clearance > radius + speed*horizon:
            return True
        if wz:
            horizon = min(horizon, 2*math.pi/abs(wz))  # A constant twist repeats after a full turn.
        steps = max(1, math.ceil(abs(wz)*horizon / (math.pi/12)))
        times = np.linspace(0., horizon, steps+1)
        turn = wz*times
        mid = yaw + turn/2
        travel = times*np.sinc(turn/(2*math.pi))
        x = xy[0] + travel*(vx*np.cos(mid)-vy*np.sin(mid))
        y = xy[1] + travel*(vx*np.sin(mid)+vy*np.cos(mid))
        c, s = np.cos(yaw+turn)[:, None], np.sin(yaw+turn)[:, None]
        xs = x[:, None] + c*corners[:, 0] - s*corners[:, 1]
        ys = y[:, None] + s*corners[:, 0] + c*corners[:, 1]
        # The largest deviation of a corner from its sampled chord is at
        # most max_acceleration * dt^2 / 8. Straight segments are exact.
        error = (abs(wz)*speed+wz*wz*radius)*(horizon/steps)**2/8
        return bool(xs.min() > self.x[0]+error and xs.max() < self.x[1]-error
                    and ys.min() > self.y[0]+error and ys.max() < self.y[1]-error)
