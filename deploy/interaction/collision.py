"""Contact checks from measured rays, independent of movable-object map masks.

The ordinary mapper deliberately owns whole object envelopes. This second,
small evidence grid never erases those envelopes when objects move. A temporary
contact query permits only the fitted target solid; all other returns and
unobserved cells keep blocking. No query mutates navigation or stored evidence.
"""
from copy import copy
from dataclasses import replace
import math
import numpy as np

from grutopia_extension.interactive_navigation.mapping import OccupancyGridMap
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear


class ContactCollisionChecker:
    def __init__(self, config, planner, profile):
        self.config, self.planner, self.profile = config, planner, profile
        self.evidence = OccupancyGridMap(replace(config, sticky_free_scale=1.))
        self.last_scan = None
        self.points = np.empty((0, 3))
        self.diagnostic = {'clear': False, 'reason': 'no_contact_scan'}
        rows, cols = np.indices(self.evidence.observed.shape)
        self.xy = np.stack((planner.limits[0][0]+(cols+.5)*planner.resolution,
                            planner.limits[1][0]+(rows+.5)*planner.resolution), axis=-1)

    def update(self, observation, layers):
        lidar = layers.filter_robot_returns(observation).get('sensors', {}).get('lidar', {})
        scan = lidar.get('physics_step')
        if scan is None or scan == self.last_scan or layers.last_scan != scan:
            return
        points = np.asarray(lidar.get('pointcloud', []), float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            self.points = np.empty((0, 3))
            return
        # The mapper has already traced these exact rays before masking object
        # ownership. Reuse them; tracing a large scan twice stalls the control loop.
        hit = getattr(layers, 'measured_hit', None)
        measured_free = getattr(layers, 'measured_free', None)
        if hit is None or measured_free is None:
            self.points = np.empty((0, 3))
            return
        free = measured_free & ~hit
        self.evidence.log_odds[hit] += self.config.occupied_update
        self.evidence.log_odds[free] -= self.config.free_update
        np.clip(self.evidence.log_odds, -4., 4., out=self.evidence.log_odds)
        self.evidence.observed[hit | free] = True
        self.points, self.last_scan = points.copy(), scan

    @staticmethod
    def _geometry(box):
        if box.local_dimensions is None or box.orientation is None or box.center_z is None:
            raise ValueError('contact_requires_full_box_geometry')
        dimensions = np.asarray(box.local_dimensions, float)
        q = np.asarray(box.orientation, float)
        if dimensions.shape != (3,) or q.shape != (4,) or not np.isfinite([*dimensions, *q, box.center_z]).all():
            raise ValueError('invalid_contact_box_geometry')
        w, x, y, z = q/np.linalg.norm(q)
        if abs(x)+abs(y) > .05:
            raise ValueError('contact_box_not_upright')
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c, s = math.cos(yaw), math.sin(yaw)
        return dimensions, np.array([[c, -s], [s, c]])

    def _mask(self, center, dimensions, rotation, margin=0.):
        local = (self.xy-np.asarray(center)) @ rotation
        # Cover cell centers and their half-diagonal, not just grid nodes.
        padding = margin+self.planner.resolution/math.sqrt(2)
        return (np.abs(local) <= dimensions[:2]/2+padding).all(axis=-1)

    def _point_mask(self, points):
        result = np.zeros_like(self.evidence.observed)
        if not len(points):
            return result
        col = np.floor((points[:, 0]-self.planner.limits[0][0])/self.planner.resolution).astype(int)
        row = np.floor((points[:, 1]-self.planner.limits[1][0])/self.planner.resolution).astype(int)
        valid = (row >= 0) & (row < result.shape[0]) & (col >= 0) & (col < result.shape[1])
        result[row[valid], col[valid]] = True
        return result

    def __call__(self, command, context, *, pushing=True):
        def finish(clear, reason):
            self.diagnostic = {'clear': bool(clear), 'reason': reason, 'scan_id': self.last_scan}
            return bool(clear)
        try:
            target = context.box
            dims, rot = self._geometry(target)
            velocity = np.asarray(command, float)
            if velocity.shape != (3,) or not np.isfinite(velocity).all() or not len(self.points):
                return finish(False, 'contact_scan_or_command_invalid')
            points = self.points
            local = np.column_stack(((points[:, :2]-target.xy) @ rot, points[:, 2]-target.center_z))
            tol = min(.04, self.profile.release_margin_m)
            inside = (np.abs(local) <= dims/2+tol).all(axis=1)
            surface = (np.abs(np.abs(local)-dims/2) <= tol).any(axis=1)
            target_returns = inside & surface
            if np.count_nonzero(target_returns) < 3:
                return finish(False, 'target_surface_not_measured')
            z0, z1 = self.config.obstacle_height
            non_target = points[(~target_returns) & (points[:, 2] >= z0) & (points[:, 2] <= z1)]
            current = self._mask(target.xy, dims, rot)
            # Only cells wholly inside the known solid may waive unknown.
            # Boundary cells still require measured free space or target returns.
            core_local = (self.xy-np.asarray(target.xy)) @ rot
            core = (np.abs(core_local) <= np.maximum(0., dims[:2]/2-self.planner.resolution/math.sqrt(2))).all(axis=-1)
            owned = self._point_mask(points[target_returns]) | core
            free = self.evidence.observed & ~self.evidence.occupied_mask()
            free = free | owned
            # Actual occupied body footprint is known, without revealing future space.
            c, s = math.cos(context.heading), math.sin(context.heading)
            body = (self.xy-context.position[:2]) @ np.array([[c, -s], [s, c]])
            p = self.profile
            body_here = ((body[..., 0] >= p.foot_x_min_m) & (body[..., 0] <= p.foot_x_max_m)
                         & (np.abs(body[..., 1]) <= p.foot_half_width_m))
            free |= body_here
            free[self._point_mask(non_target)] = False
            shadow = copy(self.planner)
            shadow.free = shadow.box_free = free
            others = [box for box in context.boxes if box.id != target.id]
            measured = float(np.linalg.norm(context.body_velocity[:2]))
            requested = float(np.linalg.norm(velocity[:2]))
            distance = p.stopping_distance(measured, pushing=pushing, command_speed=requested)
            horizon = max(.2, p.yaw_stop_seconds, distance/max(requested, measured, .05))
            if not body_motion_clear(shadow, others, context.position[:2], context.heading,
                    velocity[0], velocity[2], horizon=horizon, lateral=velocity[1], margin=p.release_margin_m):
                return finish(False, 'non_target_body_sweep_blocked')
            measured_twist = np.array([*np.asarray(context.body_velocity)[:2],
                                       float(getattr(context, 'yaw_rate', 0.))])
            if not body_motion_clear(shadow, others, context.position[:2], context.heading,
                    measured_twist[0], measured_twist[2], horizon=horizon,
                    lateral=measured_twist[1], margin=p.release_margin_m):
                return finish(False, 'measured_body_stop_sweep_blocked')
            world_velocity = np.array([[c, -s], [s, c]]) @ velocity[:2]
            # A conservative box sweep includes the requested translation and
            # measured stopping envelope; target-only cells are allowed at t=0.
            if pushing and requested > 1e-8:
                direction = world_velocity/requested
                blocked = ~free
                for box in others:
                    mask = shadow.box_mask(box)
                    if mask is None:
                        return finish(False, 'other_box_outside_map')
                    blocked |= mask
                for step in np.linspace(0., distance, max(2, int(math.ceil(distance/shadow.resolution))+1)):
                    center = np.asarray(target.xy)+direction*step
                    corners = np.array([[a, b] for a in (-1., 1.) for b in (-1., 1.)])
                    corners = (corners*(dims[:2]/2+p.release_margin_m)) @ rot.T+center
                    if not all(np.all((corners[:, axis] >= lo) & (corners[:, axis] < hi))
                               for axis, (lo, hi) in enumerate(shadow.limits)):
                        return finish(False, 'box_sweep_outside_map')
                    swept = self._mask(center, dims, rot, p.release_margin_m)
                    if np.any(swept & ~current & blocked):
                        return finish(False, 'box_sweep_blocked_or_unknown')
            return finish(True, 'measured_contact_sweep_clear')
        except (ValueError, TypeError, IndexError, AttributeError, FloatingPointError):
            return finish(False, 'invalid_contact_geometry')
