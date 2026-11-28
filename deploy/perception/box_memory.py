"""Confirmed box locations in one experiment frame. Contact evidence lives in the scan provider."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from threading import Lock

import numpy as np

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


@dataclass(frozen=True)
class BoxLocation:
    captured_at: float
    center: tuple
    yaw_rad: float  # direction of the first registered dimension
    box: BoxState
    display: dict
    from_window: bool = False


@dataclass(frozen=True)
class PushSearchSegment:
    start: tuple
    end: tuple
    valid_after: float
    expires_at: float
    hint: BoxLocation | None = None

    def contains(self, xy, tolerance):
        delta = np.subtract(self.end, self.start)
        offset = np.subtract(xy, self.start)
        t = np.clip(offset @ delta / max(delta @ delta, 1e-12), 0., 1.)
        return np.linalg.norm(offset - t*delta) <= tolerance


class BoxGuidancePose:
    """Choose a representative registered footprint for a stationary box's approach."""
    max_age = .8
    position_tolerance = .08
    angle_tolerance = math.radians(12.)

    def __init__(self):
        self.samples = deque(maxlen=5)
        self.box = None
        self.axes = self.centers = None
        self.index = 0

    def observe(self, box, stamp, *, window=False):
        if stamp is None or not math.isfinite(stamp) or box.local_dimensions is None or box.orientation is None:
            return
        if self.samples:
            latest, _, was_window = self.samples[-1]
            if stamp < latest or (stamp == latest and (window or not was_window)):
                return
            if stamp == latest:
                self.samples.pop()
        while self.samples and stamp - self.samples[0][0] > self.max_age:
            self.samples.popleft()
        self.samples.append((stamp, box, window))
        boxes = [sample[1] for sample in self.samples]
        self.centers = np.asarray([b.xy for b in boxes])
        q = np.asarray([b.orientation for b in boxes])
        yaw = np.arctan2(2*(q[:, 0]*q[:, 3] + q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2 + q[:, 3]**2))
        yaw += np.array([math.pi/2 if b.local_dimensions[1] > b.local_dimensions[0] else 0. for b in boxes])
        self.axes = np.column_stack((np.cos(yaw), np.sin(yaw)))
        # The medoid is an actual measured pose. Long-axis directions separated
        # by 180 degrees describe the same rectangular footprint.
        distances = np.linalg.norm(self.centers[:, None] - self.centers, axis=2)
        distances += max(box.local_dimensions[:2])/2 * np.abs(np.sin(yaw[:, None] - yaw))
        self.index = int(np.argmin(distances.sum(axis=1)))
        self.box = boxes[self.index]

    def recent(self, now):
        return self.box if self.samples and 0 <= now - self.samples[-1][0] <= self.max_age else None

    def moved_from(self, plan, now):
        if self.recent(now) is None or len(self.samples) < 3:
            return False
        # Require three recent captures supporting the same change relative to
        # the selected face. Re-reading a map or a fit cannot cast another vote.
        if now - self.samples[-3][0] > self.max_age:
            return False
        delta = self.centers - plan.initial_xy
        shift = delta[self.index]
        length = np.linalg.norm(shift)
        if length > self.position_tolerance:
            if np.all(delta[-3:] @ (shift/length) > self.position_tolerance):
                return True
        aligned = np.abs(self.axes @ plan.direction) >= math.cos(self.angle_tolerance)
        agreeing = np.abs(self.axes @ self.axes[self.index]) >= math.cos(self.angle_tolerance)
        return not aligned[self.index] and np.all(~aligned[-3:] & agreeing[-3:])


class BoxMapMemory:
    def __init__(self):
        self._locations = {}
        self._push_search = {}
        self._lock = Lock()

    def locations(self, now=math.inf):
        with self._lock:
            return {k: v for k, v in self._locations.items() if v.captured_at <= now}

    def reset_after_push(self, object_id, start, end, now, timeout):
        with self._lock:
            # The old orientation helps fit new returns but is no longer a mapped location.
            hint = self._locations.pop(object_id, None)
            self._push_search[object_id] = PushSearchSegment(tuple(start), tuple(end), now, now + timeout, hint)

    def reset_times(self):
        with self._lock:
            return {key: area.valid_after for key, area in self._push_search.items()}

    def push_searches(self, stamp):
        with self._lock:
            return {key: area for key, area in self._push_search.items()
                    if area.valid_after < stamp and key not in self._locations}

    def observe(self, delivery, now):
        if not delivery:
            return
        window = delivery.get('geometry_source') == 'window_two_faces'
        if not window and delivery.get('geometry_source') != 'current_lidar_cuboid_surfaces':
            return
        if delivery.get('frame', delivery.get('planning_frame', 'experiment')) != 'experiment':
            return
        for row in delivery.get('detections', ()):
            key, stamp = row.get('object_id'), row.get('captured_at')
            if (not key or not row.get('valid') or stamp is None
                    or not stamp <= now < row.get('expires_at', -math.inf)):
                continue
            if not (row.get('track_id') if window else row.get('geometry_verified')):
                continue
            dims = row.get('dimensions_m' if window else 'local_dimensions')
            center = row.get('center')
            if dims is None or len(dims) != 3 or center is None or len(center) != 3:
                continue
            long_axis = row.get('long_axis_rad', row.get('face_angle_rad') if dims[0] == dims[1] else None)
            axis_offset = math.pi/2 if dims[1] > dims[0] else 0.
            yaw = row.get('yaw_rad')
            if window and long_axis is not None:
                yaw = long_axis-axis_offset
            if yaw is None or min(dims) <= 0 or not all(math.isfinite(v) for v in (*dims, *center, yaw)):
                continue
            if not window:
                long_axis = yaw+axis_offset
            # The shared route planner expects a world AABB. Keep the measured
            # local orientation as well for the overlay and current-scan fitting.
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            envelope = (c*dims[0]+s*dims[1], s*dims[0]+c*dims[1], dims[2])
            if not window:
                envelope = tuple(row.get('dimensions', envelope))
            box = BoxState(key, tuple(center[:2]), envelope, mass_kg=None, local_dimensions=tuple(dims),
                           orientation=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)), center_z=center[2])
            display = dict(row, dimensions_m=dims, long_axis_rad=long_axis, object_id=key)
            with self._lock:
                reset = self._push_search.get(key)
                if reset and (stamp <= reset.valid_after or
                              (window and delivery.get('window_start_at', -math.inf) <= reset.valid_after)):
                    continue
                old = self._locations.get(key)
                if old is None or stamp > old.captured_at or (stamp == old.captured_at and not window):
                    self._locations[key] = BoxLocation(stamp, tuple(center), yaw, box, display, window)
