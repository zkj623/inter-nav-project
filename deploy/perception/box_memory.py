"""Confirmed box locations in one experiment frame. Contact evidence lives in the scan provider."""
from dataclasses import dataclass
import math
from threading import Lock

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


@dataclass(frozen=True)
class BoxLocation:
    captured_at: float
    center: tuple
    yaw_rad: float  # direction of the first registered dimension
    box: BoxState
    display: dict


class BoxMapMemory:
    def __init__(self):
        self._locations = {}
        self._lock = Lock()

    def locations(self, now=math.inf):
        with self._lock:
            return {k: v for k, v in self._locations.items() if v.captured_at <= now}

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
                old = self._locations.get(key)
                if old is None or stamp > old.captured_at or (stamp == old.captured_at and not window):
                    self._locations[key] = BoxLocation(stamp, tuple(center), yaw, box, display)
