"""Registered-size LiDAR perception adapter for the existing deploy pipeline.

Registry metadata supplies identity, complete dimensions and push permission.
Only a fresh measured cuboid fit supplies pose. One worker and one replaceable
pending scan keep fitting off the navigation/writer thread. Losing an established
identity requires reset/reassociation; registration never renews stale geometry.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
import queue
import threading
import time

import numpy as np

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from .box_fitting import scan_candidates, scan_tracked_candidates


DEFAULT_TRACKING = {
    'max_age_s': .25,
    'min_points': 30,
    'min_face_points': 8,
    'min_confirmations': 2,
    'residual_threshold_m': .025,
    'extent_tolerance_m': .065,
    'height_tolerance_m': .08,
    'min_height_fraction': .70,
    'min_face_coverage': .80,
    'min_inlier_fraction': .90,
    'floor_clearance_m': .025,
    'geometry_margin_m': .03,
    'cluster_cell_m': .06,
    'isolation_margin_m': .08,
    'angle_step_deg': 3.,
    'max_tracking_speed_m_s': .60,
    'max_tracking_yaw_rate_rad_s': 1.0,
    'association_margin_m': .03,
    'association_yaw_margin_deg': 5.,
    'max_scan_points': 120000,
    'max_candidates': 16,
    'min_boundary_rays': 3,
    'boundary_ray_min_gap_m': .015,
    'boundary_ray_max_gap_m': .08,
    'boundary_ray_depth_margin_m': .04,
    'boundary_ray_height_fraction': .25,
}


def validate_tracking_config(config):
    """Return an independent normalized config; reject coercive bool/NaN input."""
    if not isinstance(config, dict):
        raise ValueError('interaction tracking must be a mapping')
    unknown = set(config) - set(DEFAULT_TRACKING) - {'boxes', 'geometry_verified', 'registry_reference'}
    if unknown:
        raise ValueError('Unknown tracking settings: ' + ', '.join(sorted(unknown)))
    result = {**DEFAULT_TRACKING, **config}
    integer_names = {'min_points', 'min_face_points', 'min_confirmations', 'max_scan_points', 'max_candidates', 'min_boundary_rays'}
    for name, default in DEFAULT_TRACKING.items():
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        if name in integer_names:
            if not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        else:
            result[name] = float(value)
    if not .02 <= result['max_age_s'] <= .75:
        raise ValueError('max_age_s must be between .02 and .75 seconds')
    if (result['min_points'] < 12 or result['min_face_points'] < 4
            or not 2 <= result['min_confirmations'] <= 10
            or not 100 <= result['max_scan_points'] <= 500000
            or not 1 <= result['max_candidates'] <= 32
            or not 3 <= result['min_boundary_rays'] <= 30):
        raise ValueError('Invalid tracking point/candidate/confirmation bounds')
    for name in ('min_height_fraction', 'min_face_coverage', 'min_inlier_fraction'):
        if not .6 <= result[name] <= 1:
            raise ValueError(f'{name} must be between .6 and 1')
    for name in ('residual_threshold_m', 'extent_tolerance_m', 'height_tolerance_m',
                 'floor_clearance_m', 'geometry_margin_m', 'cluster_cell_m',
                 'isolation_margin_m', 'association_margin_m', 'boundary_ray_min_gap_m',
                 'boundary_ray_max_gap_m', 'boundary_ray_depth_margin_m'):
        if result[name] > .20:
            raise ValueError(f'{name} must not exceed .20 m')
    if (not .1 <= result['boundary_ray_height_fraction'] <= .8
            or result['boundary_ray_min_gap_m'] >= result['boundary_ray_max_gap_m']):
        raise ValueError('Invalid single-face boundary ray proof limits')
    if not 1 <= result['angle_step_deg'] <= 10:
        raise ValueError('angle_step_deg must be between 1 and 10')
    if (result['max_tracking_speed_m_s'] > 2 or result['max_tracking_yaw_rate_rad_s'] > 4
            or result['association_yaw_margin_deg'] > 15):
        raise ValueError('Tracking continuity limits exceed supported bounds')
    verified = result.setdefault('geometry_verified', False)
    if not isinstance(verified, bool):
        raise ValueError('geometry_verified must be an explicit boolean')
    reference = result.setdefault('registry_reference', '')
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError('registry_reference must name the measured box registry')
    rows = result.get('boxes')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError('boxes must contain between 1 and 8 registered boxes')
    normalized, ids = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id', 'dimensions_m', 'search_region', 'push_allowed'}:
            raise ValueError('Each registered box requires id, dimensions_m, search_region and push_allowed')
        object_id = row['id']
        if not isinstance(object_id, str) or not object_id.strip() or len(object_id) > 80 or object_id in ids:
            raise ValueError('Registered box id must be nonempty, unique and at most 80 characters')
        ids.add(object_id)
        if not isinstance(row['push_allowed'], bool):
            raise ValueError('push_allowed must be an explicit boolean')
        values = {}
        for name, length in [('dimensions_m', 3), ('search_region', 4)]:
            raw = row[name]
            if (not isinstance(raw, (list, tuple)) or len(raw) != length
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw)):
                raise ValueError(f'{name} requires {length} finite numeric values')
            values[name] = list(map(float, raw))
        dims, region = values['dimensions_m'], values['search_region']
        if any(not .15 <= d <= 2.0 for d in dims):
            raise ValueError('Registered dimensions must be between .15 and 2 m')
        if region[0] >= region[1] or region[2] >= region[3]:
            raise ValueError('search_region requires xmin < xmax and ymin < ymax')
        if max(region[1]-region[0], region[3]-region[2]) > 10:
            raise ValueError('Initial box search_region must be no wider than 10 m')
        normalized.append({'id': object_id, **values, 'push_allowed': row['push_allowed']})
    result['boxes'] = normalized
    return result


@dataclass(frozen=True)
class _Scan:
    generation: int
    frame_id: int
    captured_at: float
    points: np.ndarray
    origin: np.ndarray
    source_stamp: object
    hints: dict


class RegisteredBoxPerception:
    source = 'registered_box_lidar'

    def __init__(self, config, clock=time.monotonic, synchronous=False):
        self.config = validate_tracking_config(config)
        self.clock = clock
        self.synchronous = bool(synchronous)
        self.geometry_verified = self.config['geometry_verified']
        self.registry = {row['id']: row for row in self.config['boxes']}
        self.source_reference = self.config['registry_reference']
        payload = {'boxes': sorted(self.config['boxes'], key=lambda row: row['id']),
                   'registry_reference': self.source_reference}
        self.source_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.jobs, self.results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.generation = 0
        self._new_state()
        self.thread = None
        if not self.synchronous:
            self.thread = threading.Thread(target=self._work, daemon=True, name='registered-box-lidar')
            self.thread.start()

    def _new_state(self):
        self.last_delivery = self.last_error = self.last_success = None
        self.last_query_status = 'idle'
        self._tracks, self._visible = {}, {}
        self._last_submitted = self._last_accepted = None
        self._last_time = -math.inf
        self._coordinate_transform = None

    @staticmethod
    def _clear(channel):
        while True:
            try:
                channel.get_nowait()
            except queue.Empty:
                return

    @classmethod
    def _replace(cls, channel, value):
        cls._clear(channel)
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass

    def reset(self):
        self.generation += 1
        self._clear(self.jobs)
        self._clear(self.results)
        self._new_state()

    def close(self):
        self.closed.set()
        self.generation += 1
        self._clear(self.jobs)
        self._visible.clear()
        if self.thread is not None:
            self.thread.join(timeout=.5)

    def _invalidate(self, reason):
        self._visible.clear()
        for track in self._tracks.values():
            if track['confirmed']:
                track['lost'] = True
                track['reason'] = reason
        self.last_error, self.last_query_status = reason, 'invalid_lidar'

    def _capture(self, snapshot, now):
        if not snapshot.geometry_calibrated or snapshot.planning_frame != 'experiment':
            raise ValueError('Registered boxes require calibrated experiment-frame geometry')
        transform = json.dumps(snapshot.coordinate_transform, sort_keys=True)
        if self._coordinate_transform is not None and self._coordinate_transform != transform:
            raise ValueError('Experiment transform changed; reset tracking and reassociate identities')
        lidar = snapshot.observation['sensors']['lidar']
        if lidar.get('pointcloud_frame', 'experiment') != 'experiment':
            raise ValueError('LiDAR points must be in experiment coordinates')
        stamp = float(snapshot.lidar_at)
        age = now-stamp
        pose_age = now-float(snapshot.pose_at)
        if not math.isfinite(age) or not stamp <= now < stamp+self.config['max_age_s']:
            raise ValueError('LiDAR capture is stale, future or invalid')
        if not math.isfinite(pose_age) or not float(snapshot.pose_at) <= now < float(snapshot.pose_at)+self.config['max_age_s']:
            raise ValueError('LiDAR tracking requires fresh robot pose feedback')
        frame_id = lidar['physics_step']
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0:
            raise ValueError('LiDAR physics_step must be a nonnegative source scan identifier')
        frame_id = int(frame_id)
        if self._last_submitted is not None:
            previous_id, previous_stamp = self._last_submitted
            if frame_id == previous_id and stamp == previous_stamp:
                return None  # repeated polling cannot renew geometry or votes
            if frame_id <= previous_id or stamp <= previous_stamp:
                raise ValueError('Non-increasing LiDAR scan id or capture time')
        points = np.asarray(lidar['pointcloud'], dtype=float)
        origin = np.asarray(lidar['position'], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or len(points) > self.config['max_scan_points']
                or origin.shape != (3,) or not np.isfinite(origin).all()):
            raise ValueError('Invalid LiDAR points or paired origin')
        maximum = lidar.get('max_range')
        if maximum is not None:
            if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum <= 0:
                raise ValueError('Invalid LiDAR maximum range')
            # Match navigation: range-limit/no-hit shells cannot prove the
            # independent background returns needed by a bounded-face fit.
            points = points[np.linalg.norm(points-origin, axis=1) < .99*maximum]
        self._coordinate_transform = transform
        self._last_submitted = (frame_id, stamp)
        return _Scan(self.generation, frame_id, stamp, points.copy(), origin.copy(),
                     lidar.get('source_stamp', frame_id), self._tracking_hints(stamp))

    def _tracking_hints(self, stamp):
        return {key: track['fit'] for key, track in self._tracks.items()
                if track['confirmed'] and not track['lost']
                and track['captured_at'] < stamp < track['captured_at']+self.config['max_age_s']}

    def _fit(self, frame):
        # Share fitting between registry entries with identical dimensions.
        cache, result = {}, {}
        for object_id, row in self.registry.items():
            key = tuple(row['dimensions_m'])
            if key not in cache:
                cache[key] = scan_candidates(frame.points, key, frame.origin, self.config)
            candidates = list(cache[key])
            if object_id in frame.hints:
                bounded = scan_tracked_candidates(frame.points, key, frame.origin, self.config, frame.hints[object_id])
                for fit in bounded:
                    if not any(len(set(fit.indices).intersection(other.indices)) > .25*min(len(fit.indices), len(other.indices))
                               for other in candidates):
                        candidates.append(fit)
            result[object_id] = tuple(candidates)
        return result

    def _work(self):
        while not self.closed.is_set():
            try:
                frame = self.jobs.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                detections, error = self._fit(frame), None
            except Exception as exc:
                detections, error = {}, f'{type(exc).__name__}: {exc}'
            if not self.closed.is_set() and frame.generation == self.generation:
                self._replace(self.results, (frame, detections, error))

    def update(self, snapshot, now):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('Tracking update time must be finite')
        if self.closed.is_set():
            return (), self
        if now < self._last_time:
            self._invalidate('Tracking clock moved backwards; reset required')
            return (), self
        self._last_time = now
        # Capture validation precedes result delivery: changed frames or invalid
        # observations cannot resurrect a queued fit from the old frame.
        try:
            frame = self._capture(snapshot, now)
        except (KeyError, ValueError, TypeError) as exc:
            self._invalidate(str(exc))
            return (), self
        self._expire(now)
        if frame is not None and self.synchronous:
            try:
                detections, error = self._fit(frame), None
            except Exception as exc:
                detections, error = {}, f'{type(exc).__name__}: {exc}'
            self._replace(self.results, (frame, detections, error))
        try:
            completed, detections, error = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            if completed.generation == self.generation:
                if error:
                    self._invalidate(error)
                elif not completed.captured_at <= now < completed.captured_at+self.config['max_age_s']:
                    self._invalidate('Discarded stale or future fitting result')
                elif self._last_accepted is not None and completed.frame_id <= self._last_accepted:
                    self._invalidate('Discarded out-of-order fitting result')
                else:
                    self._accept(completed, detections, now)
        if frame is not None and not self.synchronous:
            self._replace(self.jobs, replace(frame, hints=self._tracking_hints(frame.captured_at)))
        visible = tuple(track['box'] for track in self._visible.values()) if self.geometry_verified else ()
        return visible, self

    def _expire(self, now):
        for object_id, track in list(self._visible.items()):
            if not track['captured_at'] <= now < track['captured_at']+self.config['max_age_s']:
                self._visible.pop(object_id)
                if track['confirmed']:
                    track['lost'], track['reason'] = True, 'Geometry expired; reset required'

    @staticmethod
    def _yaw_distance(a, b, dimensions):
        period = math.pi/2 if abs(dimensions[0]-dimensions[1]) < 1e-6 else math.pi
        return abs((a-b+period/2) % period-period/2)

    def _accept(self, frame, detections, now):
        proposals, reasons = {}, {}
        for object_id, row in self.registry.items():
            previous = self._tracks.get(object_id)
            if previous and previous['lost']:
                reasons[object_id] = previous['reason']
                continue
            choices = []
            for fit in detections.get(object_id, ()):
                if previous is None:
                    x0, x1, y0, y1 = row['search_region']
                    eligible = x0 <= fit.center[0] <= x1 and y0 <= fit.center[1] <= y1
                else:
                    dt = frame.captured_at-previous['captured_at']
                    distance = np.linalg.norm(np.asarray(fit.center[:2])-previous['fit'].center[:2])
                    angle = self._yaw_distance(fit.yaw_rad, previous['fit'].yaw_rad, row['dimensions_m'])
                    eligible = (dt > 0 and dt <= self.config['max_age_s']
                        and distance <= self.config['association_margin_m']+dt*self.config['max_tracking_speed_m_s']
                        and angle <= math.radians(self.config['association_yaw_margin_deg'])+dt*self.config['max_tracking_yaw_rate_rad_s'])
                if eligible:
                    choices.append(fit)
            if len(choices) == 1:
                proposals[object_id] = choices[0]
            else:
                reasons[object_id] = 'Ambiguous box identity' if choices else 'No fully constrained visible box'
        # A cluster cannot be claimed by two registry identities, even if they
        # have different dimensions or overlapping initialization regions.
        conflicts = set()
        items = list(proposals.items())
        for index, (left_id, left) in enumerate(items):
            left_indices = set(left.indices)
            for right_id, right in items[index+1:]:
                overlap = len(left_indices.intersection(right.indices))
                if overlap > .25*min(len(left.indices), len(right.indices)):
                    conflicts.update((left_id, right_id))
        for object_id in conflicts:
            proposals.pop(object_id)
            reasons[object_id] = 'Candidate claimed by multiple registered identities'
        visible, records = {}, []
        for object_id, row in self.registry.items():
            fit, previous = proposals.get(object_id), self._tracks.get(object_id)
            if fit is None:
                if previous:
                    previous['lost'] = previous['confirmed']
                    previous['reason'] = reasons.get(object_id, 'Lost observation')
                    if not previous['confirmed']:
                        self._tracks.pop(object_id)
                records.append({'object_id': object_id, 'valid': False,
                                'invalid_reason': reasons.get(object_id, 'No valid measurement')})
                continue
            # Rectangles are invariant under 180 degrees; squares under 90.
            # Choose the equivalent pose nearest the previous measurement so
            # symmetric-face changes do not fabricate a large box rotation.
            period = math.pi/2 if abs(fit.dimensions[0]-fit.dimensions[1]) < 1e-6 else math.pi
            reference_yaw = 0. if previous is None else previous['fit'].yaw_rad
            pose_yaw = reference_yaw+(fit.yaw_rad-reference_yaw+period/2) % period-period/2
            rotation_change = int(round((pose_yaw-fit.yaw_rad)/(math.pi/2)))
            if abs(pose_yaw-fit.yaw_rad) > 1e-8:
                c, s = math.cos(pose_yaw), math.sin(pose_yaw)
                sensor_local = (frame.origin[:2]-np.asarray(fit.center[:2])) @ np.array([[c, -s], [s, c]])
                observed_axes = (0, 1) if fit.measured_face_axis is None else ((fit.measured_face_axis+rotation_change) % 2,)
                face_names = tuple(('+' if sensor_local[i] > 0 else '-')+('x', 'y')[i] for i in observed_axes)
                spans = fit.face_spans[::-1] if rotation_change % 2 else fit.face_spans
                fit = replace(fit, yaw_rad=pose_yaw, visible_faces=face_names, face_spans=spans,
                              measured_face_axis=None if fit.measured_face_axis is None else observed_axes[0])
            confirmations = 1 if previous is None else previous['confirmations']+1
            confirmed = confirmations >= self.config['min_confirmations']
            c, s = abs(math.cos(fit.yaw_rad)), abs(math.sin(fit.yaw_rad))
            length, width, height = fit.dimensions
            envelope = (c*length+s*width+2*fit.uncertainty_m,
                        s*length+c*width+2*fit.uncertainty_m, height+2*fit.uncertainty_m)
            box = BoxState(object_id, fit.center[:2], envelope, yaw=0., mass_kg=None,
                           local_dimensions=fit.dimensions,
                           orientation=(math.cos(fit.yaw_rad/2), 0., 0., math.sin(fit.yaw_rad/2)),
                           center_z=fit.center[2])
            evidence = [] if previous is None else list(previous['evidence'])
            evidence.append(f'lidar:{self.generation}:{frame.frame_id}')
            track = {'box': box, 'fit': fit, 'captured_at': frame.captured_at,
                     'confirmed': confirmed, 'confirmations': confirmations, 'lost': False,
                     'reason': None, 'evidence': tuple(evidence[-self.config['min_confirmations']:])}
            self._tracks[object_id] = track
            if confirmed:
                visible[object_id] = track
            records.append({'object_id': object_id, 'valid': confirmed,
                'invalid_reason': None if confirmed else 'Awaiting distinct confirming scans',
                'center': fit.center, 'dimensions': envelope, 'local_dimensions': fit.dimensions,
                'orientation': box.orientation, 'yaw_rad': fit.yaw_rad,
                'captured_at': frame.captured_at, 'expires_at': frame.captured_at+self.config['max_age_s'],
                'scan_id': frame.frame_id, 'visible_faces': fit.visible_faces,
                'face_spans_m': fit.face_spans, 'fit_residual_m': fit.residual_m,
                'pose_uncertainty_m': fit.uncertainty_m, 'point_count': fit.point_count,
                'dimensions_source': 'measured_registry', 'height_source': 'registered_height_on_calibrated_floor',
                'geometry_source': fit.geometry_source, 'geometry_verified': self.geometry_verified,
                'boundary_free_ray_counts': fit.boundary_rays, 'measured_face_axis': fit.measured_face_axis,
                'hidden_thickness_source': 'measured_registry' if fit.measured_face_axis is not None else None,
                'permission_source': self.source, 'source_reference': self.source_reference,
                'source_digest': self.source_digest, 'push_allowed': row['push_allowed']})
        self._visible = visible
        self._last_accepted = frame.frame_id
        self.last_success, self.last_error, self.last_query_status = now, None, 'ok'
        self.last_delivery = {'frame_id': str(frame.frame_id), 'scan_id': frame.frame_id,
            'source_stamp': frame.source_stamp, 'captured_at': frame.captured_at, 'received_at': now,
            'planning_frame': 'experiment', 'session_id': self.generation,
            'geometry_source': 'current_lidar_cuboid_surfaces', 'geometry_verified': self.geometry_verified,
            'permission_source': self.source, 'source_reference': self.source_reference,
            'source_digest': self.source_digest, 'detections': records}

    def expires_at(self, object_id):
        track = self._visible.get(object_id)
        return None if track is None or track['lost'] else track['captured_at']+self.config['max_age_s']

    def permission(self, object_id, now):
        track = self._visible.get(object_id)
        if (self.closed.is_set() or not self.geometry_verified or track is None or track['lost']
                or not self.registry[object_id]['push_allowed']
                or not math.isfinite(now) or not track['captured_at'] <= now < track['captured_at']+self.config['max_age_s']):
            return None
        return InteractionPermission(object_id, 'box', 'push_aside', track['evidence'], (), float(now),
                                     self.source, self.source_reference, self.source_digest)

    def eligible_ids(self, now):
        return frozenset(key for key in self.registry if self.permission(key, now) is not None)

    def snapshot(self, now):
        return {object_id: {'category': 'box', 'confirmed': self.permission(object_id, now) is not None,
                    'frames': self._tracks.get(object_id, {}).get('confirmations', 0),
                    'latest_at': self._tracks.get(object_id, {}).get('captured_at'),
                    'source': self.source, 'source_reference': self.source_reference,
                    'source_digest': self.source_digest,
                    'invalid_reason': self._tracks.get(object_id, {}).get('reason'),
                    'geometry_verified': self.geometry_verified,
                    'expires_at': self.expires_at(object_id)} for object_id in self.registry}
