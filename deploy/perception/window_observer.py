"""Diagnostic-only temporal box tracking. This provider never supplies motion geometry."""
from dataclasses import dataclass
import math
import time

import numpy as np

from .registered_boxes import validate_tracking_config
from .window_fitting import WindowFit, find_window_boxes, refit_window_box
from .window_stream import WINDOW, WindowScanBuffer

INITIAL_POSITION_TOLERANCE = .04
INITIAL_YAW_TOLERANCE = math.radians(5.)
LOCAL_FIT_WINDOWS = (.20, .35)


@dataclass
class Measurement:
    fit: WindowFit
    support: float
    support_id: int
    end_id: int
    sensor_xy: np.ndarray
    sensor_yaw: float


class WindowTrack:
    def __init__(self):
        self.serial = 0
        self.anchor = None
        self.clear()

    def clear(self):
        self.last = self.seed = None
        self.votes = 0

    def expire(self, now):
        if self.last is not None and now-self.last.support >= WINDOW:
            self.clear()
        if self.seed is not None and now-self.seed.support > 2*WINDOW:
            self.clear()

    def search(self, cloud, labels, recent_starts, frame, row, first_id, stationary, cfg):
        if self.last is not None:
            dt = max(0., frame.captured_at-self.last.support)
            reach = cfg['association_margin_m'] + dt*cfg['max_tracking_speed_m_s']
            turn = math.radians(cfg['association_yaw_margin_deg']) + dt*cfg['max_tracking_yaw_rate_rad_s']
            # A moving box leaves several old outlines in a one-second window.
            # Prefer the newest measured outline; retain absolute scan indices for the two face stamps.
            for start, method in recent_starts:
                recent = refit_window_box(cloud[start:], labels[start:], frame.origin, row['dimensions_m'],
                                          cfg, self.last.fit, reach, turn, search_region=method == 'local_recent')
                if recent:
                    return method, recent
            return 'local', refit_window_box(cloud, labels, frame.origin, row['dimensions_m'], cfg,
                                             self.last.fit, reach, turn)
        if not stationary:
            return 'wait_for_stationary_window', []
        if self.seed is not None:
            if self.votes >= cfg['min_confirmations']:
                return 'local', refit_window_box(cloud, labels, frame.origin, row['dimensions_m'], cfg,
                                                 self.seed.fit, INITIAL_POSITION_TOLERANCE, INITIAL_YAW_TOLERANCE)
            if first_id <= self.seed.end_id:
                return 'await_distinct_window', []
        if self.anchor is not None:
            # After an occlusion the same box can be outside its startup region.
            # A different item elsewhere in that region cannot take its ID.
            return 'discovery', refit_window_box(cloud, labels, frame.origin, row['dimensions_m'], cfg,
                self.anchor, max(.08, cfg['association_margin_m']), math.radians(10.))
        return 'discovery', find_window_boxes(cloud, labels, frame.origin, row['dimensions_m'], row['search_region'], cfg)

    def update(self, match, reason, stationary, fresh, method, now, cfg):
        self.expire(now)
        if reason == 'ambiguous':
            self.clear()
        if match is None:
            if self.seed is not None:
                if stationary and reason == 'await_distinct_window':
                    return 'confirming', reason
                if not stationary:
                    self.clear()
            return ('missing' if self.last is not None else 'searching'), reason
        if self.last is not None:
            if not fresh:
                return 'missing', 'surface_evidence_expired'
            dt = match.support-self.last.support
            distance = np.linalg.norm(match.fit.center-self.last.fit.center)
            turn = match.fit.angle_to(self.last.fit)
            if (dt < 0 or distance > cfg['association_margin_m'] + dt*cfg['max_tracking_speed_m_s']
                    or turn > math.radians(cfg['association_yaw_margin_deg']) + dt*cfg['max_tracking_yaw_rate_rad_s']):
                return 'missing', 'discontinuous_fit'
            self.last = match
            self.anchor = match.fit
            return 'tracked', None
        if not stationary or now-match.support > 2*WINDOW:
            self.clear()
            return 'searching', 'wait_for_stationary_window' if not stationary else 'surface_evidence_expired'
        if self.seed is not None:
            difference = abs((match.sensor_yaw-self.seed.sensor_yaw+math.pi) % (2*math.pi)-math.pi)
            same_pose = (np.linalg.norm(match.sensor_xy-self.seed.sensor_xy) <= INITIAL_POSITION_TOLERANCE
                         and difference <= INITIAL_YAW_TOLERANCE)
            consistent = (np.linalg.norm(match.fit.center-self.seed.fit.center) <= INITIAL_POSITION_TOLERANCE
                          and match.fit.angle_to(self.seed.fit) <= INITIAL_YAW_TOLERANCE)
            if not same_pose or not consistent:
                self.clear()
        # Each discovery is a whole-registry search on a disjoint window.
        # A slow result can vote only as a stationary hypothesis. Publication
        # always requires a fit whose two measured faces are still fresh.
        if method == 'discovery':
            self.seed = match
            self.votes += 1
        if self.seed is None:
            return 'searching', 'no_two_face_fit'
        if self.votes < cfg['min_confirmations']:
            return 'confirming', 'await_distinct_window'
        if not fresh or (method == 'local' and match.support_id <= self.seed.end_id):
            return 'confirming', 'await_fresh_refit'
        self.last, self.seed = match, None
        self.anchor = match.fit
        self.serial += 1
        return 'tracked', None


class WindowBoxObserver:
    def __init__(self, config, *, live=False, world_boundary=None):
        self.config = validate_tracking_config(config)
        self.world_boundary = world_boundary
        self.live = live
        self.generation = 0
        self.closed = False
        self.reset()

    def reset(self):
        self.generation += 1
        self.buffer = WindowScanBuffer(self.config)
        self.tracks = {row['id']: WindowTrack() for row in self.config['boxes']}
        self.epoch = self.buffer.epoch
        self.last_delivery = self.last_error = None

    def invalidate(self, reason):
        self.buffer.clear(reason)
        for track in self.tracks.values():
            track.clear()
        self.last_error, self.last_delivery = reason, None

    def close(self):
        self.closed = True
        self.invalidate('closed')

    def publish(self, snapshot, now, sink, error=None):
        """Record the diagnostic alongside navigation, once per new result."""
        previous, previous_error = self.last_delivery, self.last_error
        if snapshot is None:
            self.invalidate(error or 'no_valid_observation')
        else:
            self.update(snapshot, now)
        if self.last_delivery is not None and self.last_delivery is not previous:
            sink.event('box_window_result', **self.last_delivery)
        elif self.last_error and self.last_error != previous_error:
            sink.event('box_window_result', detections=[], error=self.last_error)

    def update(self, snapshot, now):
        if self.closed:
            return (), None
        window = self.buffer.append(snapshot, now)
        if window is not None:
            self.evaluate(window, now)
        return (), None

    def evaluate(self, window, now):
        if window.epoch != self.epoch:
            for track in self.tracks.values():
                track.clear()
            self.epoch = window.epoch
        if window.error:
            self.last_error, self.last_delivery = window.error, None
            return
        begun = time.monotonic()
        frames = window.frames
        first, (frame, yaw) = frames[0][0], frames[-1]
        stamp = frame.captured_at
        stationary = all(np.linalg.norm(f.origin[:2]-frame.origin[:2]) <= INITIAL_POSITION_TOLERANCE
                         and abs((angle-yaw+math.pi) % (2*math.pi)-math.pi) <= INITIAL_YAW_TOLERANCE
                         for f, angle in frames)
        matches, methods = {}, {}
        if window.ready:
            cloud = np.concatenate([f.points for f, _ in frames])
            labels = np.repeat(np.arange(len(frames)), [len(f.points) for f, _ in frames])
            short, recent = (sum(len(f.points) for f, _ in frames if stamp-f.captured_at > span)
                             for span in LOCAL_FIT_WINDOWS)
            recent_starts = []
            if short > recent:
                recent_starts.append((short, 'local_short'))
            if recent:
                recent_starts.append((recent, 'local_recent'))
            for row in self.config['boxes']:
                key = row['id']
                track = self.tracks[key]
                track.expire(now)
                methods[key], matches[key] = track.search(cloud, labels, recent_starts, frame, row, first.frame_id,
                                                         stationary, self.config)
        finished = now + (time.monotonic()-begun if self.live else 0.)
        detections = []
        for row in self.config['boxes']:
            key = row['id']
            track = self.tracks[key]
            candidates = matches.get(key, [])
            match = None
            fresh = False
            method = methods.get(key, 'filling_window')
            reason = 'no_two_face_fit' if method in ('discovery', 'local', 'local_short', 'local_recent') else method
            if len(candidates) > 1:
                reason = 'ambiguous'
            elif candidates:
                fit = candidates[0]
                conflict = any(np.linalg.norm(fit.center-other.center) < .075
                               for other_id, others in matches.items() if other_id != key for other in others)
                support_frame = frames[fit.support_index][0]
                support = support_frame.captured_at
                fresh = finished < support+self.config['max_age_s']
                long_axis = fit.long_axis_rad if fit.long_axis_rad is not None else fit.face_angle
                sides = sorted(row['dimensions_m'][:2], reverse=True)
                if self.world_boundary is not None and not self.world_boundary.contains_box(
                        fit.center, sides, long_axis, self.config['isolation_margin_m']):
                    reason = 'box_at_world_boundary'
                elif conflict:
                    reason = 'ambiguous'
                elif not fresh and (method != 'discovery' or stamp >= support+self.config['max_age_s']):
                    reason = 'surface_evidence_expired'
                else:
                    x0, x1, y0, y1 = row['search_region']
                    in_registry = x0 <= fit.center[0] <= x1 and y0 <= fit.center[1] <= y1
                    active = track.last is not None and finished-track.last.support < WINDOW
                    if active or track.anchor is not None or in_registry:
                        match = Measurement(fit, support, support_frame.frame_id, frame.frame_id, frame.origin[:2], yaw)
                    else:
                        reason = 'outside_registration_region'
            state, reason = track.update(match, reason, stationary, fresh, method, finished, self.config)
            record = dict(object_id=key, valid=state=='tracked', tracking_state=state, invalid_reason=reason,
                          track_id=(f'{self.generation}:{key}:{track.serial}' if track.last is not None else None),
                          geometry_verified=False, push_allowed=False, candidate_count=len(candidates), search=method)
            if state == 'tracked':
                record.update(center=[*map(float, match.fit.center), row['dimensions_m'][2]/2],
                              dimensions_m=row['dimensions_m'],
                              face_angle_rad=match.fit.face_angle, angle_period_deg=90,
                              surfaces=match.fit.surfaces, fit_residual_m=match.fit.residual, captured_at=match.support,
                              expires_at=match.support+self.config['max_age_s'])
                if match.fit.long_axis_rad is not None:
                    record['long_axis_rad'] = match.fit.long_axis_rad
            detections.append(record)
        self.last_error = None
        self.last_delivery = dict(geometry_source='window_two_faces', frame='experiment', scan_id=frame.frame_id,
                                  captured_at=stamp, fit_started_at=now, received_at=finished,
                                  window_start_at=first.captured_at,
                                  window_scan_count=len(frames), session_id=self.generation,
                                  detections=detections)
