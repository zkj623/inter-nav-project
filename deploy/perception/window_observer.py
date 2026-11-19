"""Diagnostic-only temporal box tracking. This provider never supplies motion geometry."""
from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from .registered_boxes import RegisteredScanReader, validate_tracking_config
from .window_fitting import WindowFit, face_angle_difference, find_window_boxes

WINDOW = 1.
INITIAL_POSITION_TOLERANCE = .04
INITIAL_YAW_TOLERANCE = math.radians(5.)


@dataclass
class Measurement:
    fit: WindowFit
    support: float
    end_id: int
    sensor_xy: np.ndarray
    sensor_yaw: float


class WindowTrack:
    def __init__(self):
        self.serial = 0
        self.clear()

    def clear(self):
        self.last = self.seed = None
        self.votes = 0

    def region(self, row, now, cfg):
        if self.last is None or now-self.last.support >= WINDOW:
            return row['search_region']
        # Search far enough to see competing nearby boxes and a box leaving its
        # initial registration region. The tighter innovation gate applies below.
        radius = max(.40, cfg['max_tracking_speed_m_s']*WINDOW + cfg['association_margin_m'])
        x, y = self.last.fit.center
        return (x-radius, x+radius, y-radius, y+radius)

    def update(self, match, reason, first_id, stationary, now, cfg):
        if self.last is not None and now-self.last.support >= WINDOW:
            self.clear()
        if reason == 'ambiguous':
            self.clear()
        if match is None:
            if self.seed is not None and (not stationary or now-self.seed.support > 2*WINDOW):
                self.seed = None
                self.votes = 0
            return ('missing' if self.last is not None else 'searching'), reason
        if self.last is not None:
            dt = match.support-self.last.support
            distance = np.linalg.norm(match.fit.center-self.last.fit.center)
            turn = face_angle_difference(match.fit.face_angle, self.last.fit.face_angle)
            if (dt < 0 or distance > cfg['association_margin_m'] + dt*cfg['max_tracking_speed_m_s']
                    or turn > math.radians(cfg['association_yaw_margin_deg']) + dt*cfg['max_tracking_yaw_rate_rad_s']):
                return 'missing', 'discontinuous_fit'
            self.last = match
            return 'tracked', None
        if not stationary:
            self.seed = None
            self.votes = 0
            return 'searching', 'wait_for_stationary_window'
        if self.seed is not None:
            difference = abs((match.sensor_yaw-self.seed.sensor_yaw+math.pi) % (2*math.pi)-math.pi)
            same_pose = (np.linalg.norm(match.sensor_xy-self.seed.sensor_xy) <= INITIAL_POSITION_TOLERANCE
                         and difference <= INITIAL_YAW_TOLERANCE)
            consistent = (np.linalg.norm(match.fit.center-self.seed.fit.center) <= INITIAL_POSITION_TOLERANCE
                          and face_angle_difference(match.fit.face_angle, self.seed.fit.face_angle) <= INITIAL_YAW_TOLERANCE)
            if not same_pose or not consistent or now-self.seed.support > 2*WINDOW:
                self.seed = None
                self.votes = 0
        if self.seed is None or first_id > self.seed.end_id:
            self.seed = match
            self.votes += 1
        if self.votes < cfg['min_confirmations']:
            return 'confirming', 'await_distinct_window'
        self.last, self.seed = match, None
        self.serial += 1
        return 'tracked', None


class WindowBoxObserver:
    def __init__(self, config, *, live=False):
        self.config = validate_tracking_config(config)
        self.live = live
        self.generation = 0
        self.closed = False
        self.reset()

    def reset(self):
        self.generation += 1
        self.reader = RegisteredScanReader(self.config)
        self.frames = deque()
        self.tracks = {row['id']: WindowTrack() for row in self.config['boxes']}
        self.started = self.last_now = None
        self.last_delivery = self.last_error = None

    def invalidate(self, reason):
        self.frames.clear()
        self.started = None
        for track in self.tracks.values():
            track.clear()
        self.last_error, self.last_delivery = reason, None

    def close(self):
        self.closed = True
        self.invalidate('closed')

    def update(self, snapshot, now):
        if self.closed:
            return (), None
        begun = time.monotonic()
        try:
            if not math.isfinite(now) or self.last_now is not None and now < self.last_now:
                raise ValueError('Window observation clock moved backwards or is invalid')
            self.last_now = now
            frame = self.reader.capture(snapshot, now, self.generation)
            if frame is None:
                # A repeated scan earns no confirmation and never extends expiry.
                return (), None
            q = np.asarray(snapshot.observation['orientation'], float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
                raise ValueError('Window observation requires a valid robot orientation')
            w, x, y, z = q / np.linalg.norm(q)
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        except (KeyError, TypeError, ValueError) as exc:
            self.invalidate(str(exc))
            return (), None

        stamp = frame.captured_at
        if self.frames and stamp-self.frames[-1][0].captured_at >= self.config['max_age_s']:
            self.invalidate('scan_gap')
        if self.started is None:
            self.started = stamp
        while self.frames and self.frames[0][0].captured_at <= stamp-WINDOW:
            self.frames.popleft()
        self.frames.append((frame, yaw))
        ready = stamp-self.started >= WINDOW
        matches = {}
        if ready:
            cloud = np.concatenate([f.points for f, _ in self.frames])
            labels = np.repeat(np.arange(len(self.frames)), [len(f.points) for f, _ in self.frames])
            for row in self.config['boxes']:
                track = self.tracks[row['id']]
                region = track.region(row, now, self.config)
                matches[row['id']] = find_window_boxes(cloud, labels, frame.origin,
                                                       row['dimensions_m'], region, self.config)
        finished = now + (time.monotonic()-begun if self.live else 0.)
        stationary = all(np.linalg.norm(f.origin[:2]-frame.origin[:2]) <= INITIAL_POSITION_TOLERANCE
                         and abs((angle-yaw+math.pi) % (2*math.pi)-math.pi) <= INITIAL_YAW_TOLERANCE
                         for f, angle in self.frames)
        first = self.frames[0][0]
        detections = []
        for row in self.config['boxes']:
            key = row['id']
            track = self.tracks[key]
            candidates = matches.get(key, [])
            match = None
            reason = 'no_two_face_fit' if ready else 'filling_window'
            if len(candidates) > 1:
                reason = 'ambiguous'
            elif candidates:
                fit = candidates[0]
                conflict = any(np.linalg.norm(fit.center-other.center) < .075
                               for other_id, others in matches.items() if other_id != key for other in others)
                support = self.frames[fit.support_index][0].captured_at
                if conflict:
                    reason = 'ambiguous'
                elif finished >= support+self.config['max_age_s']:
                    reason = 'surface_evidence_expired'
                else:
                    x0, x1, y0, y1 = row['search_region']
                    in_registry = x0 <= fit.center[0] <= x1 and y0 <= fit.center[1] <= y1
                    active = track.last is not None and finished-track.last.support < WINDOW
                    if active or in_registry:
                        match = Measurement(fit, support, frame.frame_id, frame.origin[:2], yaw)
                    else:
                        reason = 'outside_registration_region'
            state, reason = track.update(match, reason, first.frame_id, stationary, finished, self.config)
            record = dict(object_id=key, valid=state=='tracked', tracking_state=state, invalid_reason=reason,
                          track_id=(f'{self.generation}:{key}:{track.serial}' if track.last is not None else None),
                          geometry_verified=False, push_allowed=False, candidate_count=len(candidates))
            if state == 'tracked':
                record.update(center=[*map(float, match.fit.center), row['dimensions_m'][2]/2],
                              face_angle_rad=match.fit.face_angle, angle_period_deg=90,
                              fit_residual_m=match.fit.residual, captured_at=match.support,
                              expires_at=match.support+self.config['max_age_s'])
            detections.append(record)
        self.last_error = None
        self.last_delivery = dict(geometry_source='window_two_faces', scan_id=frame.frame_id,
                                  captured_at=stamp, received_at=finished, window_start_at=first.captured_at,
                                  window_scan_count=len(self.frames), session_id=self.generation,
                                  detections=detections)
        return (), None
