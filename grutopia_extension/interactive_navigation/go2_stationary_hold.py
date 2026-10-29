"""Supported joint-target holding with a gradual return to the walking actor.

This changes joint position targets only. Actor inference and the explicit DC
motor remain active; rigid poses, velocities, gains and collisions are untouched.
"""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate


def stationary_hold_permitted(phase, skill_state=None, recovery_from_contact=True, *, healthy=True):
    """Permit free-space waiting, never an active contact or withdrawal stop."""
    if not healthy:
        return False
    if phase != 'skill':
        return phase in {'start', 'decide', 'navigate', 'approach', 'cross_passage',
                         'inspect_navigate', 'inspect_turn', 'inspect_wait'}
    if skill_state == 'brake_for_recovery':
        return not recovery_from_contact
    return skill_state in {'prepare', 'navigate_to_stage', 'settle_at_stage', 'turn_at_stage',
                           'settle_alignment', 'align_longitudinal', 'align_lateral',
                           'refine_heading', 'settle_heading', 'settle_contact',
                           'recover_preparation', 'settle_recovered_preparation', 'reobserve'}


class StationaryJointHold:
    """Capture a supported target after a measured stop, release on new intent."""
    def __init__(self, hold_seconds=.4, release_seconds=.4, ground_height=.01):
        if not all(math.isfinite(v) and v > 0 for v in [hold_seconds, release_seconds]):
            raise ValueError('Stationary hold timing must be finite and positive')
        self.hold_seconds = float(hold_seconds)
        self.release_seconds = float(release_seconds)
        self.ground_height = float(ground_height)
        if not math.isfinite(self.ground_height):
            raise ValueError('Stationary hold ground height must be finite')
        self.stop = MeasuredStopGate(self.hold_seconds)
        self.reset()

    def reset(self):
        self.stop.reset()
        self.state = 'tracking'
        self.held_target = None
        self.held_since = None
        self.release_started = None
        self.last_time = None
        self.release_fraction = 1.
        self.capture_count = 0
        self.release_count = 0
        self.reason = 'reset'

    def update(self, target, command, position, linear_velocity, angular_velocity,
               tilt_degrees, now, *, allowed=True):
        raw = np.asarray(target, dtype=np.float32)
        command = np.asarray(command, dtype=float)
        if raw.shape != (12,) or not np.isfinite(raw).all():
            raise ValueError('Stationary hold requires twelve finite joint targets')
        if command.shape != (3,) or not np.isfinite(command).all() or not math.isfinite(now):
            raise ValueError('Stationary hold command and clock must be finite')
        now = float(now)
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        p = np.asarray(position, dtype=float)
        v = np.asarray(linear_velocity, dtype=float)
        w = np.asarray(angular_velocity, dtype=float)
        supported = (p.shape == (3,) and np.isfinite(p).all()
                     and v.shape == (3,) and w.shape == (3,)
                     and np.isfinite(v).all() and np.isfinite(w).all()
                     and p[2] >= self.ground_height + .20
                     and math.isfinite(tilt_degrees) and 0. <= tilt_degrees < 18.)
        zero_request = bool(np.max(np.abs(command)) <= 1e-8)
        eligible = bool(allowed and zero_request and supported)

        if self.state == 'holding':
            if eligible:
                return self.held_target.copy()
            self.state = 'releasing'
            self.release_started = now
            self.release_count += 1
            self.release_fraction = 0.
            self.reason = ('movement_requested' if not zero_request else
                           'hold_not_permitted' if not allowed else 'body_support_changed')
            self.stop.reset()

        if self.state == 'releasing':
            fraction = min(1., max(0., (now - self.release_started) / self.release_seconds))
            self.release_fraction = fraction
            result = ((1. - fraction) * self.held_target + fraction * raw).astype(np.float32)
            if fraction >= 1. - 1e-9:
                self.state = 'tracking'
                self.held_target = None
                self.held_since = None
                self.release_started = None
                self.release_fraction = 1.
                self.stop.reset()
                return raw.copy()
            return result

        if not eligible:
            self.stop.reset()
            return raw.copy()
        if self.stop.update(p, linear_velocity, angular_velocity, tilt_degrees, now):
            self.held_target = raw.copy()
            self.held_since = now
            self.state = 'holding'
            self.capture_count += 1
            self.reason = 'complete_measured_stop'
            self.release_fraction = 0.
            return self.held_target.copy()
        return raw.copy()

    def report(self):
        return {'state': self.state, 'hold_seconds': self.hold_seconds,
                'release_seconds': self.release_seconds, 'capture_count': self.capture_count,
                'release_count': self.release_count, 'held_since': self.held_since,
                'release_fraction': self.release_fraction, 'reason': self.reason}
