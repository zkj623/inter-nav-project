"""Explicit measured geometry and response for the deployed SDK backend.

An SDK profile is an operator-supplied calibration, not an assertion of
hardware acceptance. Files must explicitly provide every measured field.
Its provenance is recorded verbatim; this module never manufactures evidence.
"""
from dataclasses import asdict, dataclass, fields
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ContactProfile:
    backend: str
    calibration_reference: str
    nose_offset_m: float
    foot_x_min_m: float
    foot_x_max_m: float
    foot_half_width_m: float
    head_x_min_m: float
    head_x_max_m: float
    head_half_width_m: float
    head_z_min_m: float
    head_z_max_m: float
    max_forward_mps: float
    max_reverse_mps: float
    max_lateral_mps: float
    max_yaw_rps: float
    forward_mps: float
    reverse_mps: float
    lateral_mps: float
    push_start_mps: float
    push_max_mps: float
    min_translation_mps: float
    min_yaw_rps: float
    linear_acceleration_mps2: float
    yaw_acceleration_rps2: float
    push_acceleration_mps2: float
    translation_stop_seconds: float
    push_stop_seconds: float
    stop_margin_m: float
    push_stop_margin_m: float
    release_margin_m: float
    yaw_stop_seconds: float
    yaw_stop_margin_rad: float
    settle_seconds: float
    stall_seconds: float
    max_contact_seconds: float
    max_tilt_degrees: float
    # SDK values must be supplied from stationary/moving tracking measurements.
    # Position tolerance applies to all cuboid surface corners, including yaw.
    box_position_tolerance_m: float
    box_settle_seconds: float
    box_max_settled_speed_mps: float

    def __post_init__(self):
        if self.backend != 'sdk':
            raise ValueError('Deploy contact backend must be sdk')
        if not isinstance(self.calibration_reference, str):
            raise ValueError('Contact calibration_reference must be a string')
        if not self.calibration_reference.strip():
            raise ValueError('SDK contact profile requires calibration_reference')
        for field in fields(self):
            if field.name in {'backend', 'calibration_reference'}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'Contact profile {field.name} must be finite')
            if field.name not in {'foot_x_min_m', 'head_z_min_m'} and value <= 0:
                raise ValueError(f'Contact profile {field.name} must be positive')
        if not self.foot_x_min_m < 0 < self.foot_x_max_m:
            raise ValueError('Contact foot bounds must straddle the base origin')
        if not 0 < self.head_x_min_m < self.head_x_max_m <= self.foot_x_max_m:
            raise ValueError('Contact head bounds must fit inside the front-foot envelope')
        if not self.head_z_min_m < self.head_z_max_m:
            raise ValueError('Contact head height bounds are inverted')
        if self.head_half_width_m > self.foot_half_width_m:
            raise ValueError('Contact head width exceeds the body footprint')
        if not self.head_x_min_m <= self.nose_offset_m <= self.head_x_max_m:
            raise ValueError('Contact nose must lie inside the declared head extent')
        if not self.min_translation_mps <= min(self.forward_mps, self.reverse_mps, self.lateral_mps):
            raise ValueError('Minimum translation speed exceeds a configured axis speed')
        if (self.forward_mps > self.max_forward_mps or self.reverse_mps > self.max_reverse_mps
                or self.lateral_mps > self.max_lateral_mps
                or not 0 < self.push_start_mps <= self.push_max_mps <= self.max_forward_mps
                or self.min_yaw_rps > self.max_yaw_rps):
            raise ValueError('Contact motion requests exceed profile speed limits')
        if self.push_acceleration_mps2 > self.linear_acceleration_mps2:
            raise ValueError('Contact push acceleration exceeds the linear acceleration limit')
        if not 0 < self.max_tilt_degrees < 30 or self.stall_seconds > self.max_contact_seconds:
            raise ValueError('Invalid contact attitude or time budget')
        if self.box_settle_seconds >= self.max_contact_seconds:
            raise ValueError('Box settling window must fit inside the contact budget')
        if self.box_position_tolerance_m >= min(self.stop_margin_m,self.release_margin_m):
            raise ValueError('Box stationary tolerance must be smaller than the contact geometry margins')

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError('Contact profile must be a mapping')
        names = {field.name for field in fields(cls)}
        unknown = set(payload) - names
        if unknown:
            raise ValueError('Unknown contact profile fields: ' + ', '.join(sorted(unknown)))
        # SDK calibration must not silently inherit simulation geometry or
        # stopping distances merely because a YAML key was misspelled/omitted.
        missing = names - set(payload)
        if missing:
            raise ValueError('SDK contact profile requires explicit fields: ' + ', '.join(sorted(missing)))
        return cls(**payload)

    def to_dict(self):
        return asdict(self)

    @property
    def head_bounds(self):
        return ((self.head_x_min_m, self.head_x_max_m),
                (-self.head_half_width_m, self.head_half_width_m),
                (self.head_z_min_m, self.head_z_max_m))

    def limit(self, command):
        """Bound translation direction as a unit and yaw independently."""
        result = np.asarray(command, dtype=float).copy()
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError('Contact command must contain three finite velocities')
        longitudinal = self.max_forward_mps if result[0] >= 0 else self.max_reverse_mps
        scale = max(1., abs(result[0]) / longitudinal, abs(result[1]) / self.max_lateral_mps)
        result[:2] /= scale
        result[2] = np.clip(result[2], -self.max_yaw_rps, self.max_yaw_rps)
        return result

    def stopping_distance(self, measured_speed, *, pushing=False, command_speed=0.):
        if not np.isfinite([measured_speed, command_speed]).all():
            raise ValueError('Contact stopping prediction requires finite measured speeds')
        margin = self.push_stop_margin_m if pushing else self.stop_margin_m
        response = self.push_stop_seconds if pushing else self.translation_stop_seconds
        return (margin + response * max(0., float(measured_speed))
                + max(0., float(command_speed))**2 / (2 * self.linear_acceleration_mps2))

    def motion_horizon(self, velocity):
        """Cover translation coast and the separately measured yaw response."""
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('Motion horizon requires finite (vx, vy, wz)')
        linear = self.translation_stop_seconds + np.linalg.norm(velocity[:2]) / (2*self.linear_acceleration_mps2)
        return max(linear, self.yaw_stop_seconds) if abs(velocity[2]) > 1e-8 else linear

    def skill_overrides(self):
        """Map measured response to the shared skill configuration."""
        return dict(front_contact_reach=self.foot_x_max_m - self.nose_offset_m,
                    forward_speed=self.forward_mps, reverse_speed=self.reverse_mps,
                    lateral_speed=self.lateral_mps, push_start_speed=self.push_start_mps,
                    push_max_speed=self.push_max_mps, push_acceleration=self.push_acceleration_mps2)
