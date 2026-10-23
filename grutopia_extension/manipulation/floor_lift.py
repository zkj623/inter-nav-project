"""Asset descriptions, bounded URDF IK, and honest floor-lift checks."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LiftAsset:
    name: str
    dimensions: tuple
    mass: float
    color: tuple
    handle_height: float
    handle_width: float = 0.024

    @property
    def grasp_offset(self):
        return np.array((0.0, 0.0, self.handle_height))


LIFT_ASSETS = {
    'box': LiftAsset('box', (0.24, 0.18, 0.16), 0.6, (0.70, 0.45, 0.20), 0.235),
    'bin': LiftAsset('bin', (0.20, 0.20, 0.22), 0.5, (0.18, 0.55, 0.48), 0.295),
}


class UrdfArmKinematics:
    def __init__(self, urdf_path, side='right', tool_offset=(0.03689, 0.0, 0.0)):
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        root = ET.parse(Path(urdf_path)).getroot()
        by_child = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
        link = f'{side}_gripper_link'
        chain = []
        while link != 'base_link':
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        self.chain = []
        self.names = []
        lower = []
        upper = []
        for joint in reversed(chain):
            origin = joint.find('origin')
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                transform[:3, :3] = Rotation.from_euler(
                    'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                ).as_matrix()
            axis_element = joint.find('axis')
            axis = np.array((1.0, 0.0, 0.0)) if axis_element is None else np.fromstring(
                axis_element.get('xyz'), sep=' '
            )
            active = joint.get('type') != 'fixed'
            self.chain.append((transform, axis, active))
            if active:
                self.names.append(joint.get('name'))
                limit = joint.find('limit')
                lower.append(float(limit.get('lower')))
                upper.append(float(limit.get('upper')))
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.tool_offset = np.asarray(tool_offset, dtype=float)

    def forward(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.lower.shape:
            raise ValueError('joint vector has the wrong shape')
        transform = np.eye(4)
        joint_index = 0
        for origin, axis, active in self.chain:
            transform = transform @ origin
            if active:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * positions[joint_index]).as_matrix()
                transform = transform @ rotation
                joint_index += 1
        transform[:3, 3] += transform[:3, :3] @ self.tool_offset
        return transform

    def solve(self, position, orientation, seed, position_tolerance=0.008):
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        seed = np.clip(np.asarray(seed, dtype=float), self.lower + 1e-7, self.upper - 1e-7)

        def residual(positions):
            transform = self.forward(positions)
            rotation_error = Rotation.from_matrix(orientation @ transform[:3, :3].T).as_rotvec()
            return np.concatenate((
                6.0 * (transform[:3, 3] - position),
                0.6 * rotation_error,
                0.005 * (positions - seed),
            ))

        result = least_squares(
            residual, seed, bounds=(self.lower + 1e-7, self.upper - 1e-7),
            max_nfev=220, ftol=1e-9, xtol=1e-9, gtol=1e-9,
        )
        achieved = self.forward(result.x)
        error = float(np.linalg.norm(achieved[:3, 3] - position))
        angle = float(np.linalg.norm(Rotation.from_matrix(orientation @ achieved[:3, :3].T).as_rotvec()))
        if error > position_tolerance or angle > 0.10:
            raise RuntimeError(f'Unreachable grasp waypoint: position error {error:.4f} m, angle {angle:.3f} rad')
        return result.x


class LiftSuccessTracker:
    def __init__(self, initial_bottom, required_lift=0.15, hold_seconds=2.0):
        if not np.isfinite((initial_bottom, required_lift, hold_seconds)).all() or required_lift <= 0 or hold_seconds <= 0:
            raise ValueError('lift and hold thresholds must be positive')
        self.initial_bottom = float(initial_bottom)
        self.required_lift = float(required_lift)
        self.hold_seconds = float(hold_seconds)
        self.held_seconds = 0.0
        self.max_lift = 0.0
        self.success = False

    def update(self, object_bottom, timestep, grasped, robot_healthy=True):
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError('timestep must be positive')
        lift = float(object_bottom) - self.initial_bottom
        if not np.isfinite(lift):
            self.held_seconds = 0.0
            return False
        self.max_lift = max(self.max_lift, lift)
        valid = lift >= self.required_lift and grasped and robot_healthy
        self.held_seconds = self.held_seconds + timestep if valid else 0.0
        self.success = self.success or self.held_seconds + 1e-8 >= self.hold_seconds
        return self.success


def lowest_box_point(position, rotation_matrix, dimensions):
    half_extent = np.asarray(dimensions, dtype=float) / 2.0
    return float(np.asarray(position)[2] - np.abs(np.asarray(rotation_matrix)[2]) @ half_extent)


def policy_observation(joint_positions, object_position, phase, progress, dimensions):
    phase_encoding = np.zeros(5, dtype=np.float32)
    phase_encoding[phase] = 1.0
    return np.concatenate((
        np.asarray(joint_positions, dtype=np.float32),
        np.asarray(object_position, dtype=np.float32),
        np.asarray(dimensions, dtype=np.float32),
        phase_encoding,
        np.array([progress], dtype=np.float32),
    ))
