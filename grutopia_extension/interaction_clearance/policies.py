"""Expert policy initialization: physical bilateral pickup, local carry, and placement."""

import math

import numpy as np

from grutopia_extension.interaction_clearance.skill_runtime import Outcome
from grutopia_extension.manipulation.bimanual_floor_lift import opposed_side_contact


class PickPlacePolicy:
    version = 'scripted_ik_swerve_v0'

    def __init__(self, environment):
        self.environment = environment
        self.completed_phases = []

    def phase_end(self, name):
        self.completed_phases.append(name)
        self.environment.recorder.snapshot(self.environment, name)
        details = {'phase': name, 'time_seconds': self.environment.elapsed_steps * self.environment.control_dt,
                   'object_position': self.environment.object.get_world_pose()[0].tolist(),
                   'base_position': self.environment.base.get_world_pose()[0].tolist(),
                   'object_bottom_m': self.environment.object_bottom()}
        self.environment.recorder.event('phase_end', details)
        print(details, flush=True)

    def interpolate(self, target, duration, phase):
        environment = self.environment
        previous = environment.command[environment.control_indices].copy()
        steps = round(duration / environment.control_dt)
        for index in range(steps):
            progress = (index + 1) / steps
            blend = progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)
            environment.advance(previous + blend * (target - previous), phase, progress)
        self.phase_end(phase)

    def hold(self, duration, phase='hold'):
        environment = self.environment
        joints = environment.command[environment.control_indices].copy()
        steps = round(duration / environment.control_dt)
        for index in range(steps):
            environment.advance(joints, phase, (index + 1) / steps)
        self.phase_end(phase)

    def drive(self, destination, phase, carrying):
        environment = self.environment
        initial = environment.base.get_world_pose()[0].copy()
        start_quaternion = environment.start_base_quaternion
        target_yaw = math.atan2(2 * (start_quaternion[0] * start_quaternion[3] + start_quaternion[1] * start_quaternion[2]),
                                1 - 2 * (start_quaternion[2] ** 2 + start_quaternion[3] ** 2))
        joints = environment.command[environment.control_indices].copy()
        initial_distance = max(float(np.linalg.norm(np.asarray(destination) - initial[:2])), 1e-6)
        settled = 0.0
        world_command = np.zeros(2)
        for _ in range(round(25 / environment.control_dt)):
            position, quaternion = environment.base.get_world_pose()
            yaw = math.atan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                             1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))
            error = np.asarray(destination) - position[:2]
            desired = 1.5 * error
            norm = float(np.linalg.norm(desired))
            if norm > 0.08:
                desired *= 0.08 / norm
            delta = desired - world_command
            magnitude = float(np.linalg.norm(delta))
            maximum_delta = 0.10 * environment.control_dt
            if magnitude > maximum_delta:
                delta *= maximum_delta / magnitude
            world_command += delta
            local = environment.rotation(quaternion).T @ np.r_[world_command, 0.0]
            yaw_error = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
            environment.base_twist = np.r_[local[:2], np.clip(2.0 * yaw_error, -0.15, 0.15)]
            progress = float(np.clip(1 - np.linalg.norm(error) / initial_distance, 0, 1))
            environment.advance(joints, phase, progress)
            if carrying:
                forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
                if environment.object_bottom() < 0.10 or not opposed_side_contact(forces):
                    raise RuntimeError('GRASP_LOST_DURING_LOCAL_CARRY')
            slow = np.linalg.norm(environment.base.get_linear_velocity()[:2]) < 0.02
            settled = settled + environment.control_dt if np.linalg.norm(error) < 0.007 and slow and abs(yaw_error) < 0.01 else 0.0
            if settled >= 0.25:
                environment.base_twist[:] = 0.0
                for _ in range(25):
                    environment.advance(joints, phase, 1.0)
                self.phase_end(phase)
                return
        environment.base_twist[:] = 0.0
        raise RuntimeError(f'LOCAL_DRIVE_TIMEOUT: destination={list(destination)}')

    def execute(self, task):
        environment = self.environment
        if task.mass_kg > 2.0 or np.any(np.abs(np.asarray(task.dimensions) - (0.4, 0.5, 0.28)) > 0.08):
            return Outcome(False, 'OUTSIDE_INITIAL_POLICY_CAPABILITY')
        if environment.object_bottom() > 0.02:
            return Outcome(False, 'PICKUP_REQUIRES_GROUNDED_OBJECT')
        try:
            center = environment.object.get_world_pose()[0].copy()
            initial = environment.command[environment.control_indices].copy()
            outside = task.dimensions[1] / 2 + 0.13
            pressed = task.dimensions[1] / 2 - environment.box.squeeze
            approach = np.r_[environment.solve_both(center, outside, environment.box.grasp_height, initial[:15]), np.zeros(4)]
            squeeze = np.r_[environment.solve_both(center, pressed, environment.box.grasp_height, approach[:15]), np.zeros(4)]
            lifted = np.r_[environment.solve_both(center, pressed, environment.box.grasp_height + 0.22, squeeze[:15]), np.zeros(4)]
            environment.validate_plan(initial, [approach, squeeze, lifted, lifted], center)
            self.interpolate(approach, 5.0, 'approach')
            self.interpolate(squeeze, 4.0, 'squeeze')
            self.interpolate(lifted, 4.0, 'lift')
            self.hold(1.0)
            if environment.object_bottom() < 0.15:
                raise RuntimeError('INSUFFICIENT_PICKUP_HEIGHT')
            current_object = environment.object.get_world_pose()[0]
            base_position = environment.base.get_world_pose()[0]
            destination = base_position[:2] + np.asarray(task.place_xy) - current_object[:2]
            self.drive(destination, 'transport', carrying=True)
            floor_center = environment.object.get_world_pose()[0].copy()
            floor_center[2] = 0.0001
            lowered = np.r_[environment.solve_both(floor_center, pressed, environment.box.grasp_height,
                                                  environment.command[environment.control_indices[:15]]), np.zeros(4)]
            self.interpolate(lowered, 4.0, 'lower')
            if environment.object_bottom() > 0.02:
                raise RuntimeError('PLACEMENT_NOT_GROUND_SUPPORTED')
            center = environment.object.get_world_pose()[0].copy()
            released = np.r_[environment.solve_both(center, outside, environment.box.grasp_height,
                                                   lowered[:15]), np.zeros(4)]
            self.interpolate(released, 4.0, 'release')
            self.interpolate(environment.initial_control, 5.0, 'retract')
            self.drive(environment.start_base_position[:2], 'return', carrying=False)
            self.hold(2.0, 'settle')
            environment.interaction_state = 'released'
            return Outcome(True, 'POLICY_SEQUENCE_COMPLETE', {'completed_phases': self.completed_phases,
                                                             'policy_version': self.version})
        except RuntimeError as error:
            environment.base_twist[:] = 0.0
            reason = 'SAFETY_REJECTION' if not environment.safety_ok else 'POLICY_EXECUTION_FAILURE'
            return Outcome(False, reason, {'error': str(error), 'completed_phases': self.completed_phases})
