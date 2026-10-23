"""Composable physical base, collision-aware dual-arm planning, and grasp/place policies."""

import json
import math

import numpy as np

from grutopia_extension.interaction_clearance.base_planner import PlanningError, SE2GridPlanner, pose_samples
from grutopia_extension.interaction_clearance.motion_geometry import (
    angle_error, base_waypoints, grasp_base_pose, opposed_object_contact, planar_transform,
    quaternion_yaw, sample_base_path,
)
from grutopia_extension.interaction_clearance.policies import PickPlacePolicy
from grutopia_extension.interaction_clearance.placement import carrying_base_pose, placement_candidates
from grutopia_extension.interaction_clearance.skill_runtime import Outcome
from grutopia_extension.interaction_clearance.task import yaw_rotation


VERSION = 'composable_region_heading_motion_roles_v4r5'


def measured_base(environment):
    position, quaternion = environment.base.get_world_pose()
    return np.r_[position[:2], quaternion_yaw(quaternion)]


def object_transform(environment):
    position, quaternion = environment.object.get_world_pose()
    transform = np.eye(4)
    transform[:3, :3] = environment.rotation(quaternion)
    transform[:3, 3] = position + transform[:3, :3] @ np.array((0, 0, environment.box.dimensions[2] / 2))
    return transform


def check_configuration(environment, joints, base, box, minimum=0.015):
    model = environment.collision_model
    transforms = model.forward(dict(zip(environment.bimanual.names, joints[:15])), base)
    distances = model.box_clearances(transforms, box, environment.box.dimensions)
    minimum_box = min(distances.values())
    self_hits = model.self_intersections(transforms)
    objects = model.objects(transforms)
    objects['task_object'] = model.fcl.CollisionObject(model.fcl.Box(*environment.box.dimensions),
                                                    model.fcl.Transform(box[:3, :3], box[:3, 3]))
    request = model.fcl.DistanceRequest(enable_signed_distance=True)
    minimum_wall = min(float(model.fcl.distance(body, wall, request, model.fcl.DistanceResult()))
                       for body in objects.values() for wall in environment.wall_objects.values())
    details = {'minimum_noncontact_clearance_m': minimum_box, 'minimum_wall_clearance_m': minimum_wall,
               'self_intersections': self_hits}
    if minimum_box < minimum or minimum_wall < minimum or self_hits:
        raise RuntimeError(f'PLANNED_COLLISION: {details}')
    return details


class FrozenBaseCollisionChecker:
    def __init__(self, environment, carrying, minimum=0.04):
        self.environment = environment
        self.model = environment.collision_model
        self.carrying = carrying
        self.minimum = minimum
        self.height = environment.base.get_world_pose()[0][2]
        positions = dict(zip(environment.robot.dof_names, environment.robot.get_joint_positions()))
        self.local = self.model.forward(positions)
        if self.model.self_intersections(self.local):
            raise RuntimeError('BASE_START_SELF_COLLISION')
        self.bodies = self.model.objects(self.local)
        self.noncontact = list(self.model.noncontact_names)
        for module in (1, 2, 3):
            steer = f'steer_motor_link{module}'
            wheel = f'wheel_motor_link{module}'
            if steer not in self.local or wheel not in self.local:
                continue
            center = self.local[steer][:3, 3]
            vertices = np.concatenate([self.model.vertices[name] @ self.local[name][:3, :3].T + self.local[name][:3, 3]
                                       for name in (steer, wheel)])
            radius = float(np.linalg.norm(vertices[:, :2] - center[:2], axis=1).max())
            lower, upper = float(vertices[:, 2].min()), float(vertices[:, 2].max())
            transform = np.eye(4)
            transform[:3, 3] = (*center[:2], (lower + upper) / 2)
            name = f'steering_sweep_{module}'
            self.local[name] = transform
            self.bodies[name] = self.model.fcl.CollisionObject(self.model.fcl.Cylinder(radius, upper - lower),
                                                              self.model.fcl.Transform(transform[:3, :3], transform[:3, 3]))
            self.noncontact.append(name)
        box = object_transform(environment)
        self.relative_box = np.linalg.inv(planar_transform(measured_base(environment), self.height)) @ box
        self.box = self.model.fcl.CollisionObject(self.model.fcl.Box(*environment.box.dimensions),
                                                self.model.fcl.Transform(box[:3, :3], box[:3, 3]))
        self.request = self.model.fcl.DistanceRequest(enable_signed_distance=True)
        self.internal_gap = None
        if carrying:
            relative = self.relative_box
            local_box = self.model.fcl.CollisionObject(self.model.fcl.Box(*environment.box.dimensions),
                                                      self.model.fcl.Transform(relative[:3, :3], relative[:3, 3]))
            self.internal_gap = min(self.distance(self.bodies[name], local_box) for name in self.noncontact)
            if self.internal_gap < 0.015:
                raise RuntimeError('HELD_OBJECT_BODY_CLEARANCE_REJECTED')

    def distance(self, first, second):
        return float(self.model.fcl.distance(first, second, self.request, self.model.fcl.DistanceResult()))

    def details(self, pose):
        base = planar_transform(pose, self.height)
        for name, body in self.bodies.items():
            transform = base @ self.local[name]
            body.setTransform(self.model.fcl.Transform(transform[:3, :3], transform[:3, 3]))
        if self.carrying:
            transform = base @ self.relative_box
            self.box.setTransform(self.model.fcl.Transform(transform[:3, :3], transform[:3, 3]))
            gap = self.internal_gap
        else:
            gap = min(self.distance(self.bodies[name], self.box) for name in self.noncontact)
        wall_gap = min(self.distance(body, wall) for body in (*self.bodies.values(), self.box)
                       for wall in self.environment.wall_objects.values())
        accepted = wall_gap >= self.minimum and gap >= (0.015 if self.carrying else self.minimum)
        return {'accepted': bool(accepted), 'minimum_noncontact_clearance_m': gap,
                'minimum_wall_clearance_m': wall_gap, 'self_intersections': []}

    def free(self, pose):
        return self.details(pose)['accepted']

    def search_margin(self):
        radii = [np.linalg.norm(vertices @ self.local[name][:3, :3].T + self.local[name][:3, 3], axis=1).max()
                 for name, vertices in self.model.vertices.items()]
        if self.carrying:
            radii.append(np.linalg.norm(self.relative_box[:2, 3]) + np.linalg.norm(self.environment.box.dimensions[:2]) / 2)
        return max(2.0, float(max(radii)) + 0.5)


class BaseMovePolicy:
    version = VERSION

    def __init__(self, environment):
        self.environment = environment

    def target(self, name):
        environment = self.environment
        state = environment.interaction_state
        position, quaternion = environment.object.get_world_pose()
        current = measured_base(environment)
        if name == 'grasp' and state == 'ready':
            return grasp_base_pose(position, quaternion_yaw(quaternion), environment.task.grasp_distance_m)
        if name == 'placement' and state == 'holding':
            selected = getattr(environment, 'selected_place_xy', environment.task.place_xy)
            return np.r_[current[:2] + np.asarray(selected) - position[:2], current[2]]
        if name == 'home' and state == 'released':
            return np.r_[environment.start_base_position[:2], quaternion_yaw(environment.start_base_quaternion)]
        raise ValueError(f'Invalid base target {name} in state {state}')

    def plan(self, target, route, carrying):
        environment = self.environment
        start = measured_base(environment)
        if route == 'auto':
            checker = FrozenBaseCollisionChecker(environment, carrying)
            corners = [start[:2], target[:2], np.asarray(environment.task.object_xy), np.asarray(environment.task.place_xy)]
            for obstacle in environment.task.scene_obstacles:
                center, half = np.asarray(obstacle.position[:2]), np.asarray(obstacle.dimensions[:2]) / 2
                corners.extend((center - half, center + half))
            margin = checker.search_margin()
            bounds = (np.min(corners, axis=0) - margin, np.max(corners, axis=0) + margin)
            yaw_delta = angle_error(target[2], start[2])
            changing_heading = carrying and abs(yaw_delta) > 1e-7
            planner_kind = 'carry_heading_astar' if changing_heading else 'xy_astar' if carrying else 'se2_astar'
            planner = SE2GridPlanner(checker.free, bounds, fixed_yaw=carrying and not changing_heading,
                                    yaw_offsets=(0.0, yaw_delta) if changing_heading else None,
                                    yaw_cost_m_per_rad=1.0 if changing_heading else 0.35)
            attempts = []
            try:
                result = planner.plan(start, target)
            except PlanningError as error:
                if not changing_heading:
                    raise
                attempts.append({'planner': planner_kind, 'error': str(error)})
                planner_kind = 'carry_full_se2_astar'
                planner = SE2GridPlanner(checker.free, bounds, yaw_cost_m_per_rad=1.0)
                result = planner.plan(start, target)
            samples = []
            for first, second in zip(result.waypoints, result.waypoints[1:]):
                for pose in pose_samples(first, second):
                    details = checker.details(pose)
                    if not details['accepted']:
                        raise RuntimeError('FINAL_BASE_PATH_CERTIFICATION_REJECTED')
                    samples.append({'pose': pose.tolist(), **details})
            if not samples:
                samples = [{'pose': start.tolist(), **checker.details(start)}]
            metadata = {'route': 'auto', 'planner': planner_kind,
                        'target': target.tolist(), 'bounds': np.asarray(bounds).tolist(),
                        'expanded_nodes': result.expanded_nodes, 'collision_queries': result.collision_queries,
                        'planning_seconds': result.search_seconds, 'used_search': result.used_search,
                        'waypoints': [point.tolist() for point in result.waypoints], 'samples': len(samples),
                        'external_planning_margin_m': checker.minimum,
                        'yaw_delta_rad': yaw_delta, 'prior_planning_attempts': attempts,
                        'search_bounds_margin_m': margin,
                        'minimum_wall_clearance_m': min(item['minimum_wall_clearance_m'] for item in samples)}
            environment.recorder.event('base_plan', metadata)
            (environment.output / f'base_search_{environment.elapsed_steps:05d}.json').write_text(json.dumps(metadata, indent=2))
            (environment.output / f'base_plan_{environment.elapsed_steps:05d}.json').write_text(json.dumps(samples, indent=2))
            return result.waypoints
        points = base_waypoints(start, target, route)
        joints = environment.robot.get_joint_positions()[environment.control_indices]
        height = environment.base.get_world_pose()[0][2]
        box = object_transform(environment)
        relative = np.linalg.inv(planar_transform(start, height)) @ box
        samples = []
        for pose in sample_base_path(points):
            base = planar_transform(pose, height)
            predicted_box = base @ relative if carrying else box
            details = check_configuration(environment, joints, base, predicted_box)
            samples.append({'pose': pose.tolist(), **details})
        environment.recorder.event('base_plan', {'route': route, 'target': target.tolist(), 'samples': len(samples),
                                                 'minimum_wall_clearance_m': min(item['minimum_wall_clearance_m'] for item in samples)})
        (environment.output / f'base_plan_{environment.elapsed_steps:05d}.json').write_text(json.dumps(samples, indent=2))
        return points

    def select_placement(self):
        environment = self.environment
        position, quaternion = environment.object.get_world_pose()
        yaw = quaternion_yaw(quaternion)
        start = measured_base(environment)
        checker = FrozenBaseCollisionChecker(environment, True)
        arm = ArmPlanPolicy(environment)
        initial = environment.robot.get_joint_positions()[environment.control_indices].copy()
        targets, phases, durations = arm.targets('place', position, initial)
        rejected = []
        for yaw_degrees in (0, -15, 15, -30, 30):
            yaw_delta = math.radians(yaw_degrees)
            goal_object_yaw = angle_error(yaw + yaw_delta, 0)
            candidates = []
            for destination in placement_candidates(environment.task, yaw_rotation(goal_object_yaw)):
                target = carrying_base_pose(start, position, destination, yaw_delta)
                details = checker.details(target)
                if not details['accepted']:
                    rejected.append({'place_xy': destination.tolist(), 'yaw_delta_degrees': yaw_degrees,
                                     'reason': 'HELD_GOAL_CLEARANCE', **details})
                    continue
                candidates.append((destination, target, details))
            candidates.sort(key=lambda item: (-min(item[2]['minimum_wall_clearance_m'], 0.15),
                                              float(np.linalg.norm(item[0] - environment.task.place_xy))))
            for destination, target, details in candidates:
                center = position.copy()
                center[:2] = destination
                base = planar_transform(target, environment.base.get_world_pose()[0][2])
                box = base @ checker.relative_box
                try:
                    arm.certify('place', initial, targets, phases, base, center, goal_object_yaw, box)
                    points = self.plan(target, 'auto', True)
                except RuntimeError as error:
                    rejected.append({'place_xy': destination.tolist(), 'yaw_delta_degrees': yaw_degrees, 'reason': str(error)})
                    continue
                environment.selected_place_xy = destination.copy()
                environment.recorder.event('placement_selected', {'place_xy': destination.tolist(),
                    'base_pose': target.tolist(), 'object_yaw_rad': goal_object_yaw,
                    'yaw_delta_degrees': yaw_degrees, 'requested_place_xy': list(environment.task.place_xy),
                    'candidate_count': len(candidates), 'rejected': rejected, 'goal_clearance': details, 'lowering_checked': True})
                return points
        environment.recorder.event('placement_rejected', {'rejected': rejected})
        raise RuntimeError('NO_FEASIBLE_PLACEMENT_IN_ALLOWED_REGION')

    def drive_segment(self, target, carrying, rotation_segment=False):
        environment = self.environment
        joints = environment.command[environment.control_indices].copy()
        world_command = np.zeros(2)
        angular_command = 0.0
        settled = 0.0
        segment_start = measured_base(environment)
        segment_delta = target[:2] - segment_start[:2]
        segment_length = float(np.dot(segment_delta, segment_delta))
        rotating_cargo = carrying and rotation_segment
        duration = self.segment_budget(segment_start, target, rotating_cargo)
        for index in range(round(duration / environment.control_dt)):
            current = measured_base(environment)
            progress = np.clip(np.dot(current[:2] - segment_start[:2], segment_delta) / max(segment_length, 1e-12), 0, 1)
            lateral_error = np.linalg.norm(current[:2] - segment_start[:2] - progress * segment_delta)
            if rotation_segment:
                lateral_error = np.linalg.norm(current[:2] - target[:2])
            if lateral_error > 0.035:
                raise RuntimeError('BASE_PATH_TRACKING_DEVIATION')
            if carrying and not rotation_segment and abs(angle_error(target[2], current[2])) > 0.07:
                raise RuntimeError('CARRYING_YAW_TRACKING_DEVIATION')
            error = target[:2] - current[:2]
            yaw_error = angle_error(target[2], current[2])
            desired = 1.5 * error
            desired *= min(1.0, 0.08 / max(np.linalg.norm(desired), 1e-9))
            delta = desired - world_command
            delta *= min(1.0, 0.10 * environment.control_dt / max(np.linalg.norm(delta), 1e-9))
            world_command += delta
            angular_speed = 0.08 if rotating_cargo else 0.20
            angular_acceleration = 0.05 if rotating_cargo else 0.3
            desired_angular = np.clip(1.5 * yaw_error, -angular_speed, angular_speed)
            angular_command += np.clip(desired_angular - angular_command,
                                       -angular_acceleration * environment.control_dt, angular_acceleration * environment.control_dt)
            local = yaw_rotation(current[2]).T @ np.r_[world_command, 0.0]
            environment.base_twist = np.r_[local[:2], angular_command]
            environment.advance(joints, 'transport' if carrying else 'base_move', index * environment.control_dt / duration)
            if carrying:
                forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
                if environment.object_bottom() < 0.10 or not opposed_object_contact(forces, quaternion_yaw(environment.object.get_world_pose()[1])):
                    raise RuntimeError('GRASP_LOST_DURING_BASE_MOVE')
            slow = np.linalg.norm(environment.base.get_linear_velocity()[:2]) < 0.02
            settled = settled + environment.control_dt if np.linalg.norm(error) < 0.007 and abs(yaw_error) < 0.008 and slow else 0.0
            if settled >= 0.25:
                environment.base_twist[:] = 0.0
                for _ in range(25):
                    environment.advance(joints, 'transport' if carrying else 'base_move', 1.0)
                return
        raise RuntimeError('BASE_MOVE_TIMEOUT')

    @staticmethod
    def segment_budget(start, target, rotating_cargo=False):
        translation = float(np.linalg.norm(target[:2] - start[:2])) / 0.08
        rotation = abs(angle_error(target[2], start[2])) / (0.08 if rotating_cargo else 0.20)
        return max(30.0, max(translation, rotation) + 8.0)

    def execute(self, task, *, target='grasp', route='auto'):
        environment = self.environment
        try:
            destination = self.target(target)
            carrying = environment.interaction_state == 'holding'
            points = self.select_placement() if target == 'placement' and route == 'auto' else self.plan(destination, route, carrying)
        except (ValueError, RuntimeError) as error:
            return Outcome(False, 'BASE_PLAN_REJECTED', {'error': str(error), 'retry_safe': True})
        environment.arm_plan = None
        try:
            for planned_start, waypoint in zip(points, points[1:]):
                current = measured_base(environment)
                if np.linalg.norm(waypoint[:2] - current[:2]) < 0.007 and abs(angle_error(waypoint[2], current[2])) < 0.008:
                    continue
                rotation_segment = (np.linalg.norm(waypoint[:2] - planned_start[:2]) < 1e-7
                                    and abs(angle_error(waypoint[2], planned_start[2])) > 1e-7)
                self.drive_segment(waypoint, carrying, rotation_segment=rotation_segment)
            environment.base_twist[:] = 0.0
            environment.recorder.snapshot(environment, 'base_' + target)
            return Outcome(True, 'BASE_TARGET_REACHED', {'target': target, 'route': route, 'pose': measured_base(environment).tolist()})
        except RuntimeError as error:
            environment.base_twist[:] = 0.0
            return Outcome(False, 'SAFETY_REJECTION' if not environment.safety_ok else 'BASE_EXECUTION_FAILURE', {'error': str(error)})


class ArmPlanPolicy:
    version = VERSION

    def __init__(self, environment):
        self.environment = environment

    def targets(self, mode, center, initial):
        environment = self.environment
        outside = environment.task.dimensions[1] / 2 + 0.13
        pressed = environment.task.dimensions[1] / 2 - environment.box.squeeze
        if mode == 'grasp':
            approach = np.r_[environment.solve_both(center, outside, environment.box.grasp_height, initial[:15]), np.zeros(4)]
            squeeze = np.r_[environment.solve_both(center, pressed, environment.box.grasp_height, approach[:15]), np.zeros(4)]
            lifted = np.r_[environment.solve_both(center, pressed, environment.box.grasp_height + 0.22, squeeze[:15]), np.zeros(4)]
            return [approach, squeeze, lifted], ['approach', 'squeeze', 'lift'], [5.0, 4.0, 4.0]
        floor = center.copy()
        floor[2] = 0.0001
        lowered = np.r_[environment.solve_both(floor, pressed, environment.box.grasp_height, initial[:15]), np.zeros(4)]
        release = np.r_[environment.solve_both(floor, outside, environment.box.grasp_height, lowered[:15]), np.zeros(4)]
        return [lowered, release, environment.initial_control.copy()], ['lower', 'release', 'retract'], [4.0, 4.0, 5.0]

    def certify(self, mode, initial, targets, phases, base, center, yaw, current_box):
        environment = self.environment
        previous, audit = initial, []
        for target, phase in zip(targets, phases):
            for progress in np.linspace(0, 1, 101):
                joints = previous + progress * (target - previous)
                box = current_box.copy()
                box[:3, :3] = yaw_rotation(yaw)
                if phase == 'lift':
                    box[:3, 3] = center + np.array((0, 0, environment.task.dimensions[2] / 2 + 0.22 * progress))
                elif mode == 'place':
                    box[:3, 3] = np.r_[center[:2], environment.task.dimensions[2] / 2 + (center[2] * (1 - progress) if phase == 'lower' else 0.0001)]
                details = check_configuration(environment, joints, base, box)
                audit.append({'phase': phase, 'progress': float(progress), **details})
            previous = target
        return audit

    def execute(self, task, *, mode='grasp'):
        environment = self.environment
        environment.arm_plan = None
        if task.mass_kg > 2.0 or np.any(np.abs(np.asarray(task.dimensions) - (0.4, 0.5, 0.28)) > 0.08):
            return Outcome(False, 'OUTSIDE_INITIAL_POLICY_CAPABILITY', {'retry_safe': True})
        if (mode == 'grasp' and environment.interaction_state != 'ready') or (mode == 'place' and environment.interaction_state != 'holding'):
            return Outcome(False, 'ARM_PLAN_PRECONDITION', {'retry_safe': True})
        try:
            center, quaternion = environment.object.get_world_pose()
            center = center.copy()
            yaw = quaternion_yaw(quaternion)
            if mode == 'grasp':
                target_base = grasp_base_pose(center, yaw, task.grasp_distance_m)
                current = measured_base(environment)
                if np.linalg.norm(target_base[:2] - current[:2]) > 0.025 or abs(angle_error(target_base[2], current[2])) > 0.035:
                    return Outcome(False, 'BASE_NOT_ALIGNED', {'retry_safe': True})
            elif np.linalg.norm(center[:2] - getattr(environment, 'selected_place_xy', task.place_xy)) > 0.055:
                return Outcome(False, 'OBJECT_NOT_AT_PLACEMENT', {'retry_safe': True})
            initial = environment.robot.get_joint_positions()[environment.control_indices].copy()
            targets, phases, durations = self.targets(mode, center, initial)
            base = planar_transform(measured_base(environment), environment.base.get_world_pose()[0][2])
            current_box = object_transform(environment)
            audit = self.certify(mode, initial, targets, phases, base, center, yaw, current_box)
            environment.arm_plan = {'mode': mode, 'initial': initial, 'targets': targets, 'phases': phases,
                                    'durations': durations, 'base': measured_base(environment), 'object': center,
                                    'object_yaw': yaw, 'step': environment.elapsed_steps}
            (environment.output / f'arm_plan_{mode}_{environment.elapsed_steps:05d}.json').write_text(json.dumps(audit, indent=2))
            environment.recorder.event('arm_plan', {'mode': mode, 'samples': len(audit), 'object_yaw_rad': yaw,
                                                  'planning': 'coupled_task_frame_IK_plus_sampled_joint_interpolation'})
            return Outcome(True, 'ARM_PLAN_READY', {'mode': mode, 'waypoints': len(targets), 'retry_safe': True})
        except RuntimeError as error:
            environment.arm_plan = None
            return Outcome(False, 'ARM_PLAN_REJECTED', {'error': str(error), 'retry_safe': True})


class ArmExecutePolicy(PickPlacePolicy):
    version = VERSION

    def __init__(self, environment, mode):
        super().__init__(environment)
        self.mode = mode

    def execute(self, task):
        environment = self.environment
        plan = environment.arm_plan
        if plan is None or plan['mode'] != self.mode:
            return Outcome(False, 'MISSING_ARM_PLAN', {'retry_safe': True})
        current = measured_base(environment)
        pose, quaternion = environment.object.get_world_pose()
        stale = (np.linalg.norm(current[:2] - plan['base'][:2]) > 0.015 or abs(angle_error(current[2], plan['base'][2])) > 0.02
                 or np.linalg.norm(pose - plan['object']) > 0.025 or abs(angle_error(quaternion_yaw(quaternion), plan['object_yaw'])) > 0.03
                 or np.max(np.abs(environment.robot.get_joint_positions()[environment.control_indices] - plan['initial'])) > 0.04)
        if stale:
            environment.arm_plan = None
            return Outcome(False, 'STALE_ARM_PLAN', {'retry_safe': True})
        environment.arm_plan = None
        try:
            for target, duration, phase in zip(plan['targets'], plan['durations'], plan['phases']):
                self.interpolate(target, duration, phase)
                if phase == 'lower' and environment.object_bottom() > 0.02:
                    raise RuntimeError('PLACEMENT_NOT_GROUND_SUPPORTED')
            if self.mode == 'grasp':
                self.hold(1.0)
                forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
                if environment.object_bottom() < 0.15 or not opposed_object_contact(forces, quaternion_yaw(environment.object.get_world_pose()[1])):
                    raise RuntimeError('GRASP_NOT_ESTABLISHED')
                environment.interaction_state = 'holding'
            else:
                self.hold(2.0, 'settle')
                if environment.object_bottom() > 0.02 or np.linalg.norm(environment.finger_force_vectors, axis=1).sum() >= 1.0:
                    raise RuntimeError('OBJECT_NOT_RELEASED')
                environment.interaction_state = 'released'
            return Outcome(True, 'GRASP_LIFT_COMPLETE' if self.mode == 'grasp' else 'PLACE_RELEASE_COMPLETE',
                           {'state': environment.interaction_state, 'completed_phases': self.completed_phases})
        except RuntimeError as error:
            environment.base_twist[:] = 0.0
            return Outcome(False, 'SAFETY_REJECTION' if not environment.safety_ok else 'ARM_EXECUTION_FAILURE', {'error': str(error)})


def primitive_registry(environment):
    return {'base_move': BaseMovePolicy(environment), 'arm_plan': ArmPlanPolicy(environment),
            'grasp_lift': ArmExecutePolicy(environment, 'grasp'), 'place_release': ArmExecutePolicy(environment, 'place')}
