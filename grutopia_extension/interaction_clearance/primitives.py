"""Composable physical base, collision-aware dual-arm planning, and grasp/place policies."""

import json
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from grutopia_extension.interaction_clearance.base_planner import PlanningError, SE2GridPlanner, pose_samples
from grutopia_extension.interaction_clearance.motion_geometry import (
    angle_error, base_waypoints, grasp_base_pose, grasp_frames, opposed_object_contact, planar_transform,
    quaternion_yaw, sample_base_path,
)
from grutopia_extension.interaction_clearance.policies import PickPlacePolicy
from grutopia_extension.interaction_clearance.placement import carrying_base_pose, placement_candidates
from grutopia_extension.interaction_clearance.skill_runtime import Outcome
from grutopia_extension.interaction_clearance.task import yaw_rotation


VERSION = 'composable_household_staged_release_and_short_support_v5r35'


def side_gripper_targets(task):
    """Spread stool contact locations tangentially; retain the original normal preload."""
    return np.array((0.03, -0.03, 0.03, -0.03)) if task.object_kind == 'stool' else np.zeros(4)


def grasp_contact_established(environment):
    """Require four signed vertical finger loads for the top/bottom edge grasp."""
    vectors = np.asarray(environment.finger_force_vectors)
    if getattr(environment, 'grasp_mode', None) == 'stool_seat_edge':
        if vectors.shape != (4, 3) or not np.isfinite(vectors).all():
            return False
        rotation = environment.rotation(environment.object.get_world_pose()[1])
        vertical = vectors @ rotation[:, 2]
        # Fixed -90 degree object-Y wrist rotation: lower/upper/upper/lower.
        return bool(np.all(vertical * np.array((-1., 1., 1., -1.)) > .5))
    return opposed_object_contact(vectors.reshape(2, 2, 3).sum(axis=1),
                                  quaternion_yaw(environment.object.get_world_pose()[1]))


def grasp_contact_retained(environment):
    """After four-contact calibration, verify loaded support and geometric retention."""
    reference = getattr(environment,'edge_native_grasp_frames',None)
    if getattr(environment,'grasp_mode',None) != 'stool_seat_edge' or reference is None:
        return grasp_contact_established(environment)

    def reject(reason,**details):
        environment.grasp_retention_diagnostics={'reason':reason,**details}
        if getattr(environment,'recorder',None) is not None:
            environment.recorder.event('grasp_retention_rejected',environment.grasp_retention_diagnostics)
        return False

    vectors=np.asarray(environment.finger_force_vectors)
    if vectors.shape!=(4,3) or not np.isfinite(vectors).all():
        return reject('invalid_contact_forces')
    position,quaternion=environment.object.get_world_pose()
    rotation=environment.rotation(quaternion)
    loads=(vectors@rotation[:,2])*np.array((-1.,1.,1.,-1.))
    if min(loads[0],loads[3])<=.5 or min(loads[1],loads[2])<-.05:
        return reject('missing_bilateral_lower_support',signed_loads_n=loads.tolist())
    if rotation[2,2] < math.cos(math.radians(5.)):
        return reject('object_tilt',vertical_axis_alignment=float(rotation[2,2]))
    angular=np.asarray(environment.object.get_angular_velocity())
    if not np.isfinite(angular).all() or np.linalg.norm(angular)>.35:
        return reject('object_angular_speed')
    positions,quaternions=environment.collision_links.get_world_poses()
    transforms={}
    for name,p,q in zip(environment.collision_link_names,positions,quaternions):
        if 'gripper_finger' not in name:continue
        pose=np.eye(4);pose[:3,3]=p;pose[:3,:3]=environment.rotation(q);transforms[name]=pose
    if len(transforms)!=4:
        return reject('missing_native_finger_pose')
    box=np.eye(4);box[:3,:3]=rotation
    box[:3,3]=position+rotation@np.array((0.,0.,environment.task.dimensions[2]/2))
    gaps=environment.collision_model.stool_finger_clearances(transforms,box,environment.task.dimensions)
    if not np.isfinite(list(gaps.values())).all() or min(gaps.values())<-.002 or max(gaps.values())>.001:
        return reject('finger_outside_retention_band',finger_gaps_m=gaps)
    origin=np.eye(4);origin[:3,:3]=rotation;origin[:3,3]=position
    p,q=environment.base.get_world_pose();base=np.eye(4);base[:3,:3]=environment.rotation(q);base[:3,3]=p
    measured=environment.robot.get_joint_positions()[environment.control_indices]
    current=[np.linalg.inv(origin)@base@tool for tool in environment.bimanual.forward(measured[:15])]
    translation=[pose[:3,3]-prior[:3,3] for pose,prior in zip(current,reference)]
    angles=[float(np.linalg.norm(Rotation.from_matrix(pose[:3,:3]@prior[:3,:3].T).as_rotvec())) for pose,prior in zip(current,reference)]
    if any(np.linalg.norm(delta[:2])>.003 for delta in translation) or max(angles)>.05:
        return reject('grasp_slip',translation_change_m=[v.tolist() for v in translation],rotation_change_rad=angles)
    environment.grasp_retention_diagnostics={'reason':'supported_and_geometrically_retained',
        'signed_loads_n':loads.tolist(),'finger_gaps_m':gaps,
        'translation_change_m':[v.tolist() for v in translation],'rotation_change_rad':angles}
    return True


class LiftGripPathError(RuntimeError):
    """A planned hand-span violation, before the corresponding motion executes."""

    def __init__(self, details, progress=None):
        self.details = details
        self.progress = None if progress is None else float(progress)
        super().__init__(f'LIFT_GRIP_PATH_INACCURATE: {details}; progress={self.progress}')


class InternalArmPathError(RuntimeError):
    """An unsafe robot/object or self geometry cannot be fixed by moving the whole placement."""
    def __init__(self, details, reason='PLANNED_COLLISION'):
        self.details = details
        super().__init__(f'{reason}: {details}')


def grasp_standoff_m(task):
    extent_reserve = max(0.0, (task.dimensions[0] - 0.40) / 2)
    if task.object_kind == 'box':
        # Overshooting arm reach forces tilted hand solutions on large boxes.
        # Retain a depth reserve while keeping the grasp in a better workspace.
        extent_reserve *= 0.25
    settling_reserve = min(0.07, task.dimensions[2] / 4) if task.object_kind == 'stool' else 0.0
    return min(0.72, task.grasp_distance_m + extent_reserve + settling_reserve)


def hand_clearance_candidates(mode, preferred):
    """Tool offsets from the task object, separate from environment safety margins."""
    candidates = (preferred, 0.13, 0.11, 0.09, 0.07, 0.05)
    # A supported object needs only a short lateral separation before retracting.
    # This fallback still requires full IK and trajectory collision certification.
    if mode == 'place':
        candidates += (0.04,)
    return tuple(dict.fromkeys(candidates))


def measured_base(environment):
    position, quaternion = environment.base.get_world_pose()
    return np.r_[position[:2], quaternion_yaw(quaternion)]


def object_transform(environment):
    position, quaternion = environment.object.get_world_pose()
    transform = np.eye(4)
    transform[:3, :3] = environment.rotation(quaternion)
    transform[:3, 3] = position + transform[:3, :3] @ np.array((0, 0, environment.box.dimensions[2] / 2))
    return transform


def check_configuration(environment, joints, base, box, minimum=0.015, finger_residuals=None):
    model = environment.collision_model
    positions = dict(zip(environment.bimanual.names, joints[:15]))
    if len(joints) >= 19:
        positions.update(zip((f'{side}_gripper_finger_joint{i}' for side in ('left', 'right') for i in (1, 2)), joints[15:19]))
    elif getattr(environment.task, 'object_kind', None) == 'stool':
        raise RuntimeError('STOOL_GRIPPER_TARGETS_MISSING')
    transforms = model.forward(positions, base)
    nominal = transforms
    if finger_residuals is not None:
        transforms = {name:pose.copy() for name,pose in nominal.items()}
        for name,residual in finger_residuals.items():
            transforms[name] = transforms[name] @ residual
    details = check_collision_transforms(environment, transforms, box, minimum)
    if finger_residuals is not None:
        # Keep both nominal and measured-offset finger envelopes clear of walls.
        request = model.fcl.DistanceRequest(enable_signed_distance=True)
        objects = model.objects(nominal)
        nominal_wall = min(float(model.fcl.distance(objects[name], wall, request, model.fcl.DistanceResult()))
            for name in finger_residuals for wall in environment.wall_objects.values())
        nominal_self = model.self_intersections(nominal)
        details['minimum_wall_clearance_m'] = min(details['minimum_wall_clearance_m'],nominal_wall)
        details['finger_tracking_model'] = 'native_pose_residual_in_commanded_link_frame'
        if nominal_self:
            raise InternalArmPathError({'self_intersections':nominal_self,
                                        'minimum_wall_clearance_m':nominal_wall},
                                       'PLANNED_NOMINAL_FINGER_COLLISION')
        if nominal_wall < environment.task.arm_obstacle_clearance_m:
            raise RuntimeError(f'PLANNED_NOMINAL_FINGER_COLLISION: {nominal_wall}, {nominal_self}')
    return details


def check_collision_transforms(environment, transforms, box, minimum=0.015):
    """Use the same gates for planned FK and native-pose finger-only calibration."""
    model = environment.collision_model
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
    wall_minimum = getattr(environment.task, 'arm_obstacle_clearance_m', minimum)
    details['required_wall_clearance_m'] = wall_minimum
    if getattr(environment, 'grasp_mode', None) == 'stool_seat_edge':
        all_gaps = model.box_clearances(transforms, box, environment.box.dimensions, include_fingers=True)
        details['minimum_finger_bbox_gap_m'] = min(g for n, g in all_gaps.items() if 'gripper_finger' in n)
        details['minimum_finger_gap_m'] = min(model.stool_finger_clearances(transforms, box, environment.box.dimensions).values())
        details['finger_clearance_geometry'] = 'stool_seat_and_legs_v1'
        if details['minimum_finger_gap_m'] < -.002:
            raise InternalArmPathError(details, 'PLANNED_FINGER_PENETRATION')
    if minimum_box < minimum or self_hits:
        raise InternalArmPathError(details)
    if minimum_wall < wall_minimum:
        raise RuntimeError(f'PLANNED_COLLISION: {details}')
    return details


class FrozenBaseCollisionChecker:
    def __init__(self, environment, carrying, minimum=None):
        self.environment = environment
        self.model = environment.collision_model
        self.carrying = carrying
        self.minimum = getattr(environment.task, 'base_obstacle_clearance_m', 0.04) if minimum is None else minimum
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
        accepted = wall_gap >= self.minimum and gap >= (0.015 if self.carrying else 0.04)
        return {'accepted': bool(accepted), 'minimum_noncontact_clearance_m': gap,
                'minimum_wall_clearance_m': wall_gap, 'required_wall_clearance_m': self.minimum, 'self_intersections': []}

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

    def target(self, name, grasp_distance_offset_m=0.0):
        environment = self.environment
        state = environment.interaction_state
        if (not isinstance(grasp_distance_offset_m, (int, float))
                or not math.isfinite(grasp_distance_offset_m) or not -.02 <= grasp_distance_offset_m <= 0.):
            raise ValueError('Grasp distance offset must be finite and within [-0.02, 0] m')
        if grasp_distance_offset_m and (name != 'grasp' or state != 'ready'):
            raise ValueError('Grasp stand-off correction is only valid before contact')
        position, quaternion = environment.object.get_world_pose()
        current = measured_base(environment)
        if name == 'grasp' and state == 'ready':
            return grasp_base_pose(position, quaternion_yaw(quaternion),
                                   grasp_standoff_m(environment.task) + grasp_distance_offset_m)
        if name == 'placement' and state == 'holding':
            selected = getattr(environment, 'selected_place_xy', environment.task.place_xy)
            center = object_transform(environment)[:3, 3]
            return np.r_[current[:2] + np.asarray(selected) - center[:2], current[2]]
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
                center = np.asarray(obstacle.position[:2])
                half = np.abs(yaw_rotation(obstacle.yaw_rad)[:2, :2]) @ (np.asarray(obstacle.dimensions[:2]) / 2)
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
        object_center = object_transform(environment)[:3, 3]
        yaw = quaternion_yaw(quaternion)
        start = measured_base(environment)
        checker = FrozenBaseCollisionChecker(environment, True)
        arm = (SeatEdgeArmPlanPolicy(environment) if getattr(environment,'grasp_mode',None) == 'stool_seat_edge'
               else ArmPlanPolicy(environment))
        initial = environment.robot.get_joint_positions()[environment.control_indices].copy()
        if hasattr(environment,'edge_commanded_grasp_frames'):
            initial = environment.command[environment.control_indices].copy()
        rejected = []
        preferred = getattr(environment, 'arm_approach_clearance_m', 0.13)
        approaches = hand_clearance_candidates('place', preferred)
        strategies = ('direct', 'staged', 'staged_high') if isinstance(arm, SeatEdgeArmPlanPolicy) else ('direct',)
        for approach_clearance in approaches:
            for strategy in strategies:
                try:
                    options = {'release_strategy': strategy} if isinstance(arm, SeatEdgeArmPlanPolicy) else {}
                    targets, phases, durations = arm.targets('place', position, initial, approach_clearance, **options)
                except RuntimeError as error:
                    rejected.append({'approach_clearance_m': approach_clearance,
                                     'release_strategy': strategy, 'reason': str(error)})
                    continue
                internal_rejection = False
                for yaw_degrees in (0, -15, 15, -30, 30):
                    yaw_delta = math.radians(yaw_degrees)
                    goal_object_yaw = angle_error(yaw + yaw_delta, 0)
                    candidates = []
                    for destination in placement_candidates(environment.task, yaw_rotation(goal_object_yaw)):
                        target = carrying_base_pose(start, object_center, destination, yaw_delta)
                        details = checker.details(target)
                        if not details['accepted']:
                            rejected.append({'place_xy': destination.tolist(), 'yaw_delta_degrees': yaw_degrees,
                                             'approach_clearance_m': approach_clearance, 'reason': 'HELD_GOAL_CLEARANCE', **details})
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
                        except InternalArmPathError as error:
                            rejected.append({'place_xy': destination.tolist(), 'yaw_delta_degrees': yaw_degrees,
                                'approach_clearance_m': approach_clearance, 'release_strategy': strategy,
                                'placement_independent_rejection': True, 'reason': str(error)})
                            internal_rejection = True
                            break
                        except RuntimeError as error:
                            rejected.append({'place_xy': destination.tolist(), 'yaw_delta_degrees': yaw_degrees,
                                             'approach_clearance_m': approach_clearance, 'release_strategy': strategy,
                                             'reason': str(error)})
                            continue
                        environment.selected_place_xy = destination.copy()
                        environment.arm_approach_clearance_m = approach_clearance
                        if isinstance(arm, SeatEdgeArmPlanPolicy):
                            environment.edge_release_strategy = strategy
                        environment.recorder.event('placement_selected', {'place_xy': destination.tolist(),
                            'base_pose': target.tolist(), 'object_yaw_rad': goal_object_yaw,
                            'yaw_delta_degrees': yaw_degrees, 'requested_place_xy': list(environment.task.place_xy),
                            'candidate_count': len(candidates), 'rejected': rejected, 'goal_clearance': details,
                            'approach_clearance_m': approach_clearance, 'release_strategy': strategy,
                            'lowering_checked': True})
                        return points
                    if internal_rejection:
                        break
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
                if environment.object_bottom() < 0.10 or not grasp_contact_retained(environment):
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

    def execute(self, task, *, target='grasp', route='auto', grasp_distance_offset_m=0.0):
        environment = self.environment
        try:
            destination = (self.target(target) if grasp_distance_offset_m == 0. else
                           self.target(target, grasp_distance_offset_m=grasp_distance_offset_m))
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
            return Outcome(True, 'BASE_TARGET_REACHED', {'target': target, 'route': route, 'pose': measured_base(environment).tolist(),
                                                        'grasp_standoff_m': grasp_standoff_m(task) + grasp_distance_offset_m if target == 'grasp' else None})
        except RuntimeError as error:
            environment.base_twist[:] = 0.0
            return Outcome(False, 'SAFETY_REJECTION' if not environment.safety_ok else 'BASE_EXECUTION_FAILURE', {'error': str(error)})


class ArmPlanPolicy:
    version = VERSION

    def __init__(self, environment):
        self.environment = environment

    def configuration(self,joints,base,box):
        return check_configuration(self.environment,joints,base,box)

    def check_squeeze_target(self, target, center):
        """Reserve at least two thirds of the side preload against IK error."""
        environment = self.environment
        preload = environment.box.squeeze
        tolerance = min(0.001, preload / 3)
        yaw = quaternion_yaw(environment.object.get_world_pose()[1])
        normal = yaw_rotation(yaw)[:, 1]
        desired = grasp_frames(center, yaw, environment.task.dimensions[1] / 2 - preload,
                               environment.box.grasp_height)[:2]
        base_position, base_quaternion = environment.base.get_world_pose()
        base_rotation = environment.rotation(base_quaternion)
        errors = []
        for tool, position, side in zip(environment.bimanual.forward(target[:15]), desired, (1, -1)):
            actual = base_position + base_rotation @ tool[:3, 3]
            errors.append(float(side * np.dot(actual - position, normal)))
        details = {'normal_errors_m': errors, 'tolerance_m': tolerance, 'preload_m': preload}
        if not np.isfinite([*errors, tolerance]).all() or tolerance <= 0 or max(map(abs, errors)) > tolerance:
            raise RuntimeError(f'GRASP_TARGET_INACCURATE: {details}')
        return details

    def targets(self, mode, center, initial, approach_clearance_m=None):
        environment = self.environment
        approach_clearance_m = getattr(environment, 'arm_approach_clearance_m', 0.13) if approach_clearance_m is None else approach_clearance_m
        outside = environment.task.dimensions[1] / 2 + approach_clearance_m
        pressed = environment.task.dimensions[1] / 2 - environment.box.squeeze
        if mode == 'grasp':
            approach = np.r_[environment.solve_both(center, outside, environment.box.grasp_height, initial[:15]), side_gripper_targets(environment.task)]
            squeeze = np.r_[environment.solve_both(center, pressed, environment.box.grasp_height, approach[:15]), side_gripper_targets(environment.task)]
            self.check_squeeze_target(squeeze, center)
            lift_targets = []
            seed = squeeze[:15]
            rotation = environment.rotation(environment.object.get_world_pose()[1])
            for progress in np.linspace(0, 1, 7)[1:]:
                seed = environment.solve_both(center, pressed, environment.box.grasp_height + 0.22 * progress,
                                              seed, object_rotation=rotation,
                                              **({'preserve_contact_normal': True} if environment.task.object_kind == 'stool' else {}))
                lift_targets.append(np.r_[seed, side_gripper_targets(environment.task)])
            return [approach, squeeze, *lift_targets], ['approach', 'squeeze'] + ['lift'] * 6, [5.0, 4.0] + [0.8] * 6
        floor = center.copy()
        floor[:2] = object_transform(environment)[:2, 3]
        floor[2] = 0.0001
        rotation = environment.rotation(environment.object.get_world_pose()[1])
        yaw = quaternion_yaw(environment.object.get_world_pose()[1])
        orientation_path = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((rotation, yaw_rotation(yaw)))))
        lower_steps = max(2, math.ceil(abs(center[2] - floor[2]) / 0.04))
        lower_targets = []
        seed = initial[:15]
        for progress in np.linspace(0, 1, lower_steps + 1)[1:]:
            waypoint = center + progress * (floor - center)
            seed = environment.solve_both(waypoint, pressed, environment.box.grasp_height, seed,
                                          object_rotation=orientation_path(progress).as_matrix())
            lower_targets.append(np.r_[seed, side_gripper_targets(environment.task)])
        lowered = lower_targets[-1]
        release = np.r_[environment.solve_both(floor, outside, environment.box.grasp_height, lowered[:15]), side_gripper_targets(environment.task)]
        lower_duration = max(0.8, 4.0 / lower_steps)
        if environment.task.object_kind == 'stool':
            # Seat-side contact permits object rocking during rapid descent.
            # Limit mean vertical descent to 2.5 cm/s without changing grips or forces.
            lower_duration = max(lower_duration, abs(center[2] - floor[2]) / (lower_steps * 0.025))
        return (lower_targets + [release, environment.initial_control.copy()], ['lower'] * lower_steps + ['release', 'retract'],
                [lower_duration] * lower_steps + [4.0, 5.0])

    def check_lift_grip(self, joints, base, box, progress=None):
        """Keep the commanded hand span within the original preload reserve."""
        environment = self.environment
        preload = environment.box.squeeze
        tolerance = min(0.001, preload / 3)
        left, right = environment.bimanual.forward(joints[:15])
        separation = base[:3, :3] @ (left[:3, 3] - right[:3, 3])
        half_separation = float(np.dot(separation, box[:3, 1]) / 2)
        error = half_separation - (environment.task.dimensions[1] / 2 - preload)
        details = {'grip_half_separation_m': half_separation, 'grip_span_error_m': error,
                   'grip_span_tolerance_m': tolerance}
        if not np.isfinite([half_separation, error, tolerance]).all() or tolerance <= 0 or abs(error) > tolerance:
            raise LiftGripPathError(details, progress)
        return details

    def certify(self, mode, initial, targets, phases, base, center, yaw, current_box, lift_distance=0.22,
                lift_origin_target=None,lift_rotation_target=None):
        environment = self.environment
        previous, audit = initial, []
        placed_box = current_box.copy()
        placed_box[:3, :3] = yaw_rotation(yaw)
        placed_box[:3, 3] = np.r_[current_box[:2, 3], environment.task.dimensions[2] / 2 + 0.0001]
        orientation_path = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((current_box[:3, :3], placed_box[:3, :3]))))
        half_height = np.array((0, 0, environment.task.dimensions[2] / 2))
        initial_origin = current_box[:3, 3] - current_box[:3, :3] @ half_height
        lift_orientation_path = (None if lift_rotation_target is None else
            Slerp([0.,1.],Rotation.from_matrix(np.stack((current_box[:3,:3],lift_rotation_target)))))
        lift_target = (initial_origin+np.array((0.,0.,lift_distance)) if lift_origin_target is None
                       else np.asarray(lift_origin_target))
        placed_origin = placed_box[:3, 3] - placed_box[:3, :3] @ half_height
        lower_steps = max(1, phases.count('lower'))
        lower_index = 0
        lift_steps = max(1, phases.count('lift'))
        lift_index = 0
        for target, phase in zip(targets, phases):
            for progress in np.linspace(0, 1, 101):
                joints = previous + progress * (target - previous)
                box = current_box.copy()
                box[:3, :3] = yaw_rotation(yaw)
                if phase == 'lift':
                    lift_progress = (lift_index + progress) / lift_steps
                    if lift_orientation_path is None and lift_origin_target is None:
                        box[:3, :3] = current_box[:3, :3]
                        box[:3, 3] = current_box[:3, 3] + np.array((0, 0, lift_distance * lift_progress))
                    else:
                        box[:3,:3] = current_box[:3,:3] if lift_orientation_path is None else lift_orientation_path(lift_progress).as_matrix()
                        predicted_origin = initial_origin+lift_progress*(lift_target-initial_origin)
                        box[:3,3] = predicted_origin+box[:3,:3]@half_height
                elif mode == 'place':
                    box = placed_box.copy()
                    if phase == 'lower':
                        lower_progress = (lower_index + progress) / lower_steps
                        box[:3, :3] = orientation_path(lower_progress).as_matrix()
                        origin = initial_origin + lower_progress * (placed_origin - initial_origin)
                        box[:3, 3] = origin + box[:3, :3] @ half_height
                details = self.configuration(joints, base, box)
                if phase == 'lift' and environment.task.object_kind == 'stool':
                    details.update(self.check_lift_grip(joints, base, box, progress))
                audit.append({'phase': phase, 'progress': float(progress), **details})
            previous = target
            if phase == 'lower':
                lower_index += 1
            if phase == 'lift':
                lift_index += 1
        return audit

    def execute(self, task, *, mode='grasp'):
        environment = self.environment
        environment.arm_plan = None
        dimensions = np.asarray(task.dimensions)
        if task.mass_kg > 2.0 or np.any(dimensions < (0.24, 0.28, 0.18)) or np.any(dimensions > (0.65, 0.65, 0.50)):
            return Outcome(False, 'OUTSIDE_INITIAL_POLICY_CAPABILITY', {'retry_safe': True})
        if (mode == 'grasp' and environment.interaction_state != 'ready') or (mode == 'place' and environment.interaction_state != 'holding'):
            return Outcome(False, 'ARM_PLAN_PRECONDITION', {'retry_safe': True})
        try:
            center, quaternion = environment.object.get_world_pose()
            center = center.copy()
            yaw = quaternion_yaw(quaternion)
            if mode == 'grasp':
                target_base = grasp_base_pose(center, yaw, grasp_standoff_m(task))
                current = measured_base(environment)
                if np.linalg.norm(target_base[:2] - current[:2]) > 0.025 or abs(angle_error(target_base[2], current[2])) > 0.035:
                    return Outcome(False, 'BASE_NOT_ALIGNED', {'retry_safe': True})
            elif np.linalg.norm(object_transform(environment)[:2, 3] - getattr(environment, 'selected_place_xy', task.place_xy)) > 0.055:
                return Outcome(False, 'OBJECT_NOT_AT_PLACEMENT', {'retry_safe': True})
            initial = environment.robot.get_joint_positions()[environment.control_indices].copy()
            if hasattr(environment,'edge_commanded_grasp_frames'):
                initial = environment.command[environment.control_indices].copy()
            base = planar_transform(measured_base(environment), environment.base.get_world_pose()[0][2])
            current_box = object_transform(environment)
            preferred = getattr(environment, 'arm_approach_clearance_m', 0.13)
            approaches = hand_clearance_candidates(mode, preferred)
            rejected = []
            for approach_clearance in approaches:
                try:
                    targets, phases, durations = self.targets(mode, center, initial, approach_clearance)
                    audit = self.certify(mode, initial, targets, phases, base, center, yaw, current_box)
                except RuntimeError as error:
                    rejected.append({'approach_clearance_m': approach_clearance, 'error': str(error)})
                    continue
                break
            else:
                raise RuntimeError(f'NO_CERTIFIED_ARM_APPROACH: {rejected}')
            environment.arm_approach_clearance_m = approach_clearance
            environment.arm_plan = {'mode': mode, 'initial': initial, 'targets': targets, 'phases': phases,
                                    'durations': durations, 'base': measured_base(environment), 'object': center,
                                    'object_yaw': yaw, 'step': environment.elapsed_steps,
                                    'approach_clearance_m': approach_clearance, 'rejected_approaches': rejected}
            (environment.output / f'arm_plan_{mode}_{environment.elapsed_steps:05d}.json').write_text(json.dumps(audit, indent=2))
            environment.recorder.event('arm_plan', {'mode': mode, 'samples': len(audit), 'object_yaw_rad': yaw,
                                                  'approach_clearance_m': approach_clearance, 'rejected_approaches': rejected,
                                                  'waypoint_durations_seconds': durations,
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

    def interpolate(self, target, duration, phase):
        environment = self.environment
        if phase != 'lift' or environment.task.object_kind != 'stool':
            return super().interpolate(target, duration, phase)
        previous = environment.command[environment.control_indices].copy()
        steps = round(duration / environment.control_dt)
        for index in range(steps):
            progress = (index + 1) / steps
            blend = progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)
            environment.advance(previous + blend * (target - previous), phase, progress)
            forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
            if not grasp_contact_retained(environment):
                raise RuntimeError('GRASP_LOST_DURING_LIFT')
        self.phase_end(phase)

    def grasp_motion(self, plan):
        environment = self.environment
        if plan['phases'][:2] != ['approach', 'squeeze']:
            raise RuntimeError('INVALID_GRASP_PLAN_SEQUENCE')
        for target, duration, phase in zip(plan['targets'][:2], plan['durations'][:2], plan['phases'][:2]):
            self.interpolate(target, duration, phase)
        forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
        if not grasp_contact_established(environment):
            raise RuntimeError('GRASP_CONTACT_MISSING_BEFORE_LIFT')
        environment.recorder.event('lift_strategy', {'strategy': 'measured_cartesian_side_grasp',
                                                    'object_kind': environment.task.object_kind})
        initial_height = float(environment.object.get_world_pose()[0][2])
        for segment in range(6):
            self.lift_segment(initial_height + 0.22 * (segment + 1) / 6, segment)

    def lift_segment(self, target_origin_height, segment, depth=0):
        environment = self.environment
        center, quaternion = environment.object.get_world_pose()
        displacement = float(target_origin_height - center[2])
        initial = environment.command[environment.control_indices].copy()
        pressed = environment.task.dimensions[1] / 2 - environment.box.squeeze
        target = np.r_[environment.solve_both(center, pressed, environment.box.grasp_height + displacement,
            initial[:15], object_rotation=environment.rotation(quaternion),
            **({'preserve_contact_normal': True} if environment.task.object_kind == 'stool' else {})), side_gripper_targets(environment.task)]
        base = planar_transform(measured_base(environment), environment.base.get_world_pose()[0][2])
        planner = ArmPlanPolicy(environment)
        try:
            audit = planner.certify('grasp', initial, [target], ['lift'], base, center, quaternion_yaw(quaternion),
                                    object_transform(environment), lift_distance=displacement)
        except LiftGripPathError as error:
            if (environment.task.object_kind != 'stool' or error.progress == 0.0 or depth >= 3
                    or abs(displacement) <= 0.005 or not np.isfinite(error.details['grip_span_error_m'])):
                raise
            midpoint = float(center[2] + displacement / 2)
            environment.recorder.event('lift_refinement', {'segment': segment, 'depth': depth,
                'target_origin_height_m': target_origin_height, 'midpoint_origin_height_m': midpoint,
                'rejected_sample_progress': error.progress, 'planning_error': error.details})
            # Each accepted half is recertified from the newly measured pose;
            # retain the 0.8 s duration so subdivision never speeds up the lift.
            self.lift_segment(midpoint, segment, depth + 1)
            self.lift_segment(target_origin_height, segment, depth + 1)
            return
        (environment.output / f'arm_plan_lift_{environment.elapsed_steps:05d}.json').write_text(json.dumps(audit, indent=2))
        environment.recorder.event('lift_replan', {'segment': segment, 'subdivision_depth': depth,
            'target_origin_height_m': target_origin_height,
            'measured_object_origin': center.tolist(), 'measured_base': measured_base(environment).tolist(),
            'minimum_body_gap_m': min(row['minimum_noncontact_clearance_m'] for row in audit),
            'minimum_wall_gap_m': min(row['minimum_wall_clearance_m'] for row in audit)})
        self.interpolate(target, 0.8, 'lift')
        forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
        if not grasp_contact_retained(environment):
            raise RuntimeError('GRASP_LOST_DURING_LIFT')

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
            if self.mode == 'grasp':
                self.grasp_motion(plan)
                self.hold(1.0)
                forces = environment.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
                if environment.object_bottom() < 0.15 or not grasp_contact_retained(environment):
                    raise RuntimeError('GRASP_NOT_ESTABLISHED')
                environment.interaction_state = 'holding'
            else:
                for phase_index, (target, duration, phase) in enumerate(zip(plan['targets'], plan['durations'], plan['phases'])):
                    self.interpolate(target, duration, phase)
                    final_lower = phase == 'lower' and (phase_index + 1 == len(plan['phases']) or plan['phases'][phase_index + 1] != 'lower')
                    if final_lower and environment.object_bottom() > 0.02:
                        raise RuntimeError('PLACEMENT_NOT_GROUND_SUPPORTED')
                self.hold(2.0, 'settle')
                if environment.object_bottom() > 0.02 or np.linalg.norm(environment.finger_force_vectors, axis=1).sum() >= 1.0:
                    raise RuntimeError('OBJECT_NOT_RELEASED')
                environment.interaction_state = 'released'
            return Outcome(True, 'GRASP_LIFT_COMPLETE' if self.mode == 'grasp' else 'PLACE_RELEASE_COMPLETE',
                           {'state': environment.interaction_state, 'completed_phases': self.completed_phases})
        except RuntimeError as error:
            environment.base_twist[:] = 0.0
            return Outcome(False, 'SAFETY_REJECTION' if not environment.safety_ok else 'ARM_EXECUTION_FAILURE', {'error': str(error)})


class SeatEdgeArmPlanPolicy(ArmPlanPolicy):
    """Shallow pinch of the top and underside of each side of the stool seat."""

    insertion_m = .01
    lift_segments = 12

    def refresh_commanded_grasp_reference(self):
        """Refresh loaded position offsets while preserving calibrated wrist rotations."""
        environment=self.environment
        if not hasattr(environment,'edge_commanded_grasp_frames'):
            return
        object_origin=object_transform(environment)
        object_origin[:3,3]-=object_origin[:3,:3]@np.array((0.,0.,environment.task.dimensions[2]/2))
        p,q=environment.base.get_world_pose();base=np.eye(4);base[:3,3]=p;base[:3,:3]=environment.rotation(q)
        command=environment.command[environment.control_indices]
        environment.edge_commanded_grasp_frames=tuple(np.linalg.inv(object_origin)@base@tool
            for tool in environment.bimanual.forward(command[:15]))
        calibrated_rotations=getattr(environment,'edge_calibrated_grasp_rotations',None)
        if calibrated_rotations is not None:
            # Repeatedly expressing the previous command in a slightly tilted
            # measured object frame integrates that tilt into later wrist goals.
            for frame,rotation in zip(environment.edge_commanded_grasp_frames,calibrated_rotations):
                frame[:3,:3]=rotation
        if environment.recorder is not None:
            environment.recorder.event('loaded_commanded_grasp_reference',{
                'tool_frames_in_object':[frame.tolist() for frame in environment.edge_commanded_grasp_frames],
                'orientation_reference':'calibrated_object_frame' if calibrated_rotations is not None else 'current_command_in_measured_object_frame',
                'native_retention_reference_unchanged':True,'joint_commands_unchanged':True})

    def refresh_finger_tracking(self):
        environment = self.environment
        if not hasattr(environment,'edge_commanded_grasp_frames'):
            return
        positions,quaternions = environment.collision_links.get_world_poses()
        actual = {}
        for name,position,quaternion in zip(environment.collision_link_names,positions,quaternions):
            if 'gripper_finger' not in name: continue
            pose=np.eye(4);pose[:3,3]=position;pose[:3,:3]=environment.rotation(quaternion);actual[name]=pose
        p,q=environment.base.get_world_pose();base=np.eye(4);base[:3,3]=p;base[:3,:3]=environment.rotation(q)
        command=environment.command[environment.control_indices]
        names=environment.bimanual.names+[f'{s}_gripper_finger_joint{i}' for s in ('left','right') for i in (1,2)]
        nominal=environment.collision_model.forward(dict(zip(names,command)),base)
        environment.edge_finger_tracking_residuals={name:np.linalg.inv(nominal[name])@pose for name,pose in actual.items()}
        if len(environment.edge_finger_tracking_residuals)!=4:
            raise RuntimeError('INCOMPLETE_NATIVE_FINGER_TRACKING')

    def configuration(self,joints,base,box):
        return check_configuration(self.environment,joints,base,box,
            finger_residuals=getattr(self.environment,'edge_finger_tracking_residuals',None))

    def finger_targets(self, opened=False):
        if not opened and hasattr(self.environment, 'edge_finger_command'):
            command = np.asarray(self.environment.edge_finger_command, dtype=float)
            signs = np.array((1., -1., 1., -1.))
            if command.shape != (4,) or not np.isfinite(command).all() or np.any(command*signs <= 0.) or np.any(command*signs >= .05):
                raise RuntimeError('INVALID_CALIBRATED_FINGER_COMMAND')
            return command.copy()
        thickness = min(.06, self.environment.task.dimensions[2] / 4)
        # At 10 mm insertion, exported guards touch a 60 mm seat near36.004mm.
        # 35.1mm gives about0.904mm closure, below the previous0.944mm estimate.
        aperture = thickness / 2 + (.015 if opened else .0051)
        if not 0. < aperture <= .05:
            raise RuntimeError('SEAT_OUTSIDE_GRIPPER_APERTURE')
        return aperture * np.array((1., -1., 1., -1.))

    def solve_edge(self, origin, half_width, seed, rotation):
        environment = self.environment
        length, width, height = environment.task.dimensions
        seat_height = height - min(.06, height / 4) / 2
        local = grasp_frames(np.zeros(3), 0., half_width, seat_height)
        wrist = Rotation.from_euler('y', -90., degrees=True).as_matrix()
        positions = [origin + rotation @ p for p in local[:2]]
        x_offset = float(getattr(environment,'edge_grasp_x_offset',0.))
        positions = [p+rotation@np.array((x_offset,0.,0.)) for p in positions]
        frames = [rotation @ wrist @ frame for frame in local[2:]]
        reference = getattr(environment, 'edge_commanded_grasp_frames', None)
        if reference is not None:
            lateral = half_width - (width/2-self.insertion_m)
            positions = [origin + rotation @ (frame[:3, 3]+np.array((0., side*lateral, 0.)))
                         for frame, side in zip(reference, (1., -1.))]
            frames = [rotation @ frame[:3, :3] for frame in reference]
        base_position, quaternion = environment.base.get_world_pose()
        base_rotation = environment.rotation(quaternion)
        obstacle = np.eye(4)
        obstacle[:3, :3] = base_rotation.T @ rotation
        obstacle[:3, 3] = base_rotation.T @ (origin + rotation @ np.array((0., 0., height/2)) - base_position)
        result = environment.bimanual.solve(
            *[base_rotation.T @ (p - base_position) for p in positions],
            *[base_rotation.T @ frame for frame in frames], seed, environment.collision_model,
            obstacle, environment.task.dimensions, clearance=.05,
            upper_limits=np.r_[min(environment.bimanual.upper[0], 0.), environment.bimanual.upper[1:]],
            reseed_torso=True, contact_normal=base_rotation.T @ rotation[:, 1], pose_weights=(30., 3.))
        if environment.recorder is not None:
            environment.recorder.event('ik_solution', {'mode': 'stool_seat_edge',
                'object_origin': np.asarray(origin).tolist(), 'half_width_m': half_width,
                'attempts': environment.bimanual.last_solve_diagnostics})
        return result

    def targets(self, mode, center, initial, approach_clearance_m=None, release_strategy=None):
        environment = self.environment
        strategy = (getattr(environment, 'edge_release_strategy', 'direct')
                    if release_strategy is None else release_strategy)
        if strategy not in ('direct', 'staged', 'staged_high'):
            raise ValueError('Unsupported stool release strategy')
        if mode=='place':
            self.refresh_commanded_grasp_reference()
        self.refresh_finger_tracking()
        clearance = .13 if approach_clearance_m is None else approach_clearance_m
        outside = environment.task.dimensions[1] / 2 + clearance
        inserted = environment.task.dimensions[1] / 2 - self.insertion_m
        opened, closed = self.finger_targets(True), self.finger_targets(False)
        rotation = environment.rotation(environment.object.get_world_pose()[1])
        if mode == 'grasp':
            rotation = yaw_rotation(quaternion_yaw(environment.object.get_world_pose()[1]))
            approach = np.r_[self.solve_edge(center, outside, initial[:15], rotation), opened]
            insert = np.r_[self.solve_edge(center, inserted, approach[:15], rotation), opened]
            squeeze = np.r_[insert[:15], closed]
            box = np.eye(4); box[:3, :3] = rotation
            box[:3, 3] = center + rotation @ np.array((0., 0., environment.task.dimensions[2]/2))
            base = planar_transform(measured_base(environment), environment.base.get_world_pose()[0][2])
            self.check_lift_grip(squeeze, base, box)
            targets, seed = [approach, insert, squeeze], squeeze[:15]
            for dz in np.linspace(0., .22, self.lift_segments+1)[1:]:
                seed = self.solve_edge(center + np.array((0., 0., dz)), inserted, seed, rotation)
                targets.append(np.r_[seed, closed])
            return (targets, ['approach', 'approach', 'squeeze'] + ['lift']*self.lift_segments,
                    [5., 4., 2.] + [.8]*self.lift_segments)
        floor = center.copy()
        floor[:2] = object_transform(environment)[:2, 3]
        floor[2] = .0001
        yaw = quaternion_yaw(environment.object.get_world_pose()[1])
        orientations = Slerp([0., 1.], Rotation.from_matrix(np.stack((rotation, yaw_rotation(yaw)))))
        steps = max(2, math.ceil(abs(center[2] - floor[2]) / .04))
        targets, seed = [], initial[:15]
        for progress in np.linspace(0., 1., steps+1)[1:]:
            seed = self.solve_edge(center + progress*(floor-center), inserted, seed, orientations(progress).as_matrix())
            targets.append(np.r_[seed, closed])
        # Open in place after support, then withdraw horizontally and retract.
        unpinch = np.r_[seed, opened]
        duration = max(.8, 4./steps, abs(center[2]-floor[2])/(steps*.025))
        if strategy != 'direct':
            # Clear the seat edge before raising the open hands. A full-width
            # floor-level withdrawal can lean the torso into the placed object.
            short_width = min(outside, environment.task.dimensions[1]/2 + .015)
            rise = .06 if strategy == 'staged' else .10
            level = yaw_rotation(yaw)
            short = self.solve_edge(floor, short_width, seed, level)
            raised_origin = floor + np.array((0., 0., rise))
            raised = self.solve_edge(raised_origin, short_width, short, level)
            withdrawn = self.solve_edge(raised_origin, outside, raised, level)
            staged = [np.r_[q, opened] for q in (short, raised, withdrawn)]
            return (targets + [unpinch, *staged, environment.initial_control.copy()],
                    ['lower']*steps + ['release', 'release', 'retract', 'retract', 'retract'],
                    [duration]*steps + [2., 2., max(2., rise/.03), 3., 5.])
        release = np.r_[self.solve_edge(floor, outside, seed, yaw_rotation(yaw)), opened]
        return (targets + [unpinch, release, environment.initial_control.copy()],
                ['lower']*steps + ['release', 'release', 'retract'], [duration]*steps + [2., 4., 5.])

    def check_lift_grip(self, joints, base, box, progress=None):
        environment = self.environment
        height = environment.task.dimensions[2]
        origin = box[:3, 3] - box[:3, :3] @ np.array((0., 0., height/2))
        seat_height = height - min(.06, height/4)/2
        local = grasp_frames(np.zeros(3), 0., environment.task.dimensions[1]/2-self.insertion_m, seat_height)
        wrist = Rotation.from_euler('y', -90., degrees=True).as_matrix()
        reference = getattr(environment, 'edge_commanded_grasp_frames', None)
        x_offset = float(getattr(environment,'edge_grasp_x_offset',0.))
        desired_positions = ([p+np.array((x_offset,0.,0.)) for p in local[:2]] if reference is None
                             else [frame[:3, 3] for frame in reference])
        desired_frames = [wrist @ frame for frame in local[2:]] if reference is None else [frame[:3, :3] for frame in reference]
        errors, angles = [], []
        for tool, desired, frame in zip(environment.bimanual.forward(joints[:15]), desired_positions, desired_frames):
            actual = base @ tool
            errors.append(box[:3, :3].T @ (actual[:3, 3]-origin)-desired)
            angles.append(float(np.linalg.norm(Rotation.from_matrix(
                (box[:3, :3] @ frame) @ actual[:3, :3].T).as_rotvec())))
        errors = np.asarray(errors)
        normal_error = float(np.max(np.abs(errors[:, 1])))
        details = {'grip_span_error_m': normal_error, 'grip_span_tolerance_m': .001,
                   'edge_position_errors_m': errors.tolist(), 'edge_orientation_errors_rad': angles}
        if (not np.isfinite(errors).all() or not np.isfinite(angles).all() or normal_error > .001
                or np.max(np.abs(errors[:, (0, 2)])) > .003 or max(angles) > .05):
            raise LiftGripPathError(details, progress)
        return details


class SeatEdgeArmExecutePolicy(ArmExecutePolicy):
    calibration_preload_m = .0009
    calibration_max_iterations = 12

    def support_target_x(self):
        """Small tangent correction for short seats; grip preload and safety limits stay fixed.

        Recorded short stools consistently pitch backwards under load despite
        centered static distance witnesses. This bounded initial correction is
        a candidate calibration parameter, still subject to physical validation.
        """
        height = float(self.environment.task.dimensions[2])
        return -.001 * float(np.clip((.36-height)/.08, 0., 1.))

    def support_alignment(self,transforms,measured,box,*,object_frame=True):
        """Query the same support geometry in object coordinates for stable witnesses.

        The legacy world frame is retained only for historical diagnostics.
        """
        if not isinstance(object_frame,bool):
            raise RuntimeError('INVALID_SUPPORT_QUERY_FRAME')
        environment=self.environment;model=environment.collision_model
        _,touches=self.finger_touch_targets(transforms,measured,box)
        rotation=box[:3,:3];origin=box[:3,3]-rotation@np.array((0.,0.,environment.task.dimensions[2]/2))
        seat_transform=(model.fcl.Transform(np.eye(3),np.zeros(3)) if object_frame
                        else model.fcl.Transform(rotation,origin))
        seat=model.fcl.CollisionObject(model._stool_geometries(environment.task.dimensions)['Seat'],seat_transform)
        request=model.fcl.DistanceRequest(enable_signed_distance=True,enable_nearest_points=True)
        points=[];gaps=[]
        for index,name,sign in ((0,'left_gripper_finger_link1',1.),(3,'right_gripper_finger_link2',-1.)):
            pose=transforms[name].copy();aperture=min(.05,touches[index]['touch_aperture_m']+.0003)
            pose[:3,3]+=transforms[name][:3,1]*(sign*aperture-measured[15+index])
            if object_frame:
                local=np.eye(4);local[:3,:3]=rotation.T@pose[:3,:3]
                local[:3,3]=rotation.T@(pose[:3,3]-origin)
                pose=local
            finger=model.fcl.CollisionObject(model.envelopes[name],model.fcl.Transform(pose[:3,:3],pose[:3,3]))
            result=model.fcl.DistanceResult();gap=float(model.fcl.distance(finger,seat,request,result))
            point=np.asarray(result.nearest_points[1])
            if not object_frame:
                point=rotation.T@(point-origin)
            bottom=environment.task.dimensions[2]-min(.06,environment.task.dimensions[2]/4)
            if not np.isfinite(point).all() or not 0.<gap<.001 or abs(point[2]-bottom)>.001:
                raise RuntimeError('UNRESOLVED_LOWER_SEAT_SUPPORT_POINT')
            points.append(point)
            gaps.append(gap)
        left,right=points
        if left[1]<=0. or right[1]>=0. or left[1]-right[1]<.1:
            raise RuntimeError('SUPPORT_POINTS_DO_NOT_STRADDLE_COM')
        alpha=left[1]/(left[1]-right[1]);support=(1-alpha)*left+alpha*right
        return {'support_x_error_m':float(support[0]),'lower_contact_points_object':[p.tolist() for p in points],
                'com_horizontal_target_m':[0.,0.],'native_geometry_probe':True,
                'support_query_frame':'object' if object_frame else 'legacy_world',
                'support_query_gaps_m':gaps}

    def local_support_target(self, initial, correction, box, *, world_reference=None):
        """Translate the last planned world frames, retaining compensation for base drift."""
        environment=self.environment
        if (np.shape(initial)!=(19,) or not np.isfinite(initial).all()
                or not np.isfinite(correction) or abs(correction)>.01):
            raise RuntimeError('INVALID_OPEN_SUPPORT_CORRECTION')
        kinematics=environment.bimanual
        position,quaternion=environment.base.get_world_pose()
        base_rotation=environment.rotation(quaternion)
        if world_reference is None:
            frames=kinematics.forward(initial[:15])
        else:
            reference=np.asarray(world_reference,dtype=float)
            if reference.shape!=(2,4,4) or not np.isfinite(reference).all():
                raise RuntimeError('INVALID_OPEN_SUPPORT_REFERENCE')
            inverse_base=np.eye(4);inverse_base[:3,:3]=base_rotation.T
            inverse_base[:3,3]=-base_rotation.T@position
            frames=tuple(inverse_base@frame for frame in reference)
        delta=base_rotation.T@box[:3,0]*correction
        positions=[frame[:3,3]+delta for frame in frames]
        # IK proposes a local pose; the unchanged full trajectory gate below
        # still decides whether the motion is allowed. No actuator gain changes.
        joints=kinematics.solve(*positions,*[frame[:3,:3] for frame in frames],initial[:15],
            upper_limits=np.r_[min(kinematics.upper[0],0.),kinematics.upper[1:]],
            reseed_torso=False,pose_weights=(30.,3.))
        for actual,position,reference in zip(kinematics.forward(joints),positions,frames):
            distance=float(np.linalg.norm(actual[:3,3]-position))
            angle=float(np.linalg.norm(Rotation.from_matrix(reference[:3,:3]@actual[:3,:3].T).as_rotvec()))
            if not np.isfinite((distance,angle)).all() or distance>.00002 or angle>.0001:
                raise RuntimeError('OPEN_SUPPORT_LOCAL_IK_INACCURATE')
        return np.r_[joints,SeatEdgeArmPlanPolicy(environment).finger_targets(True)]

    def certify_support_translation(self, initial, target, transforms, box):
        """Retain the original nominal check and add all measured link offsets."""
        environment=self.environment;model=environment.collision_model
        planner=SeatEdgeArmPlanPolicy(environment)
        center,quaternion=environment.object.get_world_pose()
        base=planar_transform(measured_base(environment),environment.base.get_world_pose()[0][2])
        audit=planner.certify('grasp',initial,[target],['approach'],base,center,
                              quaternion_yaw(quaternion),box)
        if not set(model.envelopes).issubset(transforms):
            raise RuntimeError('INCOMPLETE_NATIVE_SUPPORT_GEOMETRY')
        position,quaternion=environment.base.get_world_pose()
        actual_base=np.eye(4);actual_base[:3,3]=position;actual_base[:3,:3]=environment.rotation(quaternion)
        names=environment.bimanual.names+[f'{side}_gripper_finger_joint{i}' for side in ('left','right') for i in (1,2)]
        nominal=model.forward(dict(zip(names,initial)),actual_base)
        residuals={name:np.linalg.inv(nominal[name])@pose for name,pose in transforms.items()}
        for progress in np.linspace(0.,1.,101):
            command=initial+progress*(target-initial)
            predicted=model.forward(dict(zip(names,command)),actual_base)
            for name,residual in residuals.items():
                predicted[name]=predicted[name]@residual
            details=check_collision_transforms(environment,predicted,box)
            audit.append({'phase':'approach','progress':float(progress),
                          'geometry_model':'native_link_residual',**details})
        return audit

    def center_support_open(self):
        environment=self.environment
        world_reference=None
        for iteration in range(5):
            if environment.object_bottom()>.02:
                raise RuntimeError('SUPPORT_ALIGNMENT_REQUIRES_GROUND')
            transforms,measured,box=self.measured_calibration_state()
            alignment=self.support_alignment(transforms,measured,box)
            open_target=float(getattr(environment,'edge_support_open_target_x',0.))
            error=alignment['support_x_error_m']-open_target
            environment.recorder.event('support_alignment_probe',{'iteration':iteration,
                'open_support_target_x_m':open_target,'target_error_m':error,**alignment})
            if abs(error)<=.00005:
                return
            if iteration==4:
                break
            correction=-float(np.clip(error,-.01,.01))
            offset=float(getattr(environment,'edge_grasp_x_offset',0.))+correction
            if abs(offset)>.02:
                raise RuntimeError('SUPPORT_ALIGNMENT_OUTSIDE_BOUNDED_REACH')
            initial=environment.command[environment.control_indices].copy()
            target=self.local_support_target(initial,correction,box,world_reference=world_reference)
            audit=self.certify_support_translation(initial,target,transforms,box)
            # Save the planned pose before physics moves the base. Rebuilding it
            # from the post-motion base would discard the accumulated correction.
            position,quaternion=environment.base.get_world_pose()
            planning_base=np.eye(4);planning_base[:3,3]=position
            planning_base[:3,:3]=environment.rotation(quaternion)
            world_reference=tuple(planning_base@tool for tool in environment.bimanual.forward(target[:15]))
            environment.edge_grasp_x_offset=offset
            environment.recorder.event('support_alignment_motion',{'iteration':iteration,'hand_x_offset_m':offset,
                'correction_mode':'local_commanded_tool_translation','requested_translation_m':correction,
                'reference_frame':'last_planned_world',
                'world_commanded_tool_frames':[frame.tolist() for frame in world_reference],
                'native_link_offsets_checked':True,
                'samples':len(audit),'minimum_body_gap_m':min(a['minimum_noncontact_clearance_m'] for a in audit),
                'minimum_wall_gap_m':min(a['minimum_wall_clearance_m'] for a in audit)})
            self.interpolate(target,.8,'approach')
        raise RuntimeError('OPEN_SUPPORT_ALIGNMENT_NOT_CONVERGED')

    def reopen_for_alignment(self):
        environment=self.environment
        if environment.object_bottom()>.02:
            raise RuntimeError('REALIGNMENT_REQUIRES_GROUND_SUPPORT')
        transforms,measured,box=self.measured_calibration_state()
        target=environment.command[environment.control_indices].copy()
        target[15:]=SeatEdgeArmPlanPolicy(environment).finger_targets(True)
        self.certify_calibration(transforms,measured,target,box)
        self.interpolate(target,.8,'approach')
        for name in ('edge_commanded_grasp_frames','edge_calibrated_grasp_rotations','edge_native_grasp_frames','edge_finger_command','edge_finger_tracking_residuals','edge_world_lift_rotation'):
            if hasattr(environment,name):delattr(environment,name)

    def prepare_centered_grasp(self):
        environment=self.environment
        support_target = self.support_target_x()
        environment.edge_support_open_target_x = support_target
        for attempt in range(3):
            self.center_support_open()
            self.calibrate_contact()
            transforms,measured,box=self.measured_calibration_state()
            alignment=self.support_alignment(transforms,measured,box)
            error = alignment['support_x_error_m'] - support_target
            environment.recorder.event('loaded_support_alignment',{'attempt':attempt,
                'support_target_x_m':support_target, 'target_error_m':error, **alignment})
            if abs(error)<=.0001:
                return
            if attempt<2:
                bias=float(getattr(environment,'edge_support_open_target_x',0.))-error
                if abs(bias)>.005:
                    raise RuntimeError('SUPPORT_CLOSURE_BIAS_LIMIT')
                environment.edge_support_open_target_x=bias
                environment.recorder.event('closing_support_bias',{'attempt':attempt,
                    'loaded_support_error_m':alignment['support_x_error_m'],'next_open_support_target_x_m':bias,
                    'support_target_x_m':support_target, 'target_error_m':error,
                    'loaded_alignment_tolerance_m':.0001,'grip_preload_unchanged_m':self.calibration_preload_m})
                self.reopen_for_alignment()
        raise RuntimeError('GRASP_SUPPORT_NOT_CENTERED')

    def measured_calibration_state(self):
        environment = self.environment
        positions, quaternions = environment.collision_links.get_world_poses()
        transforms = {}
        for name, position, quaternion in zip(environment.collision_link_names, positions, quaternions):
            matrix = np.eye(4); matrix[:3, 3] = position; matrix[:3, :3] = environment.rotation(quaternion)
            transforms[name] = matrix
        joints = environment.robot.get_joint_positions()[environment.control_indices].copy()
        return transforms, joints, object_transform(environment)

    def finger_touch_targets(self, transforms, measured_joints, box):
        environment = self.environment
        model = environment.collision_model
        rotation = box[:3, :3]
        origin = box[:3, 3]-rotation @ np.array((0.,0.,environment.task.dimensions[2]/2))
        parts = [model.fcl.CollisionObject(g, model.fcl.Transform(rotation, origin))
                 for g in model._stool_geometries(environment.task.dimensions).values()]
        request = model.fcl.DistanceRequest(enable_signed_distance=True)
        targets, details = [], []
        for index, (side, number) in enumerate((('left',1),('left',2),('right',1),('right',2))):
            name = f'{side}_gripper_finger_link{number}'
            pose = transforms[name]
            sign = 1. if number == 1 else -1.
            actual = float(measured_joints[15+index])
            def gap(aperture):
                moved = pose.copy(); moved[:3,3] += pose[:3,1]*(sign*aperture-actual)
                finger = model.fcl.CollisionObject(model.envelopes[name], model.fcl.Transform(moved[:3,:3],moved[:3,3]))
                return min(float(model.fcl.distance(finger, part, request, model.fcl.DistanceResult())) for part in parts)
            lower, upper = 0., .05
            if not (gap(lower) < 0. and gap(upper) > 0.):
                raise RuntimeError(f'UNBRACKETED_FINGER_TOUCH: {name}')
            for _ in range(18):
                midpoint = (lower+upper)/2
                if gap(midpoint) > 0.: upper = midpoint
                else: lower = midpoint
            touch = (lower+upper)/2
            target = touch-self.calibration_preload_m
            if not self.calibration_preload_m < touch < .05 or gap(target) < -.002:
                raise RuntimeError(f'UNCERTIFIED_FINGER_TOUCH: {name}')
            targets.append(sign*target)
            details.append({'finger':name, 'touch_aperture_m':touch, 'command_q_m':sign*target,
                            'geometric_closure_m':self.calibration_preload_m})
        return np.asarray(targets), details

    def certify_calibration(self, transforms, measured, target, box):
        audit = []
        for progress in np.linspace(0.,1.,101):
            predicted = {name:pose.copy() for name,pose in transforms.items()}
            for index,(side,number) in enumerate((('left',1),('left',2),('right',1),('right',2))):
                name=f'{side}_gripper_finger_link{number}'
                predicted[name][:3,3] += transforms[name][:3,1]*progress*(target[15+index]-measured[15+index])
            audit.append(check_collision_transforms(self.environment,predicted,box))
        return audit

    def freeze_commanded_grasp(self):
        environment = self.environment
        command = environment.command[environment.control_indices].copy()
        base_position, base_quaternion = environment.base.get_world_pose()
        base = np.eye(4); base[:3,:3] = environment.rotation(base_quaternion); base[:3,3] = base_position
        origin = object_transform(environment)
        origin[:3,3] -= origin[:3,:3] @ np.array((0.,0.,environment.task.dimensions[2]/2))
        # Preserve commanded arm poses/preload, not measured joint deflections.
        environment.edge_commanded_grasp_frames = tuple(np.linalg.inv(origin) @ base @ tool
            for tool in environment.bimanual.forward(command[:15]))
        environment.edge_calibrated_grasp_rotations = tuple(frame[:3,:3].copy()
            for frame in environment.edge_commanded_grasp_frames)
        environment.edge_finger_command = command[15:].copy()
        environment.recorder.event('commanded_grasp_reference', {
            'finger_commands_m':command[15:].tolist(),
            'tool_frames_in_object': [frame.tolist() for frame in environment.edge_commanded_grasp_frames],
            'object_attachment':False})

    def capture_retention_reference(self):
        environment=self.environment
        origin=object_transform(environment)
        origin[:3,3]-=origin[:3,:3]@np.array((0.,0.,environment.task.dimensions[2]/2))
        p,q=environment.base.get_world_pose();base=np.eye(4);base[:3,3]=p;base[:3,:3]=environment.rotation(q)
        measured=environment.robot.get_joint_positions()[environment.control_indices]
        environment.edge_native_grasp_frames=tuple(np.linalg.inv(origin)@base@tool for tool in environment.bimanual.forward(measured[:15]))
        environment.edge_world_lift_rotation=yaw_rotation(quaternion_yaw(environment.object.get_world_pose()[1]))
        environment.recorder.event('native_grasp_retention_reference',{
            'control_step':environment.elapsed_steps if hasattr(environment,'elapsed_steps') else None,
            'tool_frames_in_object':[t.tolist() for t in environment.edge_native_grasp_frames],
            'world_lift_rotation_reference':environment.edge_world_lift_rotation.tolist(),
            'maximum_finger_gap_m':.001,'maximum_tangential_slip_m':.003,
            'maximum_relative_rotation_rad':.05,'maximum_object_tilt_deg':5.,
            'maximum_object_angular_speed_rad_s':.35,'calibration_still_requires_four_loaded_fingers':True})

    def calibrate_contact(self):
        environment = self.environment
        stable = 0
        for iteration in range(self.calibration_max_iterations):
            if environment.object_bottom() > .02:
                raise RuntimeError('CALIBRATION_REQUIRES_GROUND_SUPPORT')
            transforms, measured, box = self.measured_calibration_state()
            desired, details = self.finger_touch_targets(transforms,measured,box)
            command = environment.command[environment.control_indices].copy()
            target = command.copy()
            # Closing <=2mm over.4s; opening releases excess pressure directly.
            signs = np.array((1.,-1.,1.,-1.))
            target[15:] = signs*np.maximum(signs*command[15:]-.002,signs*desired)
            audit = self.certify_calibration(transforms,measured,target,box)
            environment.recorder.event('finger_touch_calibration', {'iteration':iteration,'fingers':details,
                'samples':len(audit),'target_q_m':target[15:].tolist(),
                'maximum_geometric_closure_m':self.calibration_preload_m,
                'minimum_body_gap_m':min(a['minimum_noncontact_clearance_m'] for a in audit),
                'minimum_wall_gap_m':min(a['minimum_wall_clearance_m'] for a in audit)})
            self.interpolate(target,.4,'squeeze')
            close = (np.max(np.abs(desired-target[15:])) <= .0001
                     and np.max(np.abs(target[15:]-command[15:])) <= .0001)
            stable = stable+1 if close and grasp_contact_established(environment) else 0
            if stable >= 2:
                self.freeze_commanded_grasp()
                self.capture_retention_reference()
                return
        raise RuntimeError('FOUR_FINGER_CALIBRATION_NOT_CONVERGED')

    def interpolate(self, target, duration, phase):
        if phase != 'lower':
            return super().interpolate(target, duration, phase)
        environment = self.environment
        previous = environment.command[environment.control_indices].copy()
        steps = round(duration/environment.control_dt)
        for index in range(steps):
            progress = (index+1)/steps
            blend = progress**3*(10-15*progress+6*progress**2)
            environment.advance(previous+blend*(target-previous), phase, progress)
            if environment.object_bottom() > .02 and not grasp_contact_retained(environment):
                raise RuntimeError('FOUR_EDGE_CONTACTS_LOST_DURING_LOWER')
        self.phase_end(phase)

    def grasp_motion(self, plan):
        if plan['phases'][:3] != ['approach', 'approach', 'squeeze']:
            raise RuntimeError('INVALID_SEAT_EDGE_GRASP_SEQUENCE')
        for target, duration, phase in zip(plan['targets'][:2], plan['durations'][:2], plan['phases'][:2]):
            self.interpolate(target, duration, phase)
        environment = self.environment
        self.prepare_centered_grasp()
        if not grasp_contact_established(environment):
            raise RuntimeError('FOUR_EDGE_CONTACTS_MISSING_BEFORE_LIFT')
        environment.recorder.event('lift_strategy', {'strategy': 'measured_cartesian_seat_edge_grasp',
            'insertion_m': SeatEdgeArmPlanPolicy.insertion_m, 'finger_closure_preload_approx_m': .000904})
        initial_height = float(environment.object.get_world_pose()[0][2])
        for segment in range(SeatEdgeArmPlanPolicy.lift_segments):
            self.lift_segment(initial_height+.22*(segment+1)/SeatEdgeArmPlanPolicy.lift_segments, segment)

    def lift_segment(self, target_origin_height, segment, depth=0):
        environment = self.environment
        center, quaternion = environment.object.get_world_pose()
        displacement = float(target_origin_height-center[2])
        initial = environment.command[environment.control_indices].copy()
        planner = SeatEdgeArmPlanPolicy(environment)
        planner.refresh_commanded_grasp_reference()
        planner.refresh_finger_tracking()
        origin_target,rotation_target=self.lift_pose_target(target_origin_height,center,environment.rotation(quaternion),initial)
        target = np.r_[planner.solve_edge(origin_target,
            environment.task.dimensions[1]/2-planner.insertion_m, initial[:15], rotation_target),
            planner.finger_targets(False)]
        base = planar_transform(measured_base(environment), environment.base.get_world_pose()[0][2])
        try:
            audit = planner.certify('grasp', initial, [target], ['lift'], base, center,
                quaternion_yaw(quaternion), object_transform(environment), lift_distance=displacement,
                lift_origin_target=origin_target,lift_rotation_target=rotation_target)
        except LiftGripPathError as error:
            if error.progress == 0. or depth >= 3 or abs(displacement) <= .005 or not np.isfinite(error.details['grip_span_error_m']):
                raise
            midpoint = float(center[2]+displacement/2)
            environment.recorder.event('lift_refinement', {'segment':segment, 'depth':depth,
                'target_origin_height_m':target_origin_height, 'midpoint_origin_height_m':midpoint,
                'rejected_sample_progress':error.progress, 'planning_error':error.details})
            self.lift_segment(midpoint,segment,depth+1)
            self.lift_segment(target_origin_height,segment,depth+1)
            return
        (environment.output/f'arm_plan_lift_{environment.elapsed_steps:05d}.json').write_text(json.dumps(audit,indent=2))
        environment.recorder.event('lift_replan', {'segment':segment, 'subdivision_depth':depth,
            'target_origin_height_m':target_origin_height, 'measured_object_origin':center.tolist(),
            'target_object_origin':origin_target.tolist(),'target_object_rotation':rotation_target.tolist(),
            'orientation_strategy':'world_level_with_commanded_grip_center_anchor',
            'minimum_body_gap_m':min(row['minimum_noncontact_clearance_m'] for row in audit),
            'minimum_wall_gap_m':min(row['minimum_wall_clearance_m'] for row in audit)})
        self.interpolate(target,.8,'lift')
        if not grasp_contact_retained(environment):
            raise RuntimeError('FOUR_EDGE_CONTACTS_LOST_DURING_LIFT')

    def lift_pose_target(self,height,center,measured_rotation,command):
        """Keep level targets fixed instead of integrating measured object tilt."""
        environment=self.environment
        rotation=np.asarray(getattr(environment,'edge_world_lift_rotation',measured_rotation)).copy()
        origin=np.asarray(center,dtype=float).copy();origin[2]=height
        reference=getattr(environment,'edge_commanded_grasp_frames',None)
        if reference is not None and hasattr(environment,'edge_world_lift_rotation'):
            p,q=environment.base.get_world_pose();base=np.eye(4);base[:3,3]=p;base[:3,:3]=environment.rotation(q)
            tools=[base@tool for tool in environment.bimanual.forward(command[:15])]
            world_center=(tools[0][:3,3]+tools[1][:3,3])/2
            local_center=(reference[0][:3,3]+reference[1][:3,3])/2
            origin[:2]=(world_center-rotation@local_center)[:2]
        return origin,rotation


def primitive_registry(environment):
    if environment.task.object_kind == 'stool':
        environment.grasp_mode = 'stool_seat_edge'
        return {'base_move': BaseMovePolicy(environment), 'arm_plan': SeatEdgeArmPlanPolicy(environment),
                'grasp_lift': SeatEdgeArmExecutePolicy(environment, 'grasp'),
                'place_release': SeatEdgeArmExecutePolicy(environment, 'place')}
    return {'base_move': BaseMovePolicy(environment), 'arm_plan': ArmPlanPolicy(environment),
            'grasp_lift': ArmExecutePolicy(environment, 'grasp'), 'place_release': ArmExecutePolicy(environment, 'place')}
