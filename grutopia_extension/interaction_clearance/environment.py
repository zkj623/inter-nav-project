"""One physical world shared by skill execution, manipulation, and local driving."""

import json
import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.task import evaluate_clearance, yaw_rotation
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox
from grutopia_extension.manipulation.r1lite_bimanual_lift_sim import R1LiteBimanualLift


class ClearanceEnvironment(R1LiteBimanualLift):
    def __init__(self, output, task):
        from pxr import Gf, UsdGeom
        from grutopia_extension.configs.robots.r1lite import R1LITE_STEER_JOINTS

        task.validate()
        self.task = task
        self.recorder = None
        self.base_twist = np.zeros(3)
        self.wall_objects = {}
        self.safety_ok = True
        self.stable_seconds = 0.0
        self.elapsed_steps = 0
        self.interaction_state = 'ready'
        self.arm_plan = None
        self.selected_place_xy = np.asarray(task.place_xy, dtype=float)
        box = BimanualBox(task.dimensions, task.mass_kg, min(0.25, task.dimensions[2] - 0.03), 0.003)
        robot_quaternion = Rotation.from_euler('z', task.robot_yaw_rad).as_quat()[[3, 0, 1, 2]]
        object_quaternion = Rotation.from_euler('z', task.object_yaw_rad).as_quat()[[3, 0, 1, 2]]
        super().__init__(output, box=box, object_xy=task.object_xy,
                         robot_position=np.array((*task.robot_xy, 0.025)), robot_orientation=robot_quaternion,
                         object_orientation=object_quaternion)
        self.steering_indices = np.array([self.robot.get_dof_index(name) for name in R1LITE_STEER_JOINTS])
        self.initial_control = self.command[self.control_indices].copy()
        self.start_base_position, self.start_base_quaternion = self.base.get_world_pose()
        framing = [np.asarray(task.robot_xy), np.asarray(task.object_xy), np.asarray(task.place_xy)]
        for obstacle in task.scene_obstacles:
            center, half = np.asarray(obstacle.position[:2]), np.asarray(obstacle.dimensions[:2]) / 2
            framing.extend((center - half, center + half))
        lower, upper = np.min(framing, axis=0) - 1.8, np.max(framing, axis=0) + 1.8
        map_center = (lower + upper) / 2
        map_span = max(upper[1] - lower[1], (upper[0] - lower[0]) * 720 / 960)
        for path, position, target, focal in (
            ('/World/BimanualOverview', (2.65, -2.0, 1.65), (0.6, 0.3, 0.45), 26),
            ('/World/BimanualFront', (1.9, -1.6, 2.55), (0.7, 0.3, 0.15), 26),
        ):
            camera = UsdGeom.Camera.Get(self.stage, path)
            up = (0, 0, 1)
            if path == '/World/BimanualFront':
                focal = 20
                height = max(5.0, map_span * focal / float(camera.GetVerticalApertureAttr().Get()) * 1.08 + 0.5)
                position, target, up = (*map_center, height), (*map_center, 0.0), (0, 1, 0)
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(*up))
            UsdGeom.Xformable(camera).MakeMatrixXform().Set(matrix.GetInverse())
            camera.GetFocalLengthAttr().Set(float(focal))
        for _ in range(10):
            self.advance(self.initial_control, 'settle', 0.0)
        self.elapsed_steps = 0
        self.initialization = self.initial_pose_metrics()
        if not self.initialization['passed']:
            raise RuntimeError(f'Initial robot/object poses not realized: {self.initialization}')
        (self.output / 'initialization.json').write_text(json.dumps(self.initialization, indent=2))
        (self.output / 'task.json').write_text(json.dumps(task.to_dict(), indent=2))
        self.stage.GetRootLayer().Export(str(self.output / 'scene.usda'))

    def configure_task_collisions(self, root, rigid_paths):
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

        super().configure_task_collisions(root, rigid_paths)
        for obstacle in self.task.scene_obstacles:
            name, position, dimensions = obstacle.name, obstacle.position, obstacle.dimensions
            shape = UsdGeom.Cube.Define(self.stage, f'/World/ClearanceScene/{name}')
            shape.CreateSizeAttr(1.0)
            shape.CreateDisplayColorAttr([Gf.Vec3f(0.68, 0.72, 0.77)])
            transform = UsdGeom.Xformable(shape)
            transform.AddTranslateOp().Set(Gf.Vec3d(*position))
            transform.AddScaleOp().Set(Gf.Vec3f(*dimensions))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim()).CreateContactOffsetAttr(0.002)
            self.wall_objects[name] = self.collision_model.fcl.CollisionObject(
                self.collision_model.fcl.Box(*dimensions), self.collision_model.fcl.Transform(np.eye(3), position)
            )
        for name, region, color in (('RequiredPassage', self.task.passage, (0.16, 0.55, 0.30)),
                                     ('AllowedPlacement', self.task.placement, (0.18, 0.38, 0.72))):
            center = (np.asarray(region.lower) + region.upper) / 2
            extent = np.asarray(region.upper) - region.lower
            for suffix, position, dimensions in (
                ('South', (center[0], region.lower[1], 0.001), (extent[0], 0.015, 0.002)),
                ('North', (center[0], region.upper[1], 0.001), (extent[0], 0.015, 0.002)),
                ('West', (region.lower[0], center[1], 0.001), (0.015, extent[1], 0.002)),
                ('East', (region.upper[0], center[1], 0.001), (0.015, extent[1], 0.002)),
            ):
                mark = UsdGeom.Cube.Define(self.stage, f'/World/ClearanceScene/{name}{suffix}')
                mark.CreateSizeAttr(1.0)
                mark.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                transform = UsdGeom.Xformable(mark)
                transform.AddTranslateOp().Set(Gf.Vec3d(*position))
                transform.AddScaleOp().Set(Gf.Vec3f(*dimensions))

    def collision_sample(self):
        try:
            sample = super().collision_sample()
            transforms = {}
            for name, position, quaternion in zip(self.collision_link_names, sample['link_positions'],
                                                  sample['link_quaternions_wxyz']):
                transform = np.eye(4)
                transform[:3, :3] = self.rotation(quaternion)
                transform[:3, 3] = position
                transforms[name] = transform
            objects = self.collision_model.objects(transforms)
            object_rotation = self.rotation(sample['object_quaternion_wxyz'])
            object_center = np.asarray(sample['object_position']) + object_rotation @ np.array((0, 0, self.box.dimensions[2] / 2))
            objects['task_object'] = self.collision_model.fcl.CollisionObject(
                self.collision_model.fcl.Box(*self.box.dimensions),
                self.collision_model.fcl.Transform(object_rotation, object_center)
            )
            minimum = float('inf')
            nearest = None
            request = self.collision_model.fcl.DistanceRequest(enable_signed_distance=True)
            for wall_name, wall in self.wall_objects.items():
                for name, body in objects.items():
                    gap = float(self.collision_model.fcl.distance(body, wall, request,
                                                                self.collision_model.fcl.DistanceResult()))
                    if gap < minimum:
                        minimum, nearest = gap, [name, wall_name]
            sample['minimum_wall_clearance_m'] = minimum
            sample['nearest_wall_pair'] = nearest
            if minimum < 0.005:
                raise RuntimeError(f'Wall clearance violation: {nearest}, {minimum:.6f} m')
            return sample
        except RuntimeError:
            self.safety_ok = False
            raise

    def prepare_base_motion(self):
        from grutopia_extension.configs.robots.r1lite import R1LITE_MODULE_POSITIONS
        from grutopia_extension.interaction_clearance.motion_geometry import bounded_swerve_targets

        if not hasattr(self, 'steering_indices'):
            return np.zeros(3)
        angles = self.robot.get_joint_positions()[self.steering_indices]
        try:
            targets, velocities = bounded_swerve_targets(self.base_twist, angles, R1LITE_MODULE_POSITIONS,
                                                         wheel_radius=0.07, max_wheel_speed=2.0)
        except ValueError as error:
            self.safety_ok = False
            raise RuntimeError(f'BASE_DRIVE_REJECTED: {error}') from error
        self.command[self.steering_indices] = targets
        return velocities

    def observe(self):
        position, quaternion = self.object.get_world_pose()
        base_position, base_quaternion = self.base.get_world_pose()
        base_rotation = self.rotation(base_quaternion)
        relative_position = base_rotation.T @ (position - base_position)
        relative_quaternion = Rotation.from_matrix(base_rotation.T @ self.rotation(quaternion)).as_quat()[[3, 0, 1, 2]]
        velocity = base_rotation.T @ self.base.get_linear_velocity()
        goal = base_rotation.T @ (np.array((*self.task.place_xy, 0.0)) - base_position)
        return {'joint_positions': self.robot.get_joint_positions()[self.control_indices].tolist(),
                'base_pose_world': [*base_position.tolist(), *base_quaternion.tolist()],
                'object_pose_world': [*position.tolist(), *quaternion.tolist()],
                'object_pose_in_base': [*relative_position.tolist(), *relative_quaternion.tolist()],
                'placement_goal_in_base': goal.tolist(), 'measured_body_velocity': velocity.tolist(),
                'object_linear_velocity': self.object.get_linear_velocity().tolist(),
                'object_angular_velocity': self.object.get_angular_velocity().tolist(),
                'hand_normal_force_vectors': self.finger_force_vectors.reshape(2, 2, 3).sum(axis=1).tolist(),
                'finger_normal_force_vectors': self.finger_force_vectors.tolist(),
                'interaction_state': self.interaction_state,
                'plan_mode': self.arm_plan['mode'] if self.arm_plan else None,
                'safety_ok': self.safety_ok}

    def initial_pose_metrics(self):
        from grutopia_extension.interaction_clearance.motion_geometry import quaternion_yaw, angle_error

        robot_position, robot_quaternion = self.base.get_world_pose()
        object_position, object_quaternion = self.object.get_world_pose()
        values = {'robot_position_error_m': float(np.linalg.norm(robot_position[:2] - self.task.robot_xy)),
                  'object_position_error_m': float(np.linalg.norm(object_position[:2] - self.task.object_xy)),
                  'robot_yaw_error_rad': abs(angle_error(quaternion_yaw(robot_quaternion), self.task.robot_yaw_rad)),
                  'object_yaw_error_rad': abs(angle_error(quaternion_yaw(object_quaternion), self.task.object_yaw_rad))}
        values['passed'] = (max(values['robot_position_error_m'], values['object_position_error_m']) < 0.02
                            and max(values['robot_yaw_error_rad'], values['object_yaw_error_rad']) < 0.03)
        values['robot_pose_world'] = [*robot_position.tolist(), *robot_quaternion.tolist()]
        values['object_pose_world'] = [*object_position.tolist(), *object_quaternion.tolist()]
        return values

    def solve_both(self, center, half_width, height, seed):
        from grutopia_extension.interaction_clearance.motion_geometry import quaternion_yaw, grasp_frames

        base_position, base_quaternion = self.base.get_world_pose()
        base_rotation = self.rotation(base_quaternion)
        yaw = quaternion_yaw(self.object.get_world_pose()[1])
        left_target, right_target, left_orientation, right_orientation = grasp_frames(center, yaw, half_width, height)
        obstacle = np.eye(4)
        obstacle[:3, :3] = base_rotation.T @ yaw_rotation(yaw)
        obstacle[:3, 3] = base_rotation.T @ (np.asarray(center) + np.array((
            0, 0, self.box.dimensions[2] / 2 + height - self.box.grasp_height)) - base_position)
        solution = self.bimanual.solve(base_rotation.T @ (left_target - base_position),
                                   base_rotation.T @ (right_target - base_position),
                                   base_rotation.T @ left_orientation, base_rotation.T @ right_orientation,
                                   seed, self.collision_model, obstacle, self.box.dimensions, clearance=0.05,
                                   upper_limits=np.r_[min(self.bimanual.upper[0], 0.0), self.bimanual.upper[1:]],
                                   reseed_torso=True)
        if self.recorder is not None:
            self.recorder.event('ik_solution', {'height_m': float(height), 'half_width_m': float(half_width),
                                               'attempts': self.bimanual.last_solve_diagnostics})
        return solution

    def evaluate(self):
        position, quaternion = self.object.get_world_pose()
        return evaluate_clearance(self.task, position, self.rotation(quaternion),
                                  float(np.linalg.norm(self.object.get_linear_velocity())),
                                  float(np.linalg.norm(self.object.get_angular_velocity())),
                                  float(np.linalg.norm(self.finger_force_vectors, axis=1).sum()),
                                  self.safety_ok, self.stable_seconds)

    def robot_passage_metrics(self):
        from grutopia_extension.interaction_clearance.placement import robot_passage_metrics
        from grutopia_extension.interaction_clearance.motion_geometry import planar_transform, quaternion_yaw

        position, quaternion = self.base.get_world_pose()
        base = planar_transform(np.r_[position[:2], quaternion_yaw(quaternion)], position[2])
        joints = dict(zip(self.robot.dof_names, self.robot.get_joint_positions()))
        return robot_passage_metrics(self.task, self.collision_model, self.collision_model.forward(joints, base))

    def wait(self, seconds):
        from grutopia_extension.interaction_clearance.skill_runtime import Outcome

        if not self.safety_ok:
            return Outcome(False, 'SAFETY_STOPPED')
        self.base_twist[:] = 0.0
        joints = self.command[self.control_indices].copy()
        steps = round(seconds / self.control_dt)
        for index in range(steps):
            self.advance(joints, 'settle', (index + 1) / max(steps, 1))
        return Outcome(True, 'WAIT_COMPLETE', {'seconds': steps * self.control_dt})

    def advance(self, joints, phase, progress):
        if self.elapsed_steps >= 7500:
            raise RuntimeError('Episode control-time budget exceeded')
        before = self.observe() if self.recorder else None
        super().step(joints)
        self.elapsed_steps += 1
        metrics = self.evaluate()
        self.stable_seconds = self.stable_seconds + self.control_dt if metrics['physically_complete'] else 0.0
        if self.recorder:
            action = np.r_[self.command[self.control_indices], self.base_twist]
            self.recorder.capture(self, phase, float(progress), before, action)

    def save_collision_audit(self):
        report = super().save_collision_audit()
        gaps = [sample['minimum_wall_clearance_m'] for sample in self.physics_audit
                if 'minimum_wall_clearance_m' in sample]
        report['minimum_wall_clearance_m'] = min(gaps) if gaps else None
        report['wall_samples'] = len(gaps)
        report['passed'] = bool(report['passed'] and self.safety_ok and gaps and min(gaps) >= 0.005)
        (self.output / 'collision_audit.json').write_text(json.dumps(report, indent=2))
        return report
