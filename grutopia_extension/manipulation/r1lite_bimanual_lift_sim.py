"""Physical bilateral side squeeze and lift without handles or attachments."""

from pathlib import Path
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.manipulation.bimanual_floor_lift import (
    BimanualBox, BimanualUrdfKinematics, create_unhandled_box, opposed_side_contact,
)
from grutopia_extension.manipulation.floor_lift import LiftSuccessTracker
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.r1lite_lift_sim import R1LiteFloorLift


class BimanualCamera:
    def __init__(self, stage, path, resolution, position, look_at, focal_length):
        from pxr import Gf, UsdGeom
        import omni.replicator.core as rep

        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(36.0 * resolution[1] / resolution[0])
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*look_at), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(matrix.GetInverse())
        self.render_product = rep.create.render_product(path, resolution)
        self.annotator = rep.AnnotatorRegistry.get_annotator('LdrColor')
        self.annotator.attach(self.render_product)

    def get_data(self):
        return {'rgba': self.annotator.get_data()}


class R1LiteBimanualLift(R1LiteFloorLift):
    phase_names = ('approach_sides', 'squeeze', 'lift', 'hold')

    def __init__(self, output, box=None, object_xy=(0.65, 0.0), robot_usd=None, robot_urdf=None,
                 arm_stiffness=1200.0, robot_position=None, robot_orientation=None, object_orientation=None):
        from omni.isaac.core.prims import RigidPrim
        from pxr import Usd, UsdPhysics
        from grutopia_extension.configs.robots.r1lite import DEFAULT_R1LITE_USD_PATH

        self.box = box or BimanualBox()
        urdf = robot_urdf or str(Path(DEFAULT_R1LITE_USD_PATH).with_suffix('.urdf'))
        self.collision_model = RobotCollisionModel(urdf)
        self.physics_audit = []
        self.minimum_noncontact_clearance = 0.005
        self.maximum_finger_penetration = 0.002
        self.grasp_x_offset = 0.0
        self.audit_enabled = False
        asset = create_unhandled_box(self.box, Path(output) / 'assets' / 'large_unhandled_box.usd')
        super().__init__(output, 'box', object_xy, robot_usd, robot_urdf,
                         object_usd=str(asset), grasp_offset=(0, 0, self.box.grasp_height),
                         object_dimensions=self.box.dimensions, object_mass=self.box.mass, bimanual_contact=True,
                         robot_position=robot_position, robot_orientation=robot_orientation,
                         object_orientation=object_orientation)
        self.bimanual = BimanualUrdfKinematics(urdf)
        left_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Robot'))
                      if prim.GetName() == 'left_gripper_link' and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(left_paths) != 1:
            raise RuntimeError('left gripper rigid body is missing')
        self.left_tool = self.world.scene.add(RigidPrim(left_paths[0], name='left_tool_measurement'))
        self.left_tool.initialize(self.world.physics_sim_view)
        self.control_names = self.bimanual.names + [f'{side}_gripper_finger_joint{index}'
                                                   for side in ('left', 'right') for index in (1, 2)]
        self.control_indices = np.array([self.robot.get_dof_index(name) for name in self.control_names])
        if not math.isfinite(arm_stiffness) or arm_stiffness <= 0:
            raise ValueError('arm stiffness must be positive and finite')
        self.arm_stiffness = arm_stiffness
        controller = self.robot.get_articulation_controller()
        stiffness, damping = controller.get_gains()
        stiffness = np.asarray(stiffness).copy()
        damping = np.asarray(damping).copy()
        for name in self.bimanual.names:
            if '_arm_joint' in name:
                index = self.robot.get_dof_index(name)
                stiffness[index] = arm_stiffness
                damping[index] = 100.0
        controller.set_gains(kps=stiffness, kds=damping, save_to_usd=True)
        self.finger_force_vectors = np.zeros((4, 3))
        current = self.robot.get_joint_positions()[self.control_indices]
        current[-4:] = 0
        for _ in range(75):
            self.step(current)
        self.initial_object_position = self.object.get_world_pose()[0].copy()
        self.camera = BimanualCamera(self.stage, '/World/BimanualOverview', (1280, 720),
                                     (2.2, -1.5, 1.3), (0.35, 0, 0.5), 28)
        self.close_camera = BimanualCamera(self.stage, '/World/BimanualFront', (960, 720),
                                           (1.9, -0.10, 0.67), (0.58, 0, 0.34), 30)
        self.side_camera = BimanualCamera(self.stage, '/World/BimanualSide', (960, 720),
                                          (0.55, -2.0, 0.72), (0.4, 0, 0.4), 30)
        for _ in range(20):
            self.step(current)
        self.stage.GetRootLayer().Export(str(self.output / 'scene.usda'))
        self.audit_enabled = True

    def configure_task_collisions(self, root, rigid_paths):
        from omni.isaac.core.prims import RigidPrimView
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

        original_paths = [prim.GetPath() for prim in Usd.PrimRange(root)
                          if prim.HasAPI(UsdPhysics.CollisionAPI)]
        guard_paths = []
        for name, vertices in self.collision_model.vertices.items():
            if name not in rigid_paths:
                raise RuntimeError(f'Cannot measure collision link {name}')
            if name.startswith(('wheel_', 'steer_')):
                continue
            shape = UsdGeom.Mesh.Define(self.stage, rigid_paths[name] + '/VisualObjectGuard')
            faces = self.collision_model.faces[name]
            shape.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
            shape.CreateFaceVertexCountsAttr([3] * len(faces))
            shape.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            shape.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
            collision = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
            collision.CreateContactOffsetAttr(0.002)
            collision.CreateRestOffsetAttr(0.0001)
            if 'gripper_finger' in name:
                material = UsdShade.Material.Get(self.stage, '/World/GripperMaterial')
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose='physics')
            guard_paths.append(shape.GetPath())
        guard_group = UsdPhysics.CollisionGroup.Define(self.stage, '/World/VisualObjectGuards')
        original_group = UsdPhysics.CollisionGroup.Define(self.stage, '/World/OriginalRobotColliders')
        guard_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(guard_paths)
        original_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(original_paths)
        guard_group.CreateFilteredGroupsRel().SetTargets([guard_group.GetPath(), original_group.GetPath()])
        original_group.CreateFilteredGroupsRel().SetTargets([guard_group.GetPath()])
        paths = [rigid_paths[name] for name in self.collision_model.names]
        self.collision_links = self.world.scene.add(RigidPrimView(paths, name='collision_measurements'))
        self.collision_link_names = [Sdf.Path(path).name for path in self.collision_links.prim_paths]
        (self.output / 'proxy_coverage.json').write_text(json.dumps(
            self.collision_model.proxy_coverage_report(), indent=2
        ))

    def collision_sample(self):
        positions, quaternions = self.collision_links.get_world_poses()
        transforms = {}
        for name, position, quaternion in zip(self.collision_link_names, positions, quaternions):
            transform = np.eye(4)
            transform[:3, :3] = self.rotation(quaternion)
            transform[:3, 3] = position
            transforms[name] = transform
        object_position, object_quaternion = self.object.get_world_pose()
        box_transform = np.eye(4)
        box_transform[:3, :3] = self.rotation(object_quaternion)
        box_transform[:3, 3] = object_position + box_transform[:3, :3] @ np.array((0, 0, self.box.dimensions[2] / 2))
        distances = self.collision_model.box_clearances(transforms, box_transform, self.box.dimensions,
                                                       include_fingers=True)
        noncontact = {name: distances[name] for name in self.collision_model.noncontact_names}
        nearest = min(noncontact, key=noncontact.get)
        finger_depth = max(0.0, -min(value for name, value in distances.items() if 'gripper_finger' in name))
        self_hits = self.collision_model.self_intersections(transforms)
        sample = {'sample': len(self.physics_audit), 'minimum_noncontact_clearance_m': noncontact[nearest],
                  'nearest_noncontact_link': nearest, 'maximum_finger_penetration_m': finger_depth,
                  'self_intersections': self_hits, 'link_positions': positions.tolist(),
                  'link_quaternions_wxyz': quaternions.tolist(), 'object_position': object_position.tolist(),
                  'object_quaternion_wxyz': object_quaternion.tolist()}
        self.physics_audit.append(sample)
        if noncontact[nearest] < self.minimum_noncontact_clearance or finger_depth > self.maximum_finger_penetration or self_hits:
            raise RuntimeError(f'Collision audit rejected physics sample: {sample["sample"]}, '
                               f'clearance={noncontact[nearest]:.6f} m at {nearest}, '
                               f'finger penetration={finger_depth:.6f} m, self={self_hits}')
        return sample

    def step(self, controlled_positions):
        if not hasattr(self, 'bimanual'):
            return super().step(controlled_positions)
        from omni.isaac.core.utils.types import ArticulationAction

        controlled_positions = np.asarray(controlled_positions, dtype=np.float32)
        if controlled_positions.shape != (19,) or not np.isfinite(controlled_positions).all():
            raise ValueError('bimanual action must contain 19 finite joint targets')
        lower = np.r_[self.bimanual.lower, 0.0, -0.05, 0.0, -0.05]
        upper = np.r_[self.bimanual.upper, 0.05, 0.0, 0.05, 0.0]
        self.command[self.control_indices] = np.clip(controlled_positions, lower, upper)
        wheel_velocities = self.prepare_base_motion()
        self.robot.apply_action(ArticulationAction(
            joint_positions=self.command[self.position_indices], joint_indices=self.position_indices,
        ))
        self.robot.apply_action(ArticulationAction(joint_velocities=wheel_velocities, joint_indices=self.wheel_indices))
        vectors = []
        for index in range(4):
            self.world.step(render=index == 3)
            if self.audit_enabled:
                self.collision_sample()
            matrix = np.asarray(self.fingers.get_contact_force_matrix(dt=self.physics_dt))
            vectors.append(matrix.reshape(4, -1, 3).sum(axis=1))
        self.finger_force_vectors = np.mean(vectors, axis=0)
        self.contact_forces = np.linalg.norm(self.finger_force_vectors, axis=1)
        self.frames += 1
        base_position, quaternion = self.base.get_world_pose()
        tilt = math.degrees(math.acos(np.clip(self.rotation(quaternion)[2, 2], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        if not np.isfinite(base_position).all() or tilt > 12.0 or base_position[2] < -0.03:
            raise RuntimeError(f'Robot unstable during bimanual grasp: tilt={tilt:.2f} deg')
        return self.robot.get_joint_positions()[self.control_indices].copy()

    def prepare_base_motion(self):
        return np.zeros(3)

    def solve_both(self, center, half_width, height, seed):
        base_position, quaternion = self.base.get_world_pose()
        rotation = self.rotation(quaternion)
        left_target = np.asarray(center) + np.array((0, half_width, height))
        right_target = np.asarray(center) + np.array((0, -half_width, height))
        left_target[0] += self.grasp_x_offset
        right_target[0] += self.grasp_x_offset
        roll = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        left_orientation = Rotation.from_euler('z', -90, degrees=True).as_matrix() @ roll
        right_orientation = Rotation.from_euler('z', 90, degrees=True).as_matrix() @ roll
        box_transform = np.eye(4)
        box_transform[:3, :3] = rotation.T
        box_transform[:3, 3] = rotation.T @ (np.asarray(center) + np.array((
            0, 0, self.box.dimensions[2] / 2 + height - self.box.grasp_height
        )) - base_position)
        return self.bimanual.solve(rotation.T @ (left_target - base_position),
                                   rotation.T @ (right_target - base_position),
                                   rotation.T @ left_orientation, rotation.T @ right_orientation, seed,
                                   self.collision_model, box_transform, self.box.dimensions, clearance=0.05)

    def tool_positions(self):
        positions = []
        for rigid, kinematics in ((self.left_tool, self.bimanual.left), (self.tool, self.bimanual.right)):
            position, quaternion = rigid.get_world_pose()
            positions.append((position + self.rotation(quaternion) @ kinematics.tool_offset).tolist())
        return positions

    def validate_plan(self, start, targets, center):
        position, quaternion = self.base.get_world_pose()
        base = np.eye(4)
        base[:3, :3] = self.rotation(quaternion)
        base[:3, 3] = position
        previous = start[:15]
        samples = []
        for phase, target in enumerate(targets):
            for progress in np.linspace(0.0, 1.0, 101):
                joints = previous + progress * (target[:15] - previous)
                transforms = self.collision_model.forward(dict(zip(self.bimanual.names, joints)), base)
                box_transform = np.eye(4)
                box_transform[:3, 3] = center + np.array((0, 0, self.box.dimensions[2] / 2))
                if phase >= 2:
                    box_transform[2, 3] += 0.22 * (progress if phase == 2 else 1.0)
                distances = self.collision_model.box_clearances(transforms, box_transform, self.box.dimensions)
                self_hits = self.collision_model.self_intersections(transforms)
                nearest = min(distances, key=distances.get)
                entry = {'phase': self.phase_names[phase], 'progress': float(progress),
                         'minimum_noncontact_clearance_m': distances[nearest], 'nearest_link': nearest,
                         'self_intersections': self_hits}
                samples.append(entry)
                if distances[nearest] < 0.015 or self_hits:
                    (self.output / 'planning_audit.json').write_text(json.dumps(samples, indent=2))
                    raise RuntimeError(f'Unsafe planned interpolation: {entry}')
            previous = target[:15]
        (self.output / 'planning_audit.json').write_text(json.dumps(samples, indent=2))
        return samples

    def save_collision_audit(self):
        (self.output / 'physics_collision_samples.json').write_text(json.dumps({
            'link_names': self.collision_link_names, 'samples': self.physics_audit
        }))
        report = {'method': 'FCL conservative convex envelopes of visual meshes and original URDF collision geometry',
                  'sampling_hz': 1.0 / self.physics_dt, 'samples': len(self.physics_audit),
                  'noncontact_clearance_threshold_m': self.minimum_noncontact_clearance,
                  'finger_penetration_tolerance_m': self.maximum_finger_penetration,
                  'minimum_noncontact_clearance_m': min((sample['minimum_noncontact_clearance_m']
                                                        for sample in self.physics_audit), default=None),
                  'maximum_finger_penetration_m': max((sample['maximum_finger_penetration_m']
                                                       for sample in self.physics_audit), default=None),
                  'self_intersection_sample_count': sum(bool(sample['self_intersections'])
                                                        for sample in self.physics_audit),
                  'self_method': 'nonadjacent visual triangle meshes; parent-child and siblings excluded',
                  'collision_guard_groups': 'visual guards contact object and ground; original self-colliders remain enabled',
                  'continuous_time_guarantee': False}
        report['passed'] = bool(self.physics_audit) and all(
            sample['minimum_noncontact_clearance_m'] >= self.minimum_noncontact_clearance
            and sample['maximum_finger_penetration_m'] <= self.maximum_finger_penetration
            and not sample['self_intersections'] for sample in self.physics_audit
        )
        (self.output / 'collision_audit.json').write_text(json.dumps(report, indent=2))
        return report

    def run(self):
        import cv2

        center = self.object.get_world_pose()[0].copy()
        start = self.robot.get_joint_positions()[self.control_indices]
        outside = self.box.dimensions[1] / 2 + 0.13
        pressed = self.box.dimensions[1] / 2 - self.box.squeeze
        approach = self.solve_both(center, outside, self.box.grasp_height, start[:15])
        squeeze = self.solve_both(center, pressed, self.box.grasp_height, approach)
        lifted = self.solve_both(center, pressed, self.box.grasp_height + 0.22, squeeze)
        targets = [np.r_[approach, np.zeros(4)], np.r_[squeeze, np.zeros(4)],
                   np.r_[lifted, np.zeros(4)], np.r_[lifted, np.zeros(4)]]
        durations = (5.0, 4.0, 4.0, 3.0)
        self.validate_plan(start, targets, center)
        tracker = LiftSuccessTracker(self.object_bottom(), required_lift=0.15, hold_seconds=2.0)
        writer = cv2.VideoWriter(str(self.output / 'bimanual_lift.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 50, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError('could not create bimanual video')
        front_writer = cv2.VideoWriter(str(self.output / 'bimanual_front.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                       50, (960, 720))
        if not front_writer.isOpened():
            writer.release()
            raise RuntimeError('could not create front-view video')
        side_writer = cv2.VideoWriter(str(self.output / 'bimanual_side.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                      50, (960, 720))
        if not side_writer.isOpened():
            writer.release()
            front_writer.release()
            raise RuntimeError('could not create side-view video')
        trajectory = []
        observations = []
        actions = []
        previous = start.copy()
        side_contact = False
        try:
            cv2.imwrite(str(self.output / 'start.jpg'), self.image())
            self.collision_sample()
            for phase, (name, target, duration) in enumerate(zip(self.phase_names, targets, durations)):
                steps = round(duration / self.control_dt)
                for step_index in range(steps):
                    progress = (step_index + 1) / steps
                    blend = progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)
                    measured = self.robot.get_joint_positions()[self.control_indices]
                    phase_encoding = np.zeros(4)
                    phase_encoding[phase] = 1
                    observation = np.r_[measured, self.object.get_world_pose()[0],
                                        self.box.dimensions, phase_encoding, progress]
                    action = previous + blend * (target - previous)
                    self.step(action)
                    hand_vectors = self.finger_force_vectors.reshape(2, 2, 3).sum(axis=1)
                    side_contact = opposed_side_contact(hand_vectors)
                    bottom = self.object_bottom()
                    tracker.update(bottom, self.control_dt, side_contact, self.max_tilt < 12.0)
                    entry = {'phase': name, 'progress': progress,
                             'object_position': self.object.get_world_pose()[0].tolist(), 'bottom': bottom,
                             'opposed_side_contact': side_contact, 'hand_force_vectors_n': hand_vectors.tolist(),
                             'tool_positions': self.tool_positions(),
                             'joint_error': float(np.max(np.abs(self.robot.get_joint_positions()[self.control_indices[:15]] - action[:15]))),
                             'noncontact_clearance_m': self.physics_audit[-1]['minimum_noncontact_clearance_m']}
                    trajectory.append(entry)
                    observations.append(observation)
                    actions.append(action)
                    frame = self.image()
                    cv2.putText(frame, f'BIMANUAL | no handles | {name} | lift {bottom-tracker.initial_bottom:.3f} m',
                                (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 250, 250), 2)
                    writer.write(frame)
                    front_frame = self.image(close=True)
                    cv2.putText(front_frame, f'TWO HANDS | {name} | lift {bottom-tracker.initial_bottom:.3f} m',
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 250, 250), 2)
                    front_writer.write(front_frame)
                    side_frame = cv2.cvtColor(np.asarray(self.side_camera.get_data()['rgba'])[:, :, :3], cv2.COLOR_RGB2BGR)
                    cv2.putText(side_frame, f'{name} | body gap {entry["noncontact_clearance_m"] * 1000:.1f} mm',
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 250, 250), 2)
                    side_writer.write(side_frame)
                previous = target.copy()
                cv2.imwrite(str(self.output / f'{phase}_{name}.jpg'), self.image())
                cv2.imwrite(str(self.output / f'{phase}_{name}_close.jpg'), self.image(close=True))
                cv2.imwrite(str(self.output / f'{phase}_{name}_side.jpg'), side_frame)
                print(json.dumps({'event': 'bimanual_phase_end', **trajectory[-1]}), flush=True)
        finally:
            writer.release()
            front_writer.release()
            side_writer.release()
            collision_report = self.save_collision_audit()
            (self.output / 'trajectory.json').write_text(json.dumps(trajectory, indent=2))
            np.savez_compressed(self.output / 'demonstration.npz', observations=np.asarray(observations, dtype=np.float32),
                                actions=np.asarray(actions, dtype=np.float32))
        result = {'robot': 'Galaxea R1 Lite', 'object': 'procedural_large_unhandled_cardboard_box',
                  'dimensions_m': list(self.box.dimensions), 'mass_kg': self.box.mass,
                  'controller': 'coupled_bimanual_urdf_ik', 'training_performed': False,
                  'success': bool(collision_report['passed'] and tracker.success and side_contact
                                  and self.object_bottom() - tracker.initial_bottom >= tracker.required_lift),
                  'maximum_lift_m': tracker.max_lift, 'hold_seconds': tracker.held_seconds,
                  'final_bottom_m': self.object_bottom(), 'opposed_side_contact': side_contact,
                  'final_hand_force_vectors_n': hand_vectors.tolist(), 'max_base_tilt_degrees': self.max_tilt,
                  'handles': False, 'grasp_attachment': False, 'object_teleport_during_rollout': False,
                  'self_collisions': True, 'frames': len(trajectory)}
        result['phase_scheduler'] = 'scripted'
        result['drive_effort_limits'] = 'imported_URDF'
        result['arm_stiffness'] = self.arm_stiffness
        result['squeeze_target_m'] = self.box.squeeze
        result['collision_audit_passed'] = collision_report['passed']
        result['minimum_noncontact_clearance_m'] = collision_report['minimum_noncontact_clearance_m']
        result['maximum_finger_penetration_m'] = collision_report['maximum_finger_penetration_m']
        result['grasp_x_offset_m'] = self.grasp_x_offset
        result['collision_geometry'] = 'original self-colliders plus conservative visual object-contact guards'
        result['continuous_time_guarantee'] = False
        result['asset_source'] = 'procedural locally authored USD; no downloaded object dataset'
        result['object_rigid_body'] = True
        (self.output / 'result.json').write_text(json.dumps(result, indent=2))
        return result
