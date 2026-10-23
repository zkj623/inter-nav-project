"""Isaac Sim floor scene with free rigid objects and contact-only grasps."""

from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.manipulation.floor_lift import (
    LIFT_ASSETS, LiftSuccessTracker, UrdfArmKinematics, lowest_box_point, policy_observation,
)


def create_object_asset(asset, destination):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Object').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(asset.mass)
    body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    body_api.CreateSolverPositionIterationCountAttr(32)
    body_api.CreateSolverVelocityIterationCountAttr(8)
    material = UsdShade.Material.Define(stage, '/Object/GripMaterial')
    physical_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical_material.CreateStaticFrictionAttr(1.0)
    physical_material.CreateDynamicFrictionAttr(0.9)
    physical_material.CreateRestitutionAttr(0.0)

    def cube(name, position, dimensions, color, yaw=0.0):
        shape = UsdGeom.Cube.Define(stage, f'/Object/{name}')
        shape.CreateSizeAttr(1.0)
        transform = UsdGeom.Xformable(shape)
        transform.AddTranslateOp().Set(Gf.Vec3d(*position))
        if yaw:
            transform.AddRotateZOp().Set(float(yaw))
        transform.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        collision.CreateContactOffsetAttr(0.002)
        collision.CreateRestOffsetAttr(0.0001)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose='physics')

    length, width, height = asset.dimensions
    if asset.name == 'box':
        cube('Body', (0.0, 0.0, height / 2), asset.dimensions, asset.color)
        cube('Lid', (0.0, 0.0, height + 0.006), (length, width, 0.012), (0.30, 0.23, 0.17))
    else:
        cube('Bottom', (0.0, 0.0, 0.009), (0.16, 0.16, 0.018), asset.color)
        radius = length / 2 - 0.007
        for index in range(20):
            angle = index * 2 * math.pi / 20
            cube(
                f'Wall{index}', (radius * math.cos(angle), radius * math.sin(angle), height / 2),
                (0.014, 2 * radius * math.tan(math.pi / 20) + 0.002, height),
                asset.color, math.degrees(angle),
            )
    handle_color = (0.12, 0.14, 0.17)
    post_x = 0.065
    post_height = asset.handle_height - height
    for name, sign in (('HandleLeft', -1), ('HandleRight', 1)):
        cube(name, (sign * post_x, 0.0, height + post_height / 2),
             (0.018, asset.handle_width, post_height), handle_color)
    cube('HandleBar', (0.0, 0.0, asset.handle_height),
         (2 * post_x + 0.018, asset.handle_width, 0.025), handle_color)
    stage.GetRootLayer().Save()
    return destination


class R1LiteFloorLift:
    control_dt = 0.02
    physics_dt = 0.005
    phase_names = ('approach', 'descend', 'close', 'lift', 'hold')

    def __init__(self, output, object_name='box', object_xy=(0.60, -0.335),
                 robot_usd=None, robot_urdf=None, lift_height=0.22, object_usd=None,
                 grasp_offset=None, object_dimensions=None, object_mass=None, bimanual_contact=False,
                 robot_position=None, robot_orientation=None, object_orientation=None):
        import carb
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema
        from omni.isaac.core import World
        from omni.isaac.core.prims import RigidPrim, RigidPrimView
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
        from grutopia_extension.configs.robots.r1lite import (
            DEFAULT_R1LITE_USD_PATH, R1LITE_ARM_HOME, R1LITE_TORSO_HOME,
        )

        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.asset = LIFT_ASSETS[object_name]
        self.dimensions = np.asarray(object_dimensions or self.asset.dimensions, dtype=float)
        self.bottom_dimensions = self.dimensions.copy()
        if not object_usd:
            self.bottom_dimensions[2] = max(self.dimensions[2], self.asset.handle_height + 0.0125)
        self.grasp_offset = np.asarray(grasp_offset or self.asset.grasp_offset, dtype=float)
        self.object_mass = self.asset.mass if object_mass is None else object_mass
        if self.dimensions.shape != (3,) or not np.isfinite(self.dimensions).all() or np.any(self.dimensions <= 0):
            raise ValueError('object dimensions must be three finite positive lengths')
        if not math.isfinite(self.object_mass) or self.object_mass <= 0:
            raise ValueError('object mass must be positive and finite')
        if object_usd and (grasp_offset is None or object_dimensions is None):
            raise ValueError('custom USD assets require --grasp-offset and --object-dimensions')
        if self.grasp_offset.shape != (3,) or not np.isfinite(self.grasp_offset).all():
            raise ValueError('grasp offset must be a finite xyz vector')
        if len(object_xy) != 2 or not np.isfinite(object_xy).all():
            raise ValueError('object xy must contain two finite coordinates')
        self.lift_height = lift_height
        self.kinematics = UrdfArmKinematics(
            robot_urdf or str(Path(DEFAULT_R1LITE_USD_PATH).with_suffix('.urdf'))
        )
        carb.settings.get_settings().set('/physics/cooking/ujitsoCollisionCooking', False)
        self.world = World(physics_dt=self.physics_dt, rendering_dt=self.control_dt,
                           stage_units_in_meters=1.0, backend='numpy', device='cpu')
        self.world.get_physics_context().set_solver_type('TGS')
        from grutopia_extension.manipulation.local_ground import add_local_ground

        add_local_ground(self.world.scene)
        self.stage = get_current_stage()
        if not object_usd:
            for name, asset in LIFT_ASSETS.items():
                create_object_asset(asset, self.output / 'assets' / f'{name}.usd')
        asset_path = Path(object_usd) if object_usd else self.output / 'assets' / f'{object_name}.usd'
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        self.object_asset_path = str(asset_path.resolve())
        add_reference_to_stage(str(asset_path.resolve()), '/World/LiftObject')
        object_prim = self.stage.GetPrimAtPath('/World/LiftObject')
        nested_bodies = [prim for prim in Usd.PrimRange(object_prim)
                         if prim != object_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if nested_bodies:
            raise ValueError('custom asset must be one rigid body at its default root, not nested rigid bodies')
        UsdPhysics.RigidBodyAPI.Apply(object_prim)
        UsdPhysics.MassAPI.Apply(object_prim).CreateMassAttr(self.object_mass)
        self.object = self.world.scene.add(RigidPrim(
            '/World/LiftObject', name='lift_object', position=np.array((*object_xy, 0.003)),
            orientation=object_orientation,
        ))
        add_reference_to_stage(robot_usd or DEFAULT_R1LITE_USD_PATH, '/World/Robot')
        root = self.stage.GetPrimAtPath('/World/Robot')
        home = {f'{side}_arm_joint{index}': value
                for side in ('left', 'right') for index, value in enumerate(R1LITE_ARM_HOME, 1)}
        home.update({f'torso_joint{index}': value for index, value in enumerate(R1LITE_TORSO_HOME, 1)})
        home.update({'right_gripper_finger_joint1': 0.045, 'right_gripper_finger_joint2': -0.045})
        effort_limits = {joint.get('name'): float(joint.find('limit').get('effort'))
                         for joint in ET.parse(self.kinematics_urdf(robot_urdf, DEFAULT_R1LITE_USD_PATH)).getroot().findall('joint')
                         if joint.find('limit') is not None}
        rigid_paths = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_paths[prim.GetName()] = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                articulation_api.CreateSolverPositionIterationCountAttr(64)
                articulation_api.CreateSolverVelocityIterationCountAttr(16)
                articulation_api.CreateEnabledSelfCollisionsAttr(True)
            if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
                continue
            name = prim.GetName()
            angular = prim.IsA(UsdPhysics.RevoluteJoint)
            kind = 'angular' if angular else 'linear'
            initial = home.get(name, 0.0)
            state = PhysxSchema.JointStateAPI.Apply(prim, kind)
            state.CreatePositionAttr(float(np.degrees(initial) if angular else initial))
            state.CreateVelocityAttr(0.0)
            drive = UsdPhysics.DriveAPI.Apply(prim, kind)
            drive.CreateMaxForceAttr(effort_limits.get(name, 40.0))
            drive.CreateStiffnessAttr(0.0 if name.startswith('wheel_') else 6000.0)
            drive.CreateDampingAttr(80.0 if name.startswith('wheel_') else 250.0)
            drive.CreateTargetPositionAttr(float(np.degrees(initial) if angular else initial))
            drive.CreateTargetVelocityAttr(0.0)
        grip_material = UsdShade.Material.Define(self.stage, '/World/GripperMaterial')
        material_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        material_api.CreateStaticFrictionAttr(1.0)
        material_api.CreateDynamicFrictionAttr(0.9)
        material_api.CreateRestitutionAttr(0.0)
        for prim in Usd.PrimRange(root):
            if 'gripper_finger' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(grip_material, materialPurpose='physics')
        self.robot = self.world.scene.add(Robot('/World/Robot', name='r1lite',
                                               position=np.array((0, 0, 0.025)) if robot_position is None else robot_position,
                                               orientation=robot_orientation))
        self.base = self.world.scene.add(RigidPrim(rigid_paths['base_link'], name='base_measurement'))
        self.tool = self.world.scene.add(RigidPrim(rigid_paths['right_gripper_link'], name='tool_measurement'))
        finger_sides = ('left', 'right') if bimanual_contact else ('right',)
        finger_paths = [rigid_paths[f'{side}_gripper_finger_link{index}']
                        for side in finger_sides for index in (1, 2)]
        self.finger_count = len(finger_paths)
        self.fingers = self.world.scene.add(RigidPrimView(
            finger_paths,
            name='finger_contacts', track_contact_forces=True,
            contact_filter_prim_paths_expr=[['/World/LiftObject'] for _ in finger_paths], max_contact_count=256,
        ))
        light = UsdLux.DomeLight.Define(self.stage, '/World/LiftDome')
        light.CreateIntensityAttr(700.0)
        sun = UsdLux.DistantLight.Define(self.stage, '/World/LiftSun')
        sun.CreateIntensityAttr(1800.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, 20))
        self.configure_task_collisions(root, rigid_paths)
        self.world.reset()
        self.control_names = self.kinematics.names + ['right_gripper_finger_joint1', 'right_gripper_finger_joint2']
        self.control_indices = np.array([self.robot.get_dof_index(name) for name in self.control_names])
        self.wheel_indices = np.array([self.robot.get_dof_index(f'wheel_motor_joint{index}') for index in (1, 2, 3)])
        self.position_indices = np.array([index for index in range(self.robot.num_dof) if index not in self.wheel_indices])
        self.home = np.array([home.get(name, 0.0) for name in self.robot.dof_names], dtype=np.float32)
        self.command = self.home.copy()
        self.robot.set_joints_default_state(positions=self.home, velocities=np.zeros(self.robot.num_dof))
        self.frames = 0
        self.max_tilt = 0.0
        self.contact_forces = np.zeros(self.finger_count)
        for _ in range(60):
            self.step(self.command[self.control_indices])
        self.initial_object_position = self.object.get_world_pose()[0].copy()
        self.initial_base_position = self.base.get_world_pose()[0].copy()
        self.tracker = LiftSuccessTracker(self.object_bottom(), required_lift=0.15, hold_seconds=2.0)
        self.stage.GetRootLayer().Export(str(self.output / 'scene.usda'))
        from grutopia_extension.interactive_navigation.visualization import FixedOverviewCamera

        self.camera = FixedOverviewCamera(resolution=(1280, 720), position=(2.25, -2.2, 1.5),
                                          look_at=(0.35, -0.15, 0.48), focal_length=22.0)
        self.close_camera = FixedOverviewCamera(resolution=(960, 720), position=(1.45, -1.15, 0.65),
                                                look_at=(0.55, -0.335, 0.35), focal_length=28.0)
        for _ in range(15):
            self.step(self.command[self.control_indices])
        print(json.dumps({'event': 'lift_scene_ready', 'joints': self.control_names,
                          'object_position': self.initial_object_position.tolist(),
                          'tool_position': self.tool_position().tolist(),
                          'joint_error': float(np.max(np.abs(self.robot.get_joint_positions() - self.home)))}), flush=True)

    @staticmethod
    def kinematics_urdf(robot_urdf, default_usd):
        return robot_urdf or str(Path(default_usd).with_suffix('.urdf'))

    def configure_task_collisions(self, root, rigid_paths):
        return None

    @staticmethod
    def rotation(quaternion):
        return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()

    def tool_position(self):
        position, quaternion = self.tool.get_world_pose()
        return position + self.rotation(quaternion) @ self.kinematics.tool_offset

    def object_bottom(self):
        position, quaternion = self.object.get_world_pose()
        rotation = self.rotation(quaternion)
        center = position + rotation @ np.array((0.0, 0.0, self.bottom_dimensions[2] / 2))
        return lowest_box_point(center, rotation, self.bottom_dimensions)

    def grasp_position(self):
        position, quaternion = self.object.get_world_pose()
        return position + self.rotation(quaternion) @ self.grasp_offset

    def solve(self, world_target, seed):
        position, quaternion = self.base.get_world_pose()
        rotation = self.rotation(quaternion)
        target_orientation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        return self.kinematics.solve(rotation.T @ (world_target - position), rotation.T @ target_orientation, seed)

    def step(self, controlled_positions):
        from omni.isaac.core.utils.types import ArticulationAction

        controlled_positions = np.asarray(controlled_positions, dtype=np.float32)
        if controlled_positions.shape != self.control_indices.shape or not np.isfinite(controlled_positions).all():
            raise ValueError('invalid joint action')
        lower = np.concatenate((self.kinematics.lower, (0.0, -0.05)))
        upper = np.concatenate((self.kinematics.upper, (0.05, 0.0)))
        self.command[self.control_indices] = np.clip(controlled_positions, lower, upper)
        self.robot.apply_action(ArticulationAction(
            joint_positions=self.command[self.position_indices], joint_indices=self.position_indices,
        ))
        self.robot.apply_action(ArticulationAction(joint_velocities=np.zeros(3), joint_indices=self.wheel_indices))
        forces = []
        for index in range(4):
            self.world.step(render=index == 3)
            matrix = np.asarray(self.fingers.get_contact_force_matrix(dt=self.physics_dt))
            forces.append(np.linalg.norm(matrix.reshape(self.finger_count, -1, 3).sum(axis=1), axis=1))
        self.contact_forces = np.mean(forces, axis=0)
        self.frames += 1
        base_position, quaternion = self.base.get_world_pose()
        tilt = math.degrees(math.acos(np.clip(self.rotation(quaternion)[2, 2], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        if not np.isfinite(base_position).all() or tilt > 12.0 or base_position[2] < -0.03:
            raise RuntimeError(f'Robot unstable: base={base_position.tolist()}, tilt={tilt:.2f} degrees')
        return self.robot.get_joint_positions()[self.control_indices].copy()

    def image(self, close=False):
        import cv2

        rgba = np.asarray((self.close_camera if close else self.camera).get_data()['rgba'])
        if rgba.size == 0:
            raise RuntimeError('camera did not produce an image')
        return cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

    def run(self, policy=None):
        import cv2

        observations = []
        actions = []
        trajectory = []
        start = self.robot.get_joint_positions()[self.control_indices]
        if policy is None:
            grasp = self.grasp_position().copy()
            approach = self.solve(grasp + np.array((0.0, 0.0, 0.18)), start[:9])
            descend = self.solve(grasp, approach)
            lifted = self.solve(grasp + np.array((0.0, 0.0, self.lift_height)), descend)
            targets = [np.r_[approach, 0.045, -0.045], np.r_[descend, 0.045, -0.045],
                       np.r_[descend, 0.0, 0.0], np.r_[lifted, 0.0, 0.0], np.r_[lifted, 0.0, 0.0]]
        else:
            targets = [start.copy() for _ in self.phase_names]
        writer = cv2.VideoWriter(str(self.output / 'lift.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError('cannot open video encoder')
        durations = (3.0, 3.0, 1.4, 3.5, 3.0)
        self.tracker = LiftSuccessTracker(self.object_bottom(), required_lift=0.15, hold_seconds=2.0)
        previous = start.copy()
        try:
            cv2.imwrite(str(self.output / 'start.jpg'), self.image())
            for phase, (name, target, duration) in enumerate(zip(self.phase_names, targets, durations)):
                steps = round(duration / self.control_dt)
                for step_index in range(steps):
                    progress = (step_index + 1) / steps
                    blend = progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)
                    observation = policy_observation(
                        self.robot.get_joint_positions()[self.control_indices],
                        self.object.get_world_pose()[0] if policy is None else self.initial_object_position,
                        phase, progress, self.dimensions,
                    )
                    expert_action = previous + blend * (target - previous)
                    action = expert_action if policy is None else policy(observation)
                    measured = self.step(action)
                    grasped = bool(np.all(self.contact_forces > 0.05))
                    bottom = self.object_bottom()
                    self.tracker.update(bottom, self.control_dt, grasped, self.max_tilt < 12.0)
                    observations.append(observation)
                    actions.append(action)
                    object_position = self.object.get_world_pose()[0]
                    trajectory.append({'phase': name, 'progress': progress, 'object_position': object_position.tolist(),
                                       'bottom': bottom, 'finger_forces': self.contact_forces.tolist(),
                                       'tool_position': self.tool_position().tolist(), 'grasped': grasped,
                                       'joint_error': float(np.max(np.abs(measured[:9] - action[:9])))})
                    frame = self.image()
                    cv2.putText(frame, f'R1 Lite | {self.asset.name} | {name} | lift {bottom - self.tracker.initial_bottom:.3f} m',
                                (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
                    writer.write(frame)
                previous = target.copy()
                cv2.imwrite(str(self.output / f'{phase}_{name}.jpg'), self.image())
                cv2.imwrite(str(self.output / f'{phase}_{name}_close.jpg'), self.image(close=True))
                print(json.dumps({'event': 'lift_phase_end', **trajectory[-1]}), flush=True)
        finally:
            writer.release()
            dataset_name = 'demonstration.npz' if policy is None else 'policy_rollout.npz'
            np.savez_compressed(self.output / dataset_name, observations=np.asarray(observations, dtype=np.float32),
                                actions=np.asarray(actions, dtype=np.float32))
            (self.output / 'trajectory.json').write_text(json.dumps(trajectory, indent=2))
        report = {
            'robot': 'Galaxea R1 Lite', 'object': self.asset.name,
            'controller': 'bounded_urdf_ik' if policy is None else 'behavior_cloning',
            'success': bool(self.tracker.success and np.all(self.contact_forces > 0.05)
                            and self.object_bottom() - self.tracker.initial_bottom >= self.tracker.required_lift),
            'maximum_lift_m': self.tracker.max_lift,
            'final_bottom_m': self.object_bottom(), 'hold_seconds': self.tracker.held_seconds,
            'required_lift_m': 0.15, 'required_hold_seconds': 2.0, 'max_base_tilt_degrees': self.max_tilt,
            'final_finger_forces_n': self.contact_forces.tolist(),
            'object_mass_kg': self.object_mass, 'object_asset': self.object_asset_path, 'self_collisions': True,
            'grasp_attachment': False, 'object_teleport_during_rollout': False,
            'phase_scheduler': 'scripted', 'frames': len(trajectory),
            'ik_used_for_policy_actions': False,
        }
        (self.output / 'result.json').write_text(json.dumps(report, indent=2))
        return report
