import numpy as np
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots.robot import Robot as IsaacRobot
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from grutopia.core.config.robot import RobotCfg
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.util import log
from grutopia_extension.configs.robots.go2 import GO2_JOINT_NAMES
from grutopia_extension.robots.go2_asset import (
    GO2_DEFAULT_JOINT_POSITIONS,
    ensure_go2_usd,
)


def apply_isaaclab_go2_physics(robot_prim_path: str) -> None:
    """Match UNITREE_GO2_CFG rigid-body damping, depenetration, and foot friction."""

    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = get_current_stage()
    robot = stage.GetPrimAtPath(robot_prim_path)
    if robot is None or not robot.IsValid():
        raise RuntimeError(f'Go2 prim is missing at {robot_prim_path}')

    material_path = f'{robot_prim_path}/Looks/isaaclab_go2_physics'
    material_prim = stage.DefinePrim(material_path, 'Material')
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr(1.0)
    material_api.CreateDynamicFrictionAttr(1.0)
    material_api.CreateRestitutionAttr(0.0)
    material = UsdShade.Material(material_prim)

    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api.CreateLinearDampingAttr(0.0)
            rigid_api.CreateAngularDampingAttr(0.0)
            rigid_api.CreateMaxDepenetrationVelocityAttr(1.0)
            rigid_api.CreateRetainAccelerationsAttr(False)
        if prim.IsA(UsdGeom.Gprim) and (
            prim.GetName().endswith('_foot') or prim.GetName() == 'foot'
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _resolve_go2_base_prim(robot_prim_path: str) -> str:
    """Find the Go2 torso prim under the spawned robot root."""

    candidates = (
        f'{robot_prim_path}/base',
        f'{robot_prim_path}/Go2/base',
        f'{robot_prim_path}/go2/base',
        f'{robot_prim_path}/UnitreeGo2/base',
    )
    for candidate in candidates:
        prim = get_prim_at_path(candidate)
        if prim is not None and prim.IsValid():
            return candidate
    raise RuntimeError(
        'Go2 base prim was not found under '
        f'{robot_prim_path}; tried {candidates}'
    )


class Go2(IsaacRobot):
    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        name: str,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        scale: np.ndarray = None,
        generate_fallback_asset: bool = True,
    ):
        resolved_usd_path = ensure_go2_usd(
            usd_path,
            generate_fallback=generate_fallback_asset,
        )
        add_reference_to_stage(prim_path=prim_path, usd_path=resolved_usd_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            scale=scale,
        )

    def configure_locomotion(self):
        from grutopia_extension.controllers.isaac_go2_ctrl import (
            DAMPING,
            SATURATION_EFFORT,
            STIFFNESS,
        )

        joint_subset = ArticulationSubset(self, GO2_JOINT_NAMES)
        joint_indices = joint_subset.joint_indices
        backend = self._articulation_view._backend_utils
        articulation_controller = self.get_articulation_controller()
        # Implicit PhysX position drives with the Isaac Lab Go2 gains: the
        # drive recomputes PD torque inside every 200 Hz physics substep, so
        # the loop stays closed during batched render advances (python-side
        # effort control froze the torque for a full rendering_dt there).
        articulation_controller.switch_control_mode('position')
        self._articulation_view.set_gains(
            kps=backend.expand_dims(np.full(12, STIFFNESS), 0),
            kds=backend.expand_dims(np.full(12, DAMPING), 0),
            save_to_usd=False,
            joint_indices=joint_indices,
        )
        self._articulation_view.set_max_efforts(
            values=backend.expand_dims(np.full(12, SATURATION_EFFORT), 0),
            joint_indices=joint_indices,
        )
        self._articulation_view.set_max_joint_velocities(
            values=backend.expand_dims(np.full(12, 30.0), 0),
            joint_indices=joint_indices,
        )
        self._articulation_view.set_solver_position_iteration_counts(
            backend.expand_dims(4, 0)
        )
        self._articulation_view.set_solver_velocity_iteration_counts(
            backend.expand_dims(0, 0)
        )
        self._articulation_view.set_enabled_self_collisions(
            backend.expand_dims(False, 0)
        )
        self._articulation_view.set_friction_coefficients(
            values=backend.expand_dims(np.zeros(12), 0),
            joint_indices=joint_indices,
        )
        apply_isaaclab_go2_physics(self.prim_path)


@BaseRobot.register('Go2Robot')
class Go2Robot(BaseRobot):
    def __init__(self, config: RobotCfg, scene: Scene):
        super().__init__(config, scene)
        self._start_position = (
            np.asarray(config.position, dtype=float)
            if config.position is not None
            else None
        )
        self._start_orientation = (
            np.asarray(config.orientation, dtype=float)
            if config.orientation is not None
            else None
        )
        log.debug(f'Go2 {config.name}: position: {self._start_position}')
        log.debug(f'Go2 {config.name}: usd_path: {config.usd_path}')

        self.isaac_robot = Go2(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=config.usd_path,
            generate_fallback_asset=config.generate_fallback_asset,
        )
        self._robot_scale = np.ones(3, dtype=float)
        if config.scale is not None:
            self._robot_scale = np.asarray(config.scale, dtype=float)
            self.isaac_robot.set_local_scale(self._robot_scale)

        self._robot_base = RigidPrim(
            prim_path=_resolve_go2_base_prim(config.prim_path),
            name=f'{config.name}_base',
        )
        self._rigid_bodies = [self._robot_base]

    def post_reset(self):
        super().post_reset()
        default_positions = np.asarray(
            GO2_DEFAULT_JOINT_POSITIONS,
            dtype=np.float32,
        )
        joint_subset = ArticulationSubset(
            self.isaac_robot,
            GO2_JOINT_NAMES,
        )
        joint_indices = np.asarray(joint_subset.joint_indices, dtype=np.int64)
        all_default_positions = np.zeros(
            self.isaac_robot.num_dof,
            dtype=np.float32,
        )
        all_default_positions[joint_indices] = default_positions
        self.isaac_robot.set_joints_default_state(
            positions=all_default_positions,
            velocities=np.zeros(self.isaac_robot.num_dof, dtype=np.float32),
        )
        self.isaac_robot.set_joint_positions(
            default_positions,
            joint_indices=joint_indices,
        )
        self.isaac_robot.set_joint_velocities(
            np.zeros(12, dtype=np.float32),
            joint_indices=joint_indices,
        )
        self.isaac_robot.configure_locomotion()
        control = getattr(self, '_go2_locomotion_control', None)
        if control is not None:
            control.reset()

    def get_rigid_bodies(self):
        return self._rigid_bodies

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_base(self) -> RigidPrim:
        return self._robot_base

    def get_world_pose(self):
        return self._robot_base.get_world_pose()

    def apply_action(self, action: dict):
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warning(f'unknown controller {controller_name} in action')
                continue
            control = self.controllers[controller_name].action_to_control(
                controller_action
            )
            self.isaac_robot.apply_action(control)

    def get_obs(self):
        position, orientation = self._robot_base.get_world_pose()
        obs = {
            'position': position,
            'orientation': orientation,
            'head_position': position,
            'joint_positions': self.isaac_robot.get_joint_positions(),
            'joint_velocities': self.isaac_robot.get_joint_velocities(),
            'controllers': {},
            'sensors': {},
        }
        for controller_name, controller in self.controllers.items():
            obs['controllers'][controller_name] = controller.get_obs()
        for sensor_name, sensor in self.sensors.items():
            obs['sensors'][sensor_name] = sensor.get_data()
        return obs
