"""Isaac Sim bridge for the pure interaction navigation state machine."""

from math import atan2, pi
from typing import Mapping

import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.xforms import get_world_pose
from omni.physx import get_physx_simulation_interface
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics

from grutopia.core.util.joint import create_joint
from grutopia_extension.interactive_navigation.profiles import SceneBindings
from grutopia_extension.interactive_navigation.state_machine import InteractionObservation


class IsaacInteractionAdapter:
    """Read task prims and perform explicitly requested simulator side effects."""

    def __init__(self, bindings: SceneBindings, disable_carried_object_collision: bool = True):
        self.bindings = bindings
        self.disable_carried_object_collision = disable_carried_object_collision
        self._validate_prim_paths()
        self._initial_door_yaw = _yaw(self._pose(bindings.door_prim_path)[1])
        self._attachment_object_position = None
        self._attachment_hand_offset = None
        self._attachment_creation_jump = None
        self._door_contact_subscription = None
        self._door_arm_contact_count = 0
        self._door_arm_contact_max_impulse = 0.0
        self._door_arm_contact_pair = None
        self.object_hand_contact_distance = 0.12
        self._object_hand_contact_count = 0
        self._object_hand_contact_seen = 0
        self._object_hand_contact_max_impulse = 0.0
        self._object_hand_contact_pair = None

    def observe(self, step: int, robot_observation: Mapping) -> InteractionObservation:
        robot_position = _vector3(robot_observation['position'])
        hand_position = self._hand_position(robot_observation)
        obstacle_position = _vector3(self._pose(self.bindings.obstacle_prim_path)[0])
        object_position = _vector3(self._pose(self.bindings.object_prim_path)[0])
        door_yaw = _yaw(self._pose(self.bindings.door_prim_path)[1])
        door_open_angle = _normalized_angle(door_yaw - self._initial_door_yaw)
        controller_observation = robot_observation.get('controllers', {}).get(
            'right_arm_ik_controller',
            {},
        )
        return InteractionObservation(
            step=step,
            robot_position=robot_position,
            hand_position=hand_position,
            obstacle_position=obstacle_position,
            object_position=object_position,
            door_open_angle=door_open_angle,
            object_attached=self.is_object_attached(),
            robot_yaw=_yaw(robot_observation['orientation']),
            ik_success=_optional_bool(controller_observation, 'success'),
            ik_finished=_optional_bool(controller_observation, 'finished'),
            ik_position_error=_optional_float(controller_observation, 'position_error'),
            door_arm_contact=self._door_arm_contact_count > 0,
            object_hand_contact=self._object_hand_is_in_contact(hand_position, object_position),
        )

    def attach_object(self):
        """Weld the object to the wrist at the current contact pose."""

        if self.is_object_attached():
            return
        self._validate_prim_paths()
        object_pose = self._pose(self.bindings.object_prim_path)
        hand_pose = self._pose(self.bindings.hand_prim_path)
        object_position = np.asarray(object_pose[0], dtype=float)
        hand_position = np.asarray(hand_pose[0], dtype=float)
        object_prim = get_prim_at_path(self.bindings.object_prim_path)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(object_prim)
        rigid_body_api.CreateKinematicEnabledAttr(False)
        mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        mass_api.CreateMassAttr(0.05)
        if self.disable_carried_object_collision:
            self.set_object_collision(False)
        self.set_wrist_contact_collision(False)

        create_joint(
            prim_path=self.bindings.attachment_joint_path,
            joint_type='FixedJoint',
            body0=self.bindings.object_prim_path,
            body1=self.bindings.hand_prim_path,
            enabled=True,
        )
        if not self.is_object_attached():
            raise RuntimeError(f'fixed joint was not enabled: {self.bindings.attachment_joint_path}')
        attached_position = np.asarray(self._pose(self.bindings.object_prim_path)[0], dtype=float)
        self._attachment_object_position = object_position
        self._attachment_hand_offset = _relative_position(
            object_position,
            hand_position,
            hand_pose[1],
        )
        self._attachment_creation_jump = float(np.linalg.norm(attached_position - object_position))

    def set_object_kinematic(self, enabled: bool):
        object_prim = get_prim_at_path(self.bindings.object_prim_path)
        UsdPhysics.RigidBodyAPI.Apply(object_prim).CreateKinematicEnabledAttr(bool(enabled))

    def lock_robot(self):
        """Fix the torso to the world while the arm performs contact-rich IK."""

        joint_path = self.bindings.robot_lock_joint_path
        if is_prim_path_valid(joint_path):
            joint_prim = get_prim_at_path(joint_path)
            joint_prim.GetStage().RemovePrim(joint_prim.GetPath())
        create_joint(
            prim_path=joint_path,
            joint_type='FixedJoint',
            body0=self.bindings.torso_prim_path,
            enabled=True,
        )

    def unlock_robot(self):
        """Release the temporary torso-to-world joint, if it exists."""

        joint_path = self.bindings.robot_lock_joint_path
        if is_prim_path_valid(joint_path):
            joint_prim = get_prim_at_path(joint_path)
            joint_prim.GetStage().RemovePrim(joint_prim.GetPath())

    def set_object_collision(self, enabled: bool):
        """Enable contact for grasping or disable it for stable carrying."""

        object_prim = get_prim_at_path(self.bindings.object_prim_path)
        for prim in Usd.PrimRange(object_prim):
            if enabled and prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).GetPointsAttr().Get():
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                    'convexHull'
                )
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(bool(enabled))

    def is_object_attached(self) -> bool:
        joint_path = self.bindings.attachment_joint_path
        if not is_prim_path_valid(joint_path):
            return False
        enabled = get_prim_at_path(joint_path).GetAttribute('physics:jointEnabled').Get()
        return enabled is not False

    def attachment_metrics(self) -> dict:
        object_pose = self._pose(self.bindings.object_prim_path)
        hand_pose = self._pose(self.bindings.hand_prim_path)
        object_position = np.asarray(object_pose[0], dtype=float)
        hand_position = np.asarray(hand_pose[0], dtype=float)
        relative_offset = _relative_position(
            object_position,
            hand_position,
            hand_pose[1],
        )
        drift = None
        if self._attachment_hand_offset is not None:
            drift = float(np.linalg.norm(relative_offset - self._attachment_hand_offset))
        drop = None
        if self._attachment_hand_offset is not None:
            drop = float(self._attachment_hand_offset[2] - relative_offset[2])
        return {
            'object_attached': self.is_object_attached(),
            'object_position': object_position.tolist(),
            'hand_position': hand_position.tolist(),
            'relative_offset': relative_offset.tolist(),
            'relative_offset_drift': drift,
            'vertical_drop': drop,
            'creation_jump': self._attachment_creation_jump,
        }

    @staticmethod
    def world_aabb(prim_path: str) -> tuple:
        if not is_prim_path_valid(prim_path):
            raise RuntimeError(f'cannot compute AABB for missing prim: {prim_path}')
        prim = get_prim_at_path(prim_path)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return _vector3(bounds.GetMin()), _vector3(bounds.GetMax())

    def enable_door_arm_contact_monitor(self):
        if self._door_contact_subscription is not None:
            return
        for path in self._arm_contact_paths():
            if is_prim_path_valid(path):
                api = PhysxSchema.PhysxContactReportAPI.Apply(get_prim_at_path(path))
                api.CreateThresholdAttr().Set(0.0)
        PhysxSchema.PhysxContactReportAPI.Apply(
            get_prim_at_path(self.bindings.door_prim_path)
        ).CreateThresholdAttr().Set(0.0)
        self._door_contact_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report,
            )
        )

    def set_arm_collision(self, enabled: bool):
        for path in self._arm_contact_paths():
            if not is_prim_path_valid(path):
                continue
            for prim in Usd.PrimRange(get_prim_at_path(path)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(
                        bool(enabled)
                    )

    def set_wrist_contact_collision(self, enabled: bool):
        """Enable the G1 wrist contact sphere used for bowl grasping."""

        collider_path = self._wrist_contact_collider_path()
        if not is_prim_path_valid(collider_path):
            return
        UsdPhysics.CollisionAPI.Apply(
            get_prim_at_path(collider_path)
        ).CreateCollisionEnabledAttr(bool(enabled))

    def prepare_object_for_hand_contact(self):
        """Keep the bowl on the table, but allow the wrist collider to touch it."""

        self.set_object_kinematic(True)
        self.set_object_collision(True)
        self.set_wrist_contact_collision(True)
        self.enable_object_hand_contact_monitor()

    def enable_object_hand_contact_monitor(self):
        for path in (self._wrist_contact_collider_path(), *self._arm_contact_paths()):
            if not is_prim_path_valid(path):
                continue
            PhysxSchema.PhysxContactReportAPI.Apply(
                get_prim_at_path(path)
            ).CreateThresholdAttr().Set(0.0)
        if is_prim_path_valid(self.bindings.object_prim_path):
            PhysxSchema.PhysxContactReportAPI.Apply(
                get_prim_at_path(self.bindings.object_prim_path)
            ).CreateThresholdAttr().Set(0.0)
        if self._door_contact_subscription is None:
            self._door_contact_subscription = (
                get_physx_simulation_interface().subscribe_contact_report_events(
                    self._on_contact_report,
                )
            )

    def object_hand_contact_metrics(self) -> dict:
        return {
            'count': self._object_hand_contact_count,
            'max_impulse': self._object_hand_contact_max_impulse,
            'last_pair': self._object_hand_contact_pair,
            'physical_contact_confirmed': self._object_hand_contact_count > 0,
        }

    def prepare_door_for_contact(self, mass: float = 0.05):
        """Keep the authored revolute door dynamic and give contact a stable mass."""

        door_prim = get_prim_at_path(self.bindings.door_prim_path)
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(door_prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(door_prim).CreateMassAttr(float(mass))
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(door_prim)
        physx_body.CreateAngularDampingAttr().Set(0.02)
        physx_body.CreateEnableCCDAttr().Set(True)
        physx_body.CreateSolverPositionIterationCountAttr().Set(16)
        physx_body.CreateSolverVelocityIterationCountAttr().Set(4)
        static_group = door_prim.GetParent().GetChild('Group_Static')
        if static_group:
            UsdPhysics.FilteredPairsAPI.Apply(
                door_prim
            ).CreateFilteredPairsRel().AddTarget(static_group.GetPath())
        for prim in Usd.PrimRange(door_prim):
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).GetPointsAttr().Get():
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                    'convexHull'
                )

    def door_arm_contact_metrics(self) -> dict:
        return {
            'count': self._door_arm_contact_count,
            'max_impulse': self._door_arm_contact_max_impulse,
            'last_pair': self._door_arm_contact_pair,
            'physical_contact_confirmed': self._door_arm_contact_count > 0,
        }

    def close(self):
        self._door_contact_subscription = None

    def _arm_contact_paths(self):
        root = self.bindings.robot_prim_path
        return tuple(
            f'{root}/{link}'
            for link in (
                'right_elbow_link',
                'right_wrist_roll_link',
                'right_wrist_pitch_link',
                'right_wrist_yaw_link',
            )
        )

    def _wrist_contact_collider_path(self) -> str:
        return f'{self.bindings.hand_prim_path}/interaction_contact_collider'

    def _object_hand_is_in_contact(self, hand_position, object_position) -> bool:
        if self._object_hand_contact_count > self._object_hand_contact_seen:
            self._object_hand_contact_seen = self._object_hand_contact_count
            return True
        if self.object_hand_contact_distance <= 0.0:
            return False
        separation = float(
            np.linalg.norm(
                np.asarray(hand_position, dtype=float) - np.asarray(object_position, dtype=float)
            )
        )
        return separation <= self.object_hand_contact_distance

    def _on_contact_report(self, contact_headers, contact_data):
        for header in contact_headers:
            if header.num_contact_data <= 0:
                continue
            paths = tuple(
                str(PhysicsSchemaTools.intToSdfPath(value))
                for value in (
                    header.actor0,
                    header.actor1,
                    header.collider0,
                    header.collider1,
                )
            )
            arm_path = next(
                (path for path in paths if any(_paths_overlap(path, arm) for arm in self._arm_contact_paths())),
                None,
            )
            door_path = next(
                (path for path in paths if _paths_overlap(path, self.bindings.door_prim_path)),
                None,
            )
            object_path = next(
                (path for path in paths if _paths_overlap(path, self.bindings.object_prim_path)),
                None,
            )
            if arm_path is not None and door_path is not None:
                self._door_arm_contact_count += int(header.num_contact_data)
                self._door_arm_contact_pair = [arm_path, door_path]
                start = int(header.contact_data_offset)
                stop = start + int(header.num_contact_data)
                for index in range(start, stop):
                    contact = contact_data[index]
                    impulse = float(np.linalg.norm(np.asarray(contact.impulse, dtype=float)))
                    self._door_arm_contact_max_impulse = max(
                        self._door_arm_contact_max_impulse,
                        impulse,
                    )
            if arm_path is not None and object_path is not None:
                self._object_hand_contact_count += int(header.num_contact_data)
                self._object_hand_contact_pair = [arm_path, object_path]
                start = int(header.contact_data_offset)
                stop = start + int(header.num_contact_data)
                for index in range(start, stop):
                    contact = contact_data[index]
                    impulse = float(np.linalg.norm(np.asarray(contact.impulse, dtype=float)))
                    self._object_hand_contact_max_impulse = max(
                        self._object_hand_contact_max_impulse,
                        impulse,
                    )

    def _hand_position(self, robot_observation: Mapping):
        controller_observation = robot_observation.get('controllers', {}).get('right_arm_ik_controller', {})
        eef_position = controller_observation.get('eef_position')
        if eef_position is not None:
            return _vector3(eef_position)
        return _vector3(self._pose(self.bindings.hand_prim_path)[0])

    def _validate_prim_paths(self):
        required_paths = (
            self.bindings.obstacle_prim_path,
            self.bindings.object_prim_path,
            self.bindings.door_prim_path,
            self.bindings.hand_prim_path,
            self.bindings.torso_prim_path,
        )
        missing = [path for path in required_paths if not is_prim_path_valid(path)]
        if missing:
            raise RuntimeError(f'interaction prims do not exist: {missing}')

    @staticmethod
    def _pose(prim_path: str):
        return get_world_pose(prim_path)


def _vector3(value) -> tuple:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3:
        raise ValueError(f'expected a 3D vector, got {value}')
    return tuple(float(component) for component in array[:3])


def _optional_bool(mapping: Mapping, key: str):
    if key not in mapping:
        return None
    return bool(mapping[key])


def _optional_float(mapping: Mapping, key: str):
    value = mapping.get(key)
    return None if value is None else float(value)


def _relative_position(object_position, frame_position, frame_orientation):
    w, x, y, z = np.asarray(frame_orientation, dtype=float).reshape(4)
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return rotation.T @ (np.asarray(object_position) - np.asarray(frame_position))


def _paths_overlap(left: str, right: str) -> bool:
    left = left.rstrip('/')
    right = right.rstrip('/')
    return left == right or left.startswith(f'{right}/') or right.startswith(f'{left}/')


def _yaw(quaternion) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=float).reshape(4)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _normalized_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi
