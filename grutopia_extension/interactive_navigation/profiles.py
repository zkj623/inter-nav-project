"""Scene profiles and JSON loading for the interaction navigation demo."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from grutopia_extension.interactive_navigation.state_machine import (
    InteractionPlan,
    Quaternion,
    Vector3,
)


@dataclass(frozen=True)
class SceneBindings:
    """USD prims observed or modified by the simulator adapter."""

    obstacle_prim_path: str
    object_prim_path: str
    door_prim_path: str
    robot_prim_path: str = '/World/env_0/robots/h1_with_hand'
    hand_prim_path: str = '/World/env_0/robots/h1_with_hand/right_hand_link'

    @property
    def attachment_joint_path(self) -> str:
        return f'{self.object_prim_path}/interaction_pick_joint'

    @property
    def torso_prim_path(self) -> str:
        return f'{self.robot_prim_path}/torso_link'

    @property
    def robot_lock_joint_path(self) -> str:
        return f'{self.robot_prim_path}/interaction_base_joint'


@dataclass(frozen=True)
class DemoProfile:
    """Robot spawn, task plan, and scene prim bindings."""

    robot_start: Vector3
    plan: InteractionPlan
    bindings: SceneBindings
    robot_orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)


def programmatic_profile() -> DemoProfile:
    """Return the calibrated profile for ``ProceduralHousehold``."""

    root = '/World/env_0/objects/interactive_household'
    return DemoProfile(
        robot_start=(0.0, 0.0, 1.05),
        plan=InteractionPlan(
            obstacle_path=((0.85, 0.0, 1.05),),
            object_path=((2.40, -0.62, 1.05),),
            door_path=((4.65, -0.10, 1.05),),
            goal_path=((7.0, 0.0, 1.05),),
            pick_offset=(0.0, 0.0, 0.04),
            door_contact_position=(4.84, -0.20, 1.05),
            door_push_position=(5.00, -0.16, 1.05),
            arm_orientation=None,
        ),
        bindings=SceneBindings(
            obstacle_prim_path=f'{root}/obstacle',
            object_prim_path=f'{root}/carry_object',
            door_prim_path=f'{root}/door',
        ),
    )


def load_profile(path: str) -> DemoProfile:
    """Load a calibrated official-scene profile from JSON."""

    with Path(path).open('r', encoding='utf-8') as profile_file:
        data = json.load(profile_file)
    if not isinstance(data, Mapping):
        raise ValueError('profile root must be a JSON object')

    plan_data = _mapping(data, 'plan')
    bindings_data = _mapping(data, 'bindings')
    orientation = plan_data.get('arm_orientation')
    carry_path = plan_data.get('carry_path')
    return DemoProfile(
        robot_start=_vector3(data['robot_start'], 'robot_start'),
        plan=InteractionPlan(
            obstacle_path=_path(plan_data['obstacle_path'], 'obstacle_path'),
            object_path=_path(plan_data['object_path'], 'object_path'),
            door_path=_path(plan_data['door_path'], 'door_path'),
            goal_path=_path(plan_data['goal_path'], 'goal_path'),
            pick_offset=_vector3(plan_data['pick_offset'], 'pick_offset'),
            door_contact_position=_vector3(plan_data['door_contact_position'], 'door_contact_position'),
            door_push_position=_vector3(plan_data['door_push_position'], 'door_push_position'),
            arm_orientation=None if orientation is None else tuple(float(value) for value in orientation),
            carry_path=None if carry_path is None else _path(carry_path, 'carry_path'),
        ),
        bindings=SceneBindings(
            obstacle_prim_path=str(bindings_data['obstacle_prim_path']),
            object_prim_path=str(bindings_data['object_prim_path']),
            door_prim_path=str(bindings_data['door_prim_path']),
            robot_prim_path=str(bindings_data.get('robot_prim_path', '/World/env_0/robots/h1_with_hand')),
            hand_prim_path=str(
                bindings_data.get('hand_prim_path', '/World/env_0/robots/h1_with_hand/right_hand_link')
            ),
        ),
        robot_orientation=_quaternion(
            data.get('robot_orientation', (1.0, 0.0, 0.0, 0.0)),
            'robot_orientation',
        ),
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent[key]
    if not isinstance(value, Mapping):
        raise ValueError(f'{key} must be a JSON object')
    return value


def _path(value: Any, name: str) -> Tuple[Vector3, ...]:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a JSON array')
    return tuple(_vector3(point, f'{name} waypoint') for point in value)


def _vector3(value: Any, name: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} must contain exactly three numbers')
    return tuple(float(component) for component in value)


def _quaternion(value: Any, name: str) -> Quaternion:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f'{name} must contain exactly four numbers')
    return tuple(float(component) for component in value)
