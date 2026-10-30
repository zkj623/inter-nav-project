"""Isaac Lab RSL-RL Go2 actor loading and observation packing."""

from pathlib import Path
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

UNITREE_OBSERVATION_DIM = 45
UNITREE_OBSERVATION_PROFILE = "unitree_go2_45"
FLAT_OBSERVATION_DIM = 48
HEIGHT_SCAN_DIM = 187
ROUGH_OBSERVATION_DIM = FLAT_OBSERVATION_DIM + HEIGHT_SCAN_DIM
ACTION_DIM = 12
HEIGHT_SCAN_OFFSET = 0.5
HEIGHT_SCAN_CLIP = (-1.0, 1.0)
DEFAULT_ACTION_SCALE = 0.25
FLAT_HIDDEN_DIMS = (128, 128, 128)
ROUGH_HIDDEN_DIMS = (512, 256, 128)


def infer_actor_hidden_dims(actor_state: dict) -> tuple[int, ...]:
    """Return hidden layer widths from a Sequential actor state dict."""

    weight_layers = sorted(
        int(key.split('.', maxsplit=1)[0])
        for key in actor_state
        if key.endswith('.weight')
    )
    if len(weight_layers) < 2:
        raise ValueError('Go2 actor checkpoint is missing linear layers')
    return tuple(
        int(actor_state[f'{index}.weight'].shape[0])
        for index in weight_layers[:-1]
    )


def infer_observation_dim(actor_state: dict) -> int:
    first_weight = actor_state.get('0.weight')
    if first_weight is None:
        raise ValueError('Go2 actor checkpoint is missing layer 0')
    return int(first_weight.shape[1])


def build_go2_actor(
    observation_dim: int,
    hidden_dims: Sequence[int],
    action_dim: int = ACTION_DIM,
) -> nn.Module:
    if observation_dim <= 0 or action_dim <= 0:
        raise ValueError('Go2 actor dimensions must be positive')
    layers: list[nn.Module] = []
    input_dim = observation_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ELU())
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, action_dim))
    return nn.Sequential(*layers)


def extract_actor_state(checkpoint: dict) -> dict:
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith('actor.'):
            actor_state[key.removeprefix('actor.')] = value
        elif key.startswith('module.actor.'):
            actor_state[key.removeprefix('module.actor.')] = value
    if not actor_state:
        raise ValueError('Go2 checkpoint does not contain an actor state dict')
    return actor_state


def load_go2_actor(path: str) -> nn.Module:
    """Load only the deterministic actor from an Isaac Lab RSL-RL checkpoint."""

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Go2 locomotion checkpoint not found: {checkpoint_path}. '
            'Run grutopia/demo/download_go2_policy.py first or pass --policy-path.'
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor_state = extract_actor_state(checkpoint)
    actor = build_go2_actor(
        infer_observation_dim(actor_state),
        infer_actor_hidden_dims(actor_state),
    )
    actor.load_state_dict(actor_state, strict=True)
    if infer_observation_dim(actor_state) == UNITREE_OBSERVATION_DIM:
        if checkpoint.get('observation_profile') != UNITREE_OBSERVATION_PROFILE:
            raise ValueError('45-dimensional Go2 checkpoint must declare unitree_go2_45 profile')
        if checkpoint.get('empirical_normalization') is not False:
            raise ValueError('Unitree Go2 adapter requires explicit unnormalized observations')
        contract_path = checkpoint_path.with_name('deployment_contract.json')
        if not contract_path.is_file():
            raise ValueError('Unitree Go2 checkpoint requires deployment_contract.json')
        contract = json.loads(contract_path.read_text())
        expected = {
            'observation_profile': UNITREE_OBSERVATION_PROFILE,
            'actor_dimension': 45, 'action_dimension': 12,
            'observation_order': ['base_ang_vel', 'projected_gravity', 'velocity_commands',
                                  'joint_pos_rel', 'joint_vel_rel', 'last_action'],
            'observation_scales': [.2, 1., 1., 1., .05, 1.],
            'observation_clip_before_scale': [-100., 100.],
            'empirical_normalization': False, 'action_scale': .25,
            'actor_hz': 50, 'physics_hz': 200, 'kp': 25., 'kd': .5,
            'saturation_torque_nm': 23.5, 'velocity_limit_rad_s': 30.,
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(f'Unitree Go2 deployment contract mismatch for {key}')
        if contract.get('actor_checkpoint_sha256') != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest():
            raise ValueError('Unitree Go2 checkpoint hash differs from deployment contract')
        actor.go2_deployment_contract = contract
    actor.eval()
    return actor


def yaw_from_wxyz(quaternion: Iterable[float]) -> float:
    """Return Z-up yaw from an Isaac Sim (w, x, y, z) quaternion."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def angular_velocity_from_quaternions(
    previous: Iterable[float],
    current: Iterable[float],
    dt: float,
) -> np.ndarray:
    """World-frame angular velocity in rad/s from two (w, x, y, z) poses."""

    if dt <= 0.0:
        raise ValueError('dt must be positive')
    q0 = np.asarray(previous, dtype=np.float64).reshape(4)
    q1 = np.asarray(current, dtype=np.float64).reshape(4)
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    delta = np.array(
        [
            w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0,
            -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
            -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
            -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0,
        ],
        dtype=np.float64,
    )
    delta[0] = np.clip(delta[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(delta[0])
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = delta[1:] / max(np.sin(angle * 0.5), 1e-12)
    return (axis * (angle / dt)).astype(np.float32)


def rotate_vector_inverse_wxyz(
    quaternion: Iterable[float],
    vector: Iterable[float],
) -> np.ndarray:
    """Rotate a world vector into the body frame of a (w, x, y, z) pose."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    vx, vy, vz = np.asarray(vector, dtype=np.float64).reshape(3)
    q_w = w
    q_vec = np.array([x, y, z], dtype=np.float64)
    value = np.array([vx, vy, vz], dtype=np.float64)
    a = value * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, value) * q_w * 2.0
    c = q_vec * np.dot(q_vec, value) * 2.0
    return (a - b + c).astype(np.float32)


def flat_ground_height_scan(
    base_height: float,
    ground_height: float = 0.0,
    size: int = HEIGHT_SCAN_DIM,
) -> np.ndarray:
    """Isaac Lab height_scan on a locally flat floor: z_base - z_hit - 0.5."""

    height = float(base_height) - float(ground_height) - HEIGHT_SCAN_OFFSET
    return np.full(
        size,
        np.clip(height, *HEIGHT_SCAN_CLIP),
        dtype=np.float32,
    )


def build_go2_observation(
    base_linear_velocity: Iterable[float],
    base_angular_velocity: Iterable[float],
    projected_gravity: Iterable[float],
    command: Iterable[float],
    joint_positions: Iterable[float],
    default_joint_positions: Iterable[float],
    joint_velocities: Iterable[float],
    last_action: Iterable[float],
    observation_dim: int,
    base_height: float = 0.4,
    ground_height: float = 0.0,
    height_scan: np.ndarray | None = None,
) -> np.ndarray:
    """Pack the Isaac Lab Go2 policy observation, including rough height scan."""

    observation = np.concatenate(
        [
            np.asarray(base_linear_velocity, dtype=np.float32).reshape(-1),
            np.asarray(base_angular_velocity, dtype=np.float32).reshape(-1),
            np.asarray(projected_gravity, dtype=np.float32).reshape(-1),
            np.asarray(command, dtype=np.float32).reshape(-1),
            np.asarray(joint_positions, dtype=np.float32).reshape(-1)
            - np.asarray(default_joint_positions, dtype=np.float32).reshape(-1),
            np.asarray(joint_velocities, dtype=np.float32).reshape(-1),
            np.asarray(last_action, dtype=np.float32).reshape(-1),
        ],
        dtype=np.float32,
    )
    if observation.shape != (FLAT_OBSERVATION_DIM,):
        raise RuntimeError(
            'Go2 proprioceptive observation has shape '
            f'{observation.shape}, expected ({FLAT_OBSERVATION_DIM},)'
        )
    if observation_dim == UNITREE_OBSERVATION_DIM:
        # Unitree's actor omits base linear velocity and height. Native Lab
        # clips each observation term before scaling it; preserve that order.
        unitree = np.clip(observation[3:], -100., 100.).copy()
        unitree[:3] *= .2
        unitree[21:33] *= .05
        return unitree
    if observation_dim == FLAT_OBSERVATION_DIM:
        return observation
    if observation_dim != ROUGH_OBSERVATION_DIM:
        raise RuntimeError(
            f'unsupported Go2 policy observation dim {observation_dim}; '
            f'expected {UNITREE_OBSERVATION_DIM}, {FLAT_OBSERVATION_DIM} or {ROUGH_OBSERVATION_DIM}'
        )
    if height_scan is None:
        height_scan = flat_ground_height_scan(base_height, ground_height)
    height_scan = np.clip(
        np.asarray(height_scan, dtype=np.float32).reshape(-1),
        *HEIGHT_SCAN_CLIP,
    )
    if height_scan.shape != (HEIGHT_SCAN_DIM,):
        raise RuntimeError(
            'Go2 height scan has shape '
            f'{height_scan.shape}, expected ({HEIGHT_SCAN_DIM},)'
        )
    return np.concatenate([observation, height_scan], dtype=np.float32)
