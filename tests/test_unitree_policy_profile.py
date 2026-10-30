"""Validate the trained Unitree profile and its artifact/actuation contract."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

policy = load('unitree_profile_policy_test', 'grutopia_extension/controllers/go2_policy.py')
asset = load('unitree_profile_asset_test', 'grutopia_extension/robots/go2_asset.py')
NAMES = [f'{leg}_{part}_joint' for part in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR')]
DEFAULTS = np.asarray(asset.GO2_DEFAULT_JOINT_POSITIONS, dtype=np.float32)

@pytest.fixture
def artifact(tmp_path):
    actor = policy.build_go2_actor(45, (16, 8))
    path = tmp_path / 'actor_checkpoint.pt'
    checkpoint = {'model_state_dict': {'actor.' + key: value for key, value in actor.state_dict().items()},
                  'observation_profile': 'unitree_go2_45', 'empirical_normalization': False}
    torch.save(checkpoint, path)
    contract = {'actor_checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'observation_profile': 'unitree_go2_45', 'actor_dimension': 45, 'action_dimension': 12,
        'observation_order': ['base_ang_vel', 'projected_gravity', 'velocity_commands',
                              'joint_pos_rel', 'joint_vel_rel', 'last_action'],
        'observation_scales': [.2, 1., 1., 1., .05, 1.],
        'observation_clip_before_scale': [-100., 100.], 'empirical_normalization': False,
        'action_scale': .25, 'actor_hz': 50, 'physics_hz': 200, 'kp': 25., 'kd': .5,
        'saturation_torque_nm': 23.5, 'velocity_limit_rad_s': 30., 'joint_names': NAMES,
        'default_joint_positions': DEFAULTS.tolist()}
    (tmp_path / 'deployment_contract.json').write_text(json.dumps(contract))
    return path, checkpoint, contract

def test_verified_actor_load_preserves_output(artifact):
    path, checkpoint, _ = artifact
    actor = policy.load_go2_actor(str(path))
    reference = policy.build_go2_actor(45, (16, 8))
    reference.load_state_dict(policy.extract_actor_state(checkpoint))
    obs = torch.randn((4, 45))
    torch.testing.assert_close(actor(obs), reference(obs), rtol=0, atol=0)

@pytest.mark.parametrize('mutation,expected', [
    ({'observation_scales': [1.] * 6}, 'observation_scales'),
    ({'observation_order': ['wrong']}, 'observation_order'),
    ({'empirical_normalization': True}, 'empirical_normalization'),
    ({'physics_hz': 100}, 'physics_hz'),
    ({'actor_checkpoint_sha256': '0' * 64}, 'hash'),
])
def test_mismatched_export_contract_is_rejected(artifact, mutation, expected):
    path, _, contract = artifact
    contract.update(mutation)
    path.with_name('deployment_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=expected):
        policy.load_go2_actor(str(path))

def test_unidentified_45_dimensional_actor_is_rejected(artifact):
    path, checkpoint, _ = artifact
    del checkpoint['observation_profile']
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='declare'):
        policy.load_go2_actor(str(path))

def test_missing_sidecar_is_rejected(artifact):
    path, _, _ = artifact
    path.with_name('deployment_contract.json').unlink()
    with pytest.raises(ValueError, match='deployment_contract'):
        policy.load_go2_actor(str(path))

def test_native_clip_precedes_gyro_and_joint_speed_scaling():
    values = policy.build_go2_observation(
        [999, 999, 999], [200, -200, 2], [0, 0, -1], [.6, -.3, .1],
        DEFAULTS + 2, DEFAULTS, [200, -200] * 6, range(12), 45, base_height=99, ground_height=98)
    assert values.shape == (45,)
    np.testing.assert_allclose(values[:9], [20, -20, .4, 0, 0, -1, .6, -.3, .1])
    np.testing.assert_allclose(values[9:21], 2)
    np.testing.assert_allclose(values[21:33], [5, -5] * 6)
    np.testing.assert_allclose(values[33:], range(12))

def make_controller(monkeypatch, artifact, **kwargs):
    path, _, _ = artifact
    class Subset:
        def __init__(self, articulation, names):
            self.joint_indices = np.arange(12)
        def get_joint_positions(self):
            return DEFAULTS.copy()
        def get_joint_velocities(self):
            return np.zeros(12)
    for name, attrs in {
        'omni.isaac.core.articulations': {'ArticulationSubset': Subset},
        'omni.isaac.core.utils.types': {'ArticulationAction': SimpleNamespace},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'grutopia_extension.controllers.go2_policy', policy)
    monkeypatch.setitem(sys.modules, 'grutopia_extension.robots.go2_asset', asset)
    ctrl = load('unitree_profile_control_test', 'grutopia_extension/controllers/isaac_go2_ctrl.py')
    robot = SimpleNamespace(get_world_pose=lambda: (np.array([0., 0., .4]), np.array([1., 0., 0., 0.])),
                            get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
    return ctrl.Go2RSLControl(str(path), robot, NAMES, **kwargs)

@pytest.mark.parametrize('kwargs', [{'actuator_mode': 'implicit_pd'},
                                   {'actuator_mode': 'explicit_dc', 'stationary_joint_hold': True}])
def test_controller_rejects_untrained_actuation_mode(monkeypatch, artifact, kwargs):
    with pytest.raises(ValueError, match='explicit_dc without stationary'):
        make_controller(monkeypatch, artifact, **kwargs)

@pytest.mark.parametrize('field', ['joint_names', 'default_joint_positions'])
def test_controller_rejects_wrong_joint_contract(monkeypatch, artifact, field):
    path, _, contract = artifact
    contract[field] = contract[field][::-1]
    path.with_name('deployment_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='joint'):
        make_controller(monkeypatch, artifact, actuator_mode='explicit_dc')

def test_action_history_and_motor_target_share_native_clip(monkeypatch, artifact):
    control = make_controller(monkeypatch, artifact, actuator_mode='explicit_dc')
    control._simulation_time = lambda: 0.
    control.policy = lambda obs: torch.tensor([[200., -200.] * 6])
    control._infer_policy()
    np.testing.assert_allclose(control.raw_policy_action, [200., -200.] * 6)
    np.testing.assert_allclose(control.last_action, [100., -100.] * 6)
    np.testing.assert_allclose(control.applied_joint_positions, DEFAULTS + np.array([25., -25.] * 6))
