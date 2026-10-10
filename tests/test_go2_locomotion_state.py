"""Deployment-contract tests without starting Isaac or a physics simulation."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy_module = load_module('go2_policy_test', 'grutopia_extension/controllers/go2_policy.py')
asset_module = load_module('go2_asset_test', 'grutopia_extension/robots/go2_asset.py')
JOINTS = [f'{leg}_{part}_joint' for part in ('hip', 'thigh', 'calf')
          for leg in ('FL', 'FR', 'RL', 'RR')]
DEFAULTS = np.asarray(asset_module.GO2_DEFAULT_JOINT_POSITIONS, dtype=np.float32)
@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    actor = policy_module.build_go2_actor(235, (32, 16))
    path = tmp_path / 'rsl.pt'
    torch.save({'model_state_dict': {'actor.' + k: v for k, v in actor.state_dict().items()}}, path)
    return str(path)


def test_controller_command_transition_cadence_and_reset(monkeypatch, checkpoint):
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
    monkeypatch.setitem(sys.modules, 'grutopia_extension.controllers.go2_policy', policy_module)
    monkeypatch.setitem(sys.modules, 'grutopia_extension.robots.go2_asset', asset_module)
    ctrl_module = load_module('go2_ctrl_test', 'grutopia_extension/controllers/isaac_go2_ctrl.py')
    articulation = SimpleNamespace(
        get_world_pose=lambda: (np.array([0, 0, .5]), np.array([1, 0, 0, 0])),
        get_linear_velocity=lambda: np.zeros(3),
        get_angular_velocity=lambda: np.array([0, 0, 1]),
    )
    control = ctrl_module.Go2RSLControl(checkpoint, articulation, JOINTS)
    clock = [0.]
    control._simulation_time = lambda: clock[0]
    control.set_command(2, 0, -2)
    np.testing.assert_allclose(control.get_obs()['command'], [1.5, 0, -1.5])
    packed = []
    pack = ctrl_module.build_go2_observation

    def capture(*args, **kwargs):
        obs = pack(*args, **kwargs)
        packed.append(obs.copy())
        return obs

    monkeypatch.setattr(ctrl_module, 'build_go2_observation', capture)
    control.step()
    for now in [.005, .01, .015]:
        clock[0] = now
        control.step()
    assert len(packed) == control.policy_inferences == 1
    clock[0] = .02
    control.step()
    assert len(packed) == control.policy_inferences == 2
    # Changing command must preserve the RSL observation and inference cadence.
    for request in [(.35, -.04, .2), (0., 0., 0.), (-.1, 0., 0.)]:
        control.set_command(*request)
        clock[0] += .02
        control.step()
        np.testing.assert_allclose(packed[-1][9:12], request, atol=1e-7)
        np.testing.assert_array_equal(control.get_obs()['policy_command'], packed[-1][9:12])
    control.reset()
    np.testing.assert_array_equal(control.last_action, 0)
    np.testing.assert_array_equal(control.policy_command, 0)
    np.testing.assert_array_equal(control.applied_joint_positions, DEFAULTS)
    assert control._last_inference_time is None
    assert control.policy_inferences == 0
    control.step()
    clock[0] = 0.  # A simulation clock rewind must reset and resume inference.
    control.step()
    assert control._last_inference_time == 0.
    np.testing.assert_array_equal(packed[-1][36:48], 0)
    with pytest.raises(ValueError, match='finite'):
        control.set_command(float('nan'), 0, 0)


def test_direct_and_path_wrappers_share_one_policy_state(monkeypatch):
    class Base:
        def __init__(self, config, robot, scene):
            self.robot = robot
        @classmethod
        def register(cls, name):
            return lambda implementation: implementation

    created = []

    def make_control(*args, **kwargs):
        control = SimpleNamespace(
            joint_subset=object(),
            observation_dim=235,
        )
        created.append(control)
        return control

    dependencies = {
        'omni.isaac.core.scenes': {'Scene': object},
        'omni.isaac.core.utils.types': {'ArticulationAction': object},
        'grutopia.core.robot.controller': {'BaseController': Base},
        'grutopia.core.robot.robot': {'BaseRobot': object},
        'grutopia_extension.configs.controllers': {'Go2MoveBySpeedControllerCfg': object},
        'grutopia_extension.controllers.isaac_go2_ctrl': {'Go2RSLControl': make_control},
    }
    for name, attrs in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    module = load_module('go2_wrapper_test', 'grutopia_extension/controllers/go2_move_by_speed_controller.py')
    robot = SimpleNamespace(isaac_robot=object())
    config = SimpleNamespace(policy_weights_path='policy.pt', joint_names=JOINTS, ground_height=.15)
    direct = module.Go2MoveBySpeedController(config, robot, None)
    path = module.Go2MoveBySpeedController(config, robot, None)
    assert direct._control is path._control
    assert len(created) == 1
    config.policy_weights_path = 'different.pt'
    with pytest.raises(ValueError, match='same locomotion'):
        module.Go2MoveBySpeedController(config, robot, None)
