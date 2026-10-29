"""Regression checks for physics configuration before the first simulation step."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def install(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


def load(relative):
    spec = importlib.util.spec_from_file_location('physics_initialization_test', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('profile', ['legacy', 'isaaclab_training'])
def test_asset_scope_survives_core_resolving_the_articulation_to_base(monkeypatch, profile):
    events = []

    class NativeView:
        _backend_utils = SimpleNamespace(expand_dims=np.expand_dims)
        velocity_limits = np.array([30.1] * 8 + [15.7] * 4)
        def set_gains(self, **kwargs):
            pass
        def set_max_joint_velocities(self, values, **kwargs):
            self.velocity_limits = values[0].copy()
        def set_max_efforts(self, values, **kwargs):
            self.max_efforts = values.copy()
        def set_solver_position_iteration_counts(self, values):
            pass
        def set_solver_velocity_iteration_counts(self, values):
            pass
        def set_enabled_self_collisions(self, values):
            pass
        def set_friction_coefficients(self, **kwargs):
            pass

    class NativeRobot:
        def __init__(self, prim_path, **kwargs):
            events.append(('initialize', prim_path))
            self.prim_path = prim_path + '/base'
            self._articulation_view = NativeView()
        def get_articulation_controller(self):
            return SimpleNamespace(switch_control_mode=lambda mode: None)

    class BaseRobot:
        @classmethod
        def register(cls, name):
            return lambda value: value

    install(monkeypatch, 'omni.isaac.core.articulations',
            ArticulationSubset=lambda *a: SimpleNamespace(joint_indices=np.arange(12)))
    install(monkeypatch, 'omni.isaac.core.prims', RigidPrim=object)
    install(monkeypatch, 'omni.isaac.core.robots.robot', Robot=NativeRobot)
    install(monkeypatch, 'omni.isaac.core.scenes', Scene=object)
    install(monkeypatch, 'omni.isaac.core.utils.prims', get_prim_at_path=lambda p: None)
    install(monkeypatch, 'omni.isaac.core.utils.stage',
            add_reference_to_stage=lambda **kw: events.append(('reference', kw['prim_path'])))
    install(monkeypatch, 'grutopia.core.robot.robot', BaseRobot=BaseRobot)
    install(monkeypatch, 'grutopia_extension.configs.robots.go2', GO2_JOINT_NAMES=list(range(12)))
    install(monkeypatch, 'grutopia_extension.robots.go2_asset',
            GO2_DEFAULT_JOINT_POSITIONS=[0.] * 12, ensure_go2_usd=lambda p, **kw: p)
    install(monkeypatch, 'grutopia_extension.controllers.isaac_go2_ctrl',
            STIFFNESS=25., DAMPING=.5, SATURATION_EFFORT=23.5, VELOCITY_LIMIT=30.)
    module = load('grutopia_extension/robots/go2.py')
    monkeypatch.setattr(module, 'apply_isaaclab_go2_physics', lambda p: events.append(('physics', p)))
    training_calls = []
    monkeypatch.setattr(module, 'apply_training_go2_physics', lambda p: training_calls.append(p))
    robot = module.Go2('/World/Go2', 'asset.usd', 'go2', physics_profile=profile)
    assert events == [('reference', '/World/Go2'), ('physics', '/World/Go2'), ('initialize', '/World/Go2')]
    assert robot.prim_path == '/World/Go2/base'
    robot.configure_locomotion('explicit_dc')
    assert events[-1] == ('physics', '/World/Go2')
    expected = [30.] * 12 if profile == 'isaaclab_training' else [30.1] * 8 + [15.7] * 4
    np.testing.assert_allclose(robot._articulation_view.velocity_limits, expected)
    assert training_calls == (['/World/Go2'] * 2 if profile == 'isaaclab_training' else [])
    np.testing.assert_allclose(robot._articulation_view.max_efforts, 1.e9)


def test_floor_contact_is_authored_before_scene_registration(monkeypatch):
    from grutopia_extension.configs.objects import FixedCubeCfg
    events = []

    class Material:
        def __init__(self, **kwargs):
            events.append(('material', kwargs))
    class Cube:
        prim = None
        def __init__(self, **kwargs):
            events.append(('cube', kwargs))
        def set_torsional_patch_radius(self, value):
            events.append(('patch', value))
        def set_min_torsional_patch_radius(self, value):
            events.append(('min_patch', value))
    class Common:
        def __init__(self, config):
            self._config = config
        @classmethod
        def register(cls, name):
            return lambda value: value

    install(monkeypatch, 'omni.isaac.core.objects', FixedCuboid=Cube)
    install(monkeypatch, 'omni.isaac.core.materials', PhysicsMaterial=Material)
    install(monkeypatch, 'omni.isaac.core.utils.semantics', add_update_semantics=lambda *a: None)
    install(monkeypatch, 'grutopia.core.scene.object', ObjectCommon=Common, Scene=object)
    module = load('grutopia_extension/objects/fixed_cube.py')
    cfg = FixedCubeCfg(name='floor', prim_path='/World/Floor', position=(0, 0, .005),
                       scale=(8, 8, .01), color=(.5, .5, .5), static_friction=1., dynamic_friction=1.,
                       torsional_patch_radius=0., min_torsional_patch_radius=0.)
    scene = SimpleNamespace(add=lambda cube: events.append(('register', cube)))
    module.FixedCube(cfg).set_up_scene(scene)
    assert [event[0] for event in events] == ['material', 'cube', 'patch', 'min_patch', 'register']
    assert events[0][1]['static_friction'] == events[0][1]['dynamic_friction'] == 1.
    assert isinstance(events[1][1]['physics_material'], Material)
    assert not events[0][1]['prim_path'].startswith(cfg.prim_path + '/')
    assert events[2][1] == events[3][1] == 0.
    events.clear()
    cfg.static_friction = None
    with pytest.raises(ValueError, match='friction'):
        module.FixedCube(cfg).set_up_scene(scene)
    assert not events
