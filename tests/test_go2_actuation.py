import importlib.util,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_actuation import dc_motor_efforts,physics_steps_per_command


def test_motor_torque_speed_envelope_matches_isaac_lab():
    velocity=np.array([0.,15.,30.,-15.])
    np.testing.assert_allclose(dc_motor_efforts(np.ones(4),np.zeros(4),velocity),[23.5,11.75,0.,23.5])
    np.testing.assert_allclose(dc_motor_efforts(-np.ones(4),np.zeros(4),-velocity),[-23.5,-11.75,0.,-23.5])


def test_invalid_motor_state_is_rejected():
    with pytest.raises(ValueError):dc_motor_efforts([1,2],[0],[0,0])
    with pytest.raises(ValueError):dc_motor_efforts([1],[0],[float('nan')])


def test_exact_physics_steps_keep_every_render_at_a_command_boundary():
    physical=0;frames=[]
    for tick in range(100):
        count=physics_steps_per_command(tick,'explicit_dc')
        for index in range(count):
            physical+=1
            if physical%8==0:
                assert index==count-1
                frames.append(tick)
    assert physical==160 and len(frames)==20
    assert set(np.diff(frames))=={5}
    assert physical*.005==.8
    assert physics_steps_per_command(4,'implicit_pd')==1


def test_efforts_update_between_policy_inferences_and_reject_batched_physics(monkeypatch):
    for name,values in {'omni.isaac.core.articulations':{'ArticulationSubset':object},
                        'omni.isaac.core.utils.types':{'ArticulationAction':SimpleNamespace}}.items():
        module=ModuleType(name);module.__dict__.update(values);monkeypatch.setitem(sys.modules,name,module)
    root=Path(__file__).resolve().parents[1]
    for name,relative in [('grutopia_extension.controllers.go2_policy','grutopia_extension/controllers/go2_policy.py'),
                          ('grutopia_extension.robots.go2_asset','grutopia_extension/robots/go2_asset.py')]:
        spec=importlib.util.spec_from_file_location(name,root/relative);dependency=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dependency);monkeypatch.setitem(sys.modules,name,dependency)
    path=Path(__file__).resolve().parents[1]/'grutopia_extension/controllers/isaac_go2_ctrl.py'
    spec=importlib.util.spec_from_file_location('explicit_go2_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    control=module.Go2RSLControl.__new__(module.Go2RSLControl)
    control.actuator_mode='explicit_dc';control._last_control_time=None;control.applied_joint_positions=np.ones(12)
    clock=[0.];velocity=np.zeros(12)
    control._simulation_time=lambda:clock[0];control._should_infer=lambda:False
    control.joint_subset=SimpleNamespace(joint_indices=np.arange(12),get_joint_positions=lambda:np.zeros(12),get_joint_velocities=lambda:velocity)
    first=control.step();np.testing.assert_allclose(first.joint_efforts,23.5)
    velocity[:]=15.;clock[0]=.005
    second=control.step();np.testing.assert_allclose(second.joint_efforts,11.75)
    clock[0]=.025
    with pytest.raises(RuntimeError,match='5 ms'):control.step()


def test_run_configuration_propagates_exact_dc_to_both_controller_trees():
    from grutopia_extension.interactive_navigation.go2_navigation_runner import Go2NavigationRunConfig,build_go2_navigation_config
    from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
    cfg=build_go2_navigation_config(programmatic_go2_profile(),Go2NavigationRunConfig(actuator_mode='explicit_dc'))
    assert cfg.simulator.physics_dt==cfg.simulator.rendering_dt==.005
    assert cfg.simulator.rendering_interval==7
    robot=cfg.task_config.episodes[0].robots[0]
    assert robot.controllers[0].actuator_mode=='explicit_dc'
    assert robot.controllers[1].sub_controllers[0].sub_controllers[0].actuator_mode=='explicit_dc'
