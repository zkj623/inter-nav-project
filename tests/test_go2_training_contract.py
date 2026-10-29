from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from grutopia_extension.interactive_navigation.go2_model_contract import validate_training_model_contract


def good_contract():
    return {'physics_profile':'isaaclab_training','actuator_mode':'explicit_dc','robot_scale':[1,1,1],
        'body_count':19,'total_mass_kg':15.019,'collider_count':27,
        'requested_robot_asset':'/models/go2.usdc','resolved_robot_asset':'/models/go2.usdc','actual_asset_references':['/models/go2.usdc'],
        'policy_sha256':'a'*64,'robot_asset_sha256':'b'*64,
        'native_joint_max_velocities':[[30.]*12],'native_gains':[[[0.]*12],[[0.]*12]],'native_joint_friction':[[0.]*12],
        'solver_position_iterations':[4],'solver_velocity_iterations':[0],'self_collisions':[0],
        'max_angular_velocity_degrees_per_s':1000.,'max_linear_velocity_mps':1000.,
        'leg_materials':{f'{leg}_{part}':{'static':.8,'dynamic':.6,'combine':'multiply'}
            for leg in ['FL','FR','RL','RR'] for part in ['foot','calf']}}


def test_valid_native_contract_is_json_serializable():
    result=validate_training_model_contract(good_contract())
    assert result['training_contract_valid']
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('key,value',[
    ('physics_profile','legacy'),('body_count',39),('total_mass_kg',17.019),('collider_count',26),
    ('robot_scale',[.65,.65,.65]),('native_joint_max_velocities',[[30.1]*8+[15.7]*4]),
    ('native_joint_max_velocities',[]),('native_gains',[[[25.]*12],[[.5]*12]]),
    ('solver_velocity_iterations',[16]),('max_angular_velocity_degrees_per_s',5729.58),
    ('actual_asset_references',['/models/other.usdc']),('requested_robot_asset','/models/other.usdc'),
    ('leg_materials',{}),('policy_sha256',None),
])
def test_observed_asset_and_physics_mismatches_are_rejected(key,value):
    report=good_contract();report[key]=value
    with pytest.raises(RuntimeError,match='Go2 training contract mismatch'):validate_training_model_contract(report)


def test_missing_contract_is_not_vacuously_valid():
    with pytest.raises(RuntimeError):validate_training_model_contract({})


def asset_module():
    path=Path(__file__).resolve().parents[1]/'grutopia_extension/robots/go2_asset.py'
    spec=importlib.util.spec_from_file_location('composed_go2_asset_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make_small_reference(tmp_path):
    from pxr import Usd,UsdGeom,UsdPhysics
    geometry=tmp_path/'parts.usda';stage=Usd.Stage.CreateNew(str(geometry))
    root=UsdGeom.Xform.Define(stage,'/Go2');stage.SetDefaultPrim(root.GetPrim())
    base=UsdGeom.Xform.Define(stage,'/Go2/base');UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    sphere=UsdGeom.Sphere.Define(stage,'/Go2/base/collisions');UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    for leg in ['FL','FR','RL','RR']:
        for joint in ['hip','thigh','calf']:UsdPhysics.RevoluteJoint.Define(stage,f'/Go2/base/{leg}_{joint}_joint')
    stage.GetRootLayer().Save()
    reference=tmp_path/'go2.usda';stage=Usd.Stage.CreateNew(str(reference))
    root=UsdGeom.Xform.Define(stage,'/Go2');root.GetPrim().GetReferences().AddReference('./parts.usda')
    stage.SetDefaultPrim(root.GetPrim());stage.GetRootLayer().Save()
    return reference


def test_small_composed_asset_is_honored_before_sibling_or_download(monkeypatch,tmp_path):
    module=asset_module();path=make_small_reference(tmp_path)
    assert path.stat().st_size<module._MIN_ISAACLAB_USD_BYTES
    assert module.is_composed_go2_usd(path)
    # A sibling that satisfies the legacy size check must not override an
    # explicit, valid composition.
    monkeypatch.setattr(module,'is_isaaclab_go2_usd',lambda p:Path(p).name=='isaaclab_go2.usd')
    monkeypatch.setattr(module,'_try_download_official_usd',lambda p:pytest.fail('Unexpected asset substitution'))
    assert Path(module.ensure_go2_usd(str(path),generate_fallback=False))==path


def test_asset_validation_requires_go2_joint_structure(tmp_path):
    from pxr import Usd,UsdGeom
    path=tmp_path/'not_go2.usda';stage=Usd.Stage.CreateNew(str(path))
    root=UsdGeom.Xform.Define(stage,'/Object');stage.SetDefaultPrim(root.GetPrim());stage.GetRootLayer().Save()
    assert not asset_module().is_composed_go2_usd(path)


def test_physics_profile_configuration_is_explicit():
    from grutopia_extension.configs.robots.go2 import Go2RobotCfg
    assert Go2RobotCfg().physics_profile=='legacy'
    assert Go2RobotCfg(physics_profile='isaaclab_training').physics_profile=='isaaclab_training'
    with pytest.raises(ValueError):Go2RobotCfg(physics_profile='unverified')
