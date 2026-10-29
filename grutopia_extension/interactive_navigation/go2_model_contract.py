"""Record requested and actual Go2 policy/asset/physics instead of trusting labels."""
from pathlib import Path
import hashlib
import numpy as np


def _values(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def collect_model_contract(robot,policy_path,requested_asset,profile):
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
    art=robot.isaac_robot;view=art._articulation_view;ctrl=robot._go2_locomotion_control
    stage=get_current_stage();root=stage.GetPrimAtPath(art._go2_asset_root_path)
    refs=root.GetMetadata('references')
    references=[] if refs is None else [r.assetPath for r in refs.GetAddedOrExplicitItems()]
    resolved=str(getattr(art,'_go2_resolved_usd_path',requested_asset))
    policy=Path(policy_path).expanduser().resolve();asset=Path(resolved)
    masses=_values(view.get_body_masses());indices=ctrl.joint_subset.joint_indices
    colliders=sum(p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
        and p.GetAttribute('physics:collisionEnabled').Get() is not False
        for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()))
    legs={}
    for name in view.body_names:
        if not name.endswith(('_foot','_calf')):continue
        prim=stage.GetPrimAtPath(art._go2_asset_root_path+'/'+name)
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if not material:legs[name]=None;continue
        api=UsdPhysics.MaterialAPI(material.GetPrim());physx=PhysxSchema.PhysxMaterialAPI(material.GetPrim())
        legs[name]={'static':api.GetStaticFrictionAttr().Get(),'dynamic':api.GetDynamicFrictionAttr().Get(),
            'combine':physx.GetFrictionCombineModeAttr().Get()}
    base=PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(art._go2_asset_root_path+'/base'))
    max_linear=base.GetMaxLinearVelocityAttr().Get()
    report={'policy_path':str(policy),'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
        'policy_observation_dim':ctrl.observation_dim,'requested_robot_asset':str(requested_asset),
        'resolved_robot_asset':resolved,'actual_asset_references':references,
        'robot_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest() if asset.is_file() else None,
        'physics_profile':profile,'actuator_mode':ctrl.actuator_mode,'robot_scale':_values(robot._robot_scale),
        'body_count':view.num_bodies,'body_names':view.body_names,'body_masses':masses,'total_mass_kg':float(np.asarray(masses).sum()),
        'collider_count':int(colliders),'solver_position_iterations':_values(view.get_solver_position_iteration_counts()),
        'solver_velocity_iterations':_values(view.get_solver_velocity_iteration_counts()),
        'self_collisions':_values(view.get_enabled_self_collisions()),
        'native_joint_max_velocities':_values(view.get_joint_max_velocities(joint_indices=indices)),
        'native_joint_friction':_values(view.get_friction_coefficients(joint_indices=indices)),
        'native_gains':[_values(x) for x in view.get_gains(joint_indices=indices)],
        'shape_material_properties':_values(view._physics_view.get_material_properties()),
        'leg_materials':legs,'max_angular_velocity_degrees_per_s':base.GetMaxAngularVelocityAttr().Get(),
        'max_linear_velocity_mps':float(max_linear) if np.isfinite(max_linear) else 'inf'}
    return report


def validate_training_model_contract(report):
    def matches(value,expected,shape):
        try:array=np.asarray(value,float)
        except (TypeError,ValueError):return False
        return bool(array.shape==shape and np.isfinite(array).all() and np.allclose(array,expected,atol=1e-4))
    requested=report.get('requested_robot_asset');resolved=report.get('resolved_robot_asset')
    checks={
        'profile_selected':report.get('physics_profile')=='isaaclab_training',
        'explicit_dc':report.get('actuator_mode')=='explicit_dc',
        'original_scale':matches(report.get('robot_scale'),[1.,1.,1.],(3,)),
        'nineteen_bodies':report.get('body_count')==19,
        'intended_mass':abs(report.get('total_mass_kg',0.)-15.019)<.01,
        'all_colliders_present':report.get('collider_count')==27,
        'requested_asset_used':bool(requested and resolved and Path(requested).resolve()==Path(resolved).resolve()
            and resolved in report.get('actual_asset_references',[])),
        'source_hash_recorded':bool(report.get('robot_asset_sha256') and report.get('policy_sha256')),
        'native_speed_limits':matches(report.get('native_joint_max_velocities'),30.,(1,12)),
        'native_drives_off':matches(report.get('native_gains'),0.,(2,1,12)),
        'joint_friction_zero':matches(report.get('native_joint_friction'),0.,(1,12)),
        'solver_position':report.get('solver_position_iterations')==[4],
        'solver_velocity':report.get('solver_velocity_iterations')==[0],
        'self_collision_off':report.get('self_collisions')==[0],
        'rigid_angular_limit':report.get('max_angular_velocity_degrees_per_s')==1000.,
        'rigid_linear_limit':report.get('max_linear_velocity_mps')==1000.,
    }
    legs=report.get('leg_materials',{})
    checks['eight_leg_materials']=len(legs)==8 and all(v is not None and v.get('combine')=='multiply'
        and np.isclose(v.get('static',-1),.8) and np.isclose(v.get('dynamic',-1),.6) for v in legs.values())
    report['training_contract_checks']=checks;report['training_contract_valid']=all(checks.values())
    if not report['training_contract_valid']:
        raise RuntimeError('Go2 training contract mismatch: '+', '.join(k for k,v in checks.items() if not v))
    return report
