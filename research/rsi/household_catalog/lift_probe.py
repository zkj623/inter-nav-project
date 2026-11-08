"""Isolated catalog lift probes. No base transport, no changes to frozen runtime."""
import argparse, json, sys, traceback, hashlib, os
from pathlib import Path
from dataclasses import asdict
from catalog_support import DEFAULT_CATALOG, REPOSITORY, input_fingerprint, verify_catalog

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=DEFAULT_CATALOG)
p.add_argument('--runtime', type=Path, default=REPOSITORY)
p.add_argument('--robot-usd', type=Path, default=os.environ.get('R1LITE_USD_PATH'))
p.add_argument('--collision-deps', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--asset', required=True)
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--geometry-only', action='store_true')
a = p.parse_args()
verify_catalog(a.root)
if a.robot_usd is None:
    p.error('Set --robot-usd or R1LITE_USD_PATH')
if not a.robot_usd.is_file() or not a.robot_usd.with_suffix('.urdf').is_file():
    p.error('Robot USD and sibling URDF must both exist')
os.environ['R1LITE_USD_PATH'] = str(a.robot_usd.resolve())
a.output.mkdir(parents=True, exist_ok=False)
record = next(x for x in json.loads((a.root/'asset_manifest.json').read_text())['assets'] if x['id']==a.asset)
result = dict(asset_id=a.asset, input_fingerprint=input_fingerprint(a.runtime,a.root,a.robot_usd), success=False, transport_executed=False, training_performed=False,
    scope='exploratory original-scale single-rigid-body side-grasp lift',
    mass_kg=1.0, mass_source='explicit common screening assumption; not measured',
    static_friction=1.0, dynamic_friction=0.9, hold_required_seconds=3.0,
    minimum_lift_m=0.15, geometry_audit='same PhysX cooked convex union for native/planned finger queries; conservative body envelope',
    runtime_sha256=hashlib.sha256((a.runtime/'grutopia_extension/interaction_clearance/primitives.py').read_bytes()).hexdigest())
(a.output/'protocol.json').write_text(json.dumps(result, indent=2))
sys.path[:0] = [str(a.runtime)]
if a.collision_deps:
    sys.path.insert(0, str(a.collision_deps))
from isaacsim import SimulationApp
app = SimulationApp({'headless':True, 'active_gpu':a.gpu, 'physics_gpu':a.gpu,
                     'multi_gpu':False, 'renderer':'RayTracedLighting'})
env = recorder = None
try:
    import carb, numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from load_asset import add_asset
    import grutopia_extension.manipulation.r1lite_bimanual_lift_sim as lift_module
    from grutopia_extension.interaction_clearance.environment import ClearanceEnvironment
    from grutopia_extension.interaction_clearance.task import ClearanceTask, Region, SceneObstacle
    from grutopia_extension.interaction_clearance.primitives import primitive_registry, grasp_contact_retained
    from grutopia_extension.interaction_clearance.recording import EpisodeRecorder
    from scipy.spatial.transform import Rotation
    from geometry_adapter import install_geometry_audit,configure_surface_grasp
    install_geometry_audit()
    carb.settings.get_settings().set_string_array('/app/mdl/additionalUserPaths',
        [str(a.root/'assets'/a.asset/'source/Materials')])
    def create_catalog_asset(box, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(destination))
        UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
        prim = add_asset(stage, a.asset, '/Object', mass_kg=1., static_friction=1., dynamic_friction=.9,
                         package_root=a.root)
        stage.SetDefaultPrim(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(32)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverVelocityIterationCountAttr(8)
        meshes = []
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI):
                if UsdGeom.Imageable(child).ComputeVisibility()==UsdGeom.Tokens.invisible:
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
                    continue
                c = PhysxSchema.PhysxCollisionAPI.Apply(child)
                c.CreateContactOffsetAttr(.002); c.CreateRestOffsetAttr(.0001)
                meshes.append(str(child.GetPath()))
        if not meshes: raise RuntimeError('NO_NATIVE_MESH_COLLIDERS')
        from bake_collision_meshes import bake_collision_meshes
        meshes=bake_collision_meshes(stage,prim,a.output/'baked_collision_manifest.json')
        (a.output/'colliders.json').write_text(json.dumps(dict(approximation='convexDecomposition', meshes=meshes),indent=2))
        stage.GetRootLayer().Save()
        return destination
    # Process-local injection only: preserve all robot/controller source files.
    lift_module.create_unhandled_box = create_catalog_asset
    task = ClearanceTask(task_id='catalog_lift_'+a.asset, object_id=a.asset,
        dimensions=tuple(record['dimensions_m']), mass_kg=1., object_xy=(.65,0.),
        place_xy=(.65,2.), placement=Region((-.5,1.),(1.8,3.)),
        scene_obstacles=(SceneObstacle('DistantBoundary',(4.,4.,.5),(.1,.1,1.)),))
    env = ClearanceEnvironment(a.output, task)
    result['cooked_geometry_sha256']=env.object_geometry.sha256
    result['cooked_convex_hulls']=len(env.object_geometry.parts)
    result['surface_grasp']=configure_surface_grasp(env)
    result['nominal_preload_m']=env.box.squeeze
    if a.geometry_only:
        result['success']=True;result['reason']='COOKED_GEOMETRY_AND_SURFACE_PROFILE_READY'
        raise SystemExit(0)
    recorder = EpisodeRecorder(a.output, task, 'isolated_catalog_lift_probe_v1')
    env.recorder = recorder
    start_bottom = env.object_bottom()
    start_base = env.base.get_world_pose()[0].copy()
    recorder.snapshot(env,'initial')
    registry = primitive_registry(env)
    outcome = registry['arm_plan'].execute(task, mode='grasp')
    if outcome.reason == 'OUTSIDE_INITIAL_POLICY_CAPABILITY':
        # Extend the legacy size whitelist for this exploratory script only.
        # All actual dimensions, IK, path checks, physics and force limits stay intact.
        from grutopia_extension.interaction_clearance.primitives import measured_base, object_transform, hand_clearance_candidates
        from grutopia_extension.interaction_clearance.motion_geometry import planar_transform, quaternion_yaw
        from grutopia_extension.interaction_clearance.skill_runtime import Outcome
        planner=registry['arm_plan']
        initial=env.robot.get_joint_positions()[env.control_indices].copy()
        center,quat=env.object.get_world_pose(); yaw=quaternion_yaw(quat)
        base=planar_transform(measured_base(env),env.base.get_world_pose()[0][2])
        rejected=[]
        result['legacy_size_whitelist_extended']=True
        for clearance in hand_clearance_candidates('grasp',.13):
            try:
                targets,phases,durations=planner.targets('grasp',center,initial,clearance)
                audit=planner.certify('grasp',initial,targets,phases,base,center,yaw,object_transform(env))
            except RuntimeError as exc:
                rejected.append(dict(clearance=clearance,error=str(exc))); continue
            env.arm_plan=dict(mode='grasp',initial=initial,targets=targets,phases=phases,
                durations=durations,base=measured_base(env),object=center,object_yaw=yaw,
                step=env.elapsed_steps,approach_clearance_m=clearance,rejected_approaches=rejected)
            (a.output/'extended_grasp_plan.json').write_text(json.dumps(audit,indent=2))
            outcome=Outcome(True,'EXTENDED_SIZE_PLAN_READY',dict(clearance=clearance))
            break
        else:
            outcome=Outcome(False,'NO_CERTIFIED_EXTENDED_SIZE_APPROACH',dict(rejected=rejected))
    result['plan'] = asdict(outcome)
    if not outcome.success: raise RuntimeError('PLAN_REJECTED: '+str(outcome.details))
    outcome = registry['grasp_lift'].execute(task)
    result['grasp_lift'] = asdict(outcome)
    if not outcome.success: raise RuntimeError('GRASP_LIFT_FAILED: '+str(outcome.details))
    # Independently observe a continuous three-second stationary hold.
    reference = None
    holds = []
    target = env.command[env.control_indices].copy()
    for i in range(150):
        env.advance(target,'hold',(i+1)/150)
        pose, quat = env.object.get_world_pose()
        tools = env.tool_positions()
        relative = []
        # Relative pose to each actual end effector, not world drift alone.
        ot=np.eye(4);ot[:3,3]=pose;ot[:3,:3]=env.rotation(quat)
        for rigid in (env.left_tool,env.tool):
            tp,tq=rigid.get_world_pose(); tool=np.eye(4);tool[:3,3]=tp;tool[:3,:3]=env.rotation(tq)
            relative.append(np.linalg.inv(tool)@ot)
        if reference is None: reference=relative
        slip=max(float(np.linalg.norm(x[:3,3]-y[:3,3])) for x,y in zip(relative,reference))
        angle=max(float(Rotation.from_matrix(x[:3,:3].T@y[:3,:3]).magnitude()) for x,y in zip(relative,reference))
        row=dict(step=i, lift_m=float(env.object_bottom()-start_bottom), slip_m=slip,
                 relative_rotation_rad=angle, bilateral_contact=bool(grasp_contact_retained(env)),
                 base_drift_m=float(np.linalg.norm(env.base.get_world_pose()[0][:2]-start_base[:2])))
        holds.append(row)
        if row['lift_m']<.15 or slip>.003 or angle>.05 or not row['bilateral_contact']:
            raise RuntimeError('UNSTABLE_HOLD: '+str(row))
    result['success']=True
    result['reason']='STATIONARY_LIFT_AND_3S_HOLD_PASSED_ONCE'
except Exception as exc:
    result['reason']=str(exc)
    (a.output/'exception.txt').write_text(traceback.format_exc())
finally:
    if 'holds' in globals():
        (a.output/'hold_samples.json').write_text(json.dumps(holds,indent=2))
    if env is not None:
        result['physics_controls']=env.elapsed_steps
        try:
            result['collision_audit']=env.save_collision_audit()
            result['success'] &= bool(result['collision_audit']['passed'])
            if recorder: recorder.snapshot(env,'final')
        except Exception as exc:
            result['success']=False;result['finalization_error']=str(exc)
    if recorder: recorder.close(result)
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print('PROBE_RESULT '+json.dumps(result),flush=True)
    app.close()
