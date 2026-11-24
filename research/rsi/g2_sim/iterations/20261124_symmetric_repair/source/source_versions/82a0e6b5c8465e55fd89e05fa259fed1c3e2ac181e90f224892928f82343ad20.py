"""Physical G2 catalog test with contact witnesses; no welds or object-following."""
from pathlib import Path
import sys,json,traceback,argparse,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=['/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps']
p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--candidate',type=int,default=0);p.add_argument('--repeat',type=int,default=1);p.add_argument('--gpu',type=int,default=0);p.add_argument('--diagnostic-no-self-collision',action='store_true');p.add_argument('--plan-dir',default='realpose_plans');p.add_argument('--pose-only',action='store_true');args=p.parse_args()
OUT=ROOT/'realpose_trials'/args.plan_dir/args.asset/f'candidate{args.candidate}_repeat{args.repeat}'
try:
    OUT.mkdir(parents=True,exist_ok=False)
except FileExistsError:
    deadline=time.monotonic()+1200
    while not (OUT/'result.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Existing trial has no completed result')
        time.sleep(2)
    print('EXISTING_COMPLETED_TRIAL '+str(OUT),flush=True)
    sys.exit(0)
plan_doc=json.loads((ROOT/args.plan_dir/(args.asset+'.json')).read_text())
plan=plan_doc['plans'][args.candidate] if plan_doc['plans'] else None
profile=json.loads((ROOT/'ground_lift_training_config.json').read_text())
assert plan.get('profile_id')==profile['profile_id'],'REALPOSE_PROFILE_MISMATCH'
assert all(abs(plan['q_start'].get(n,0)-v)<1e-8 for n,v in profile['common_initial_joint_positions_rad'].items()),'REALPOSE_COMMON_HOME_MISMATCH'
from collection_recorder import Recorder
head_pilot=None
recorder=Recorder(OUT,['idx21_arm_l_joint1', 'idx22_arm_l_joint2', 'idx23_arm_l_joint3', 'idx24_arm_l_joint4', 'idx25_arm_l_joint5', 'idx26_arm_l_joint6', 'idx27_arm_l_joint7', 'idx61_arm_r_joint1', 'idx62_arm_r_joint2', 'idx63_arm_r_joint3', 'idx64_arm_r_joint4', 'idx65_arm_r_joint5', 'idx66_arm_r_joint6', 'idx67_arm_r_joint7', 'idx31_gripper_l_inner_joint1', 'idx71_gripper_r_inner_joint1'],dict(asset=args.asset,placement=plan.get('collection_placement',{}),plan_dir=args.plan_dir,plan_sha256=hashlib.sha256((ROOT/args.plan_dir/(args.asset+'.json')).read_bytes()).hexdigest(),baseline_harness='directprep_basket_rectangular_trial.py',baseline_commit='52f4d1d1141f2634a4d1cfa26d3fe668e8cf149f'))
result=dict(test_type='empty_pose_hold' if args.pose_only else 'upright_grasp',asset=args.asset,candidate=args.candidate,repeat=args.repeat,success=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gravity_compensation=True,
    cartesian_feedback='arms-only feedback; fixed upright support planes; finger load balancing; continuous nominal lift interpolation; bounded squeeze; four-finger preload gate; no object pose writes',
    initialization='USD joint states, link transforms and drive targets seeded before world.reset per official Genie Sim guidance',
    physics_step_protocol='one explicit 1/120s step per control tick; render does not advance physics; motion duration x1.75; 10fps video',mass_kg=1.,mass_source='common screening assumption, not a real measured mass',
    static_friction=1.,dynamic_friction=.9,required_lift_m=.15,hold_seconds=3.,
    max_object_tilt_deg=20.,max_hold_relative_drift_m=.02,self_collision_enabled=True,
    max_loaded_visual_gap_m=.0015,max_contact_point_surface_error_m=.0025,
    robot_fixed_to_world=True,object_attachment=False,transport_tested=False,
    gripper_coupling='native PhysX mimic constraints; only master finger joints actuated',
    runtime='Isaac Sim 4.2 compatibility harness, official CRS + OmniPicker geometry')
(OUT/'protocol.json').write_text(json.dumps(result,indent=2))
sys.argv.extend(['--/plugins/carb.tasking.plugin/threadCount=16','--/plugins/omni.tbb.globalcontrol/maxThreadCount=16','--/persistent/physics/numThreads=16'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':args.gpu,'physics_gpu':args.gpu,'multi_gpu':False})
class TrialRejected(RuntimeError):pass
writer=None;samples=[];self_contacts=[];nonhand_contacts=[];last_frame=None;frames=0;step=0
try:
    import cv2,trimesh,fcl,omni.usd
    from scipy.spatial.transform import Rotation
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim,RigidPrimView
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.utils.viewports import set_camera_view
    from omni.isaac.sensor import Camera
    from g2_kinematics import Kinematics,transform
    k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf')
    from ideal_constraints import Envelope
    envelope=Envelope()
    for joint in k.joints:
        if 'body_joint' in joint['name']:joint['effort']=1e7
        elif 'gripper' in joint['name'] and not joint['mimic']:joint['effort']=min(joint['effort'],.8)
    result['repair_gripper_gains']={'kp':10.0,'kd':2.0,'effort_limits':'master grippers capped at0.8Nm; arms unchanged'}
    result['drive_assumption']='IDEAL body kp100000 kd5000 cap1e7; other joints baseline; corrected inertia; not real hardware validation'
    from fast_gravity_compensation import GravityCompensation
    model_gravity=GravityCompensation()
    result['gravity_implementation_sha256']=model_gravity.implementation_sha256
    result['gravity_equivalence']='102 pose tests vs archived scalar code; max error2.85e-13Nm; same URDF, mass, axes and torque caps'
    result['gravity_compensation_method']='URDF dU/dq; ideal body torque cap; other original limits'
    result['envelope']={'max_torso_tilt_deg':20.,'max_body_joint_deg':180.,'max_static_effort_fraction':.8}
    result['support_height_m']=plan.get('support_height_m',0.) if plan else 0.
    assets=ROOT/'robot_assets'
    # Prepare off the live stage so disabled legacy meshes cannot enqueue
    # needless convex decomposition before their collision schemas are stripped.
    stage=Usd.Stage.Open(str(assets/'g2_contact_inertia_fixed.usdc'))
    qpre=plan['q_pre'] if plan else {'idx31_gripper_l_inner_joint1':-.785,'idx71_gripper_r_inner_joint1':-.785}
    q_grasp_pre=qpre.copy()
    qpre=plan.get('q_default',plan.get('q_start',qpre)) if plan else qpre
    result['includes_descent']=bool(plan and 'q_start' in plan)
    seed_fk=k.fk(qpre,transform((0,0,.04)))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in seed_fk:
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(seed_fk[prim.GetName()].T.tolist()))
            body=UsdPhysics.RigidBodyAPI(prim);body.CreateVelocityAttr(Gf.Vec3f(0.));body.CreateAngularVelocityAttr(Gf.Vec3f(0.))
    # Each closed-chain gripper is represented as a mimic tree. Its CAD linkage
    # parts intentionally overlap at pins; internal contacts are not external
    # robot self-collisions. Keep hand-vs-body/other-hand/object contacts enabled.
    exclusions=[]
    for side in ('l','r'):
        assembly=[prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                  and (prim.GetName().startswith(f'gripper_{side}_') or prim.GetName()==f'arm_{side}_link7')]
        for i,prim in enumerate(assembly):
            targets=[q.GetPath() for q in assembly[i+1:]]
            if targets:
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(targets)
                exclusions.extend([[str(prim.GetPath()),str(t)] for t in targets])
    result['self_collision_exclusions']=exclusions
    # Consecutive two-joint housings share a mechanical assembly (e.g. wrist
    # links 5/6/7). Exclude only these local pairs, plus the head mount assembly.
    adj={n:set() for n in k.links}
    for j in k.joints:adj[j['parent']].add(j['child']);adj[j['child']].add(j['parent'])
    # Fixed URDF frames are merged by the importer, so include their neighbors
    # when deciding adjacency between the surviving rigid links.
    for j in k.joints:
        if j['kind']=='fixed':
            a,b=j['parent'],j['child'];neighbors=(adj[a]|adj[b])-{a,b}
            adj[a]|=neighbors;adj[b]|=neighbors
            for n in neighbors:adj[n]|={a,b}
    rigid=[prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    for i,prim in enumerate(rigid):
        a=prim.GetName();extra=[]
        for other in rigid[i+1:]:
            b=other.GetName()
            near=b in adj.get(a,set()) or any(b in adj.get(x,set()) for x in adj.get(a,set()))
            head=(a.startswith('head_') or a=='body_link5') and (b.startswith('head_') or b=='body_link5')
            if near or head:extra.append(other.GetPath());exclusions.append([str(prim.GetPath()),str(other.GetPath())])
        if extra:
            rel=UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel();rel.SetTargets(list(set(rel.GetTargets()+extra)))
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    def contacts(headers,data):
        for h in headers:
            a=str(PhysicsSchemaTools.intToSdfPath(h.actor0));b=str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            object_pair=(a.startswith('/robot/') and b.startswith('/World/TestObject')) or (b.startswith('/robot/') and a.startswith('/World/TestObject'))
            if object_pair:
                actor=a if a.startswith('/robot/') else b
                vals=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
                if not Path(actor).name.startswith('gripper_') and vals and len(nonhand_contacts)<500:
                    nonhand_contacts.append(dict(actor=actor,impulse=sum(float(np.linalg.norm(v.impulse)) for v in vals),minimum_separation=min(float(v.separation) for v in vals)))
            if not (a.startswith('/robot/') and b.startswith('/robot/')):continue
            vals=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
            if vals and len(self_contacts)<500:
                self_contacts.append(dict(actor0=a,actor1=b,impulse=sum(float(np.linalg.norm(v.impulse)) for v in vals),
                    minimum_separation=min(float(v.separation) for v in vals)))
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    if args.diagnostic_no_self_collision:
        result['self_collision_enabled']=False;result['diagnostic_only']=True
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
    # Author true mimic constraints and remove follower actuators. Reference signs
    # obey PhysX q_follower + gearing*q_master + offset = 0.
    joints={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    for name,prim in joints.items():
        j=k.by_name.get(name)
        if not j:continue
        drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
        drive.CreateTypeAttr('force')
        drive.CreateMaxForceAttr(j['effort'])
        drive.CreateStiffnessAttr(100000. if 'body_joint' in name else 1000. if 'gripper' not in name else 10.0)
        drive.CreateDampingAttr(5000. if 'body_joint' in name else 100. if 'gripper' not in name else 2.0)
        m=j['mimic'];initial=qpre.get(m['joint'],0)*float(m.get('multiplier',1))+float(m.get('offset',0)) if m else qpre.get(name,0.)
        state=PhysxSchema.JointStateAPI.Apply(prim,'angular');state.GetPositionAttr().Set(float(np.degrees(initial)));state.GetVelocityAttr().Set(0.)
        drive.CreateTargetPositionAttr(float(np.degrees(initial)));drive.CreateTargetVelocityAttr(0.)
        if j['mimic']:
            m=j['mimic'];drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.);drive.CreateMaxForceAttr(0.)
            for schema in list(prim.GetAppliedSchemas()):
                if schema.startswith('PhysxMimicJointAPI:'):
                    prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,schema.split(':',1)[1])
            mimic=PhysxSchema.PhysxMimicJointAPI.Apply(prim,'rotX')
            mimic.GetReferenceJointRel().SetTargets([joints[m['joint']].GetPath()])
            mimic.GetGearingAttr().Set(-float(m.get('multiplier',1)))
            mimic.GetOffsetAttr().Set(-np.degrees(float(m.get('offset',0))))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            solver=PhysxSchema.PhysxArticulationAPI.Apply(prim)
            solver.CreateSolverPositionIterationCountAttr(64);solver.CreateSolverVelocityIterationCountAttr(16)
    result['constraint_accuracy']='articulation solver 64 position / 16 velocity iterations; original geometry/material/actuator limits retained'
    # Training isolation: lock all non-arm/non-gripper joints, and anchor root.
    frozen={}
    for name,prim in joints.items():
        j=k.by_name.get(name)
        if not j or 'arm_' in name or 'gripper' in name or 'body_joint' in name or ('head_joint' in name and plan and plan.get('q_prepare')):continue
        if j['kind'] in ('revolute','continuous'):
            value=float(qpre.get(name,0.));frozen[name]=value
            joint=UsdPhysics.RevoluteJoint(prim)
            joint.CreateLowerLimitAttr(float(np.degrees(value)))
            joint.CreateUpperLimitAttr(float(np.degrees(value)))
    rootprim=next(p for p in stage.Traverse() if p.GetName()==k.root and p.HasAPI(UsdPhysics.RigidBodyAPI))
    anchor=UsdPhysics.FixedJoint.Define(stage,'/World/TrainingBaseAnchor')
    anchor.CreateBody1Rel().SetTargets([rootprim.GetPath()])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,.04));anchor.CreateLocalPos1Attr(Gf.Vec3f(0.))
    result['frozen_joint_targets_rad']=frozen
    result['training_scope']='fixed root and non-arm joint limits; arms and grippers only; common arm/gripper initial pose; collision-audited approach'
    object_folder='strategy15_surface_assets' if args.asset in ['stool_rectangular', 'stool_wood_four_leg', 'table_side', 'basket_rectangular'] else 'object_assets'
    result['object_collision_asset']=str(ROOT/object_folder/(args.asset+'.usdc'))
    objprim=stage.DefinePrim('/World/TestObject','Xform');objprim.GetReferences().AddReference(str(ROOT/object_folder/(args.asset+'.usdc')))
    disabled_removed=0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            for schema in (UsdPhysics.CollisionAPI,UsdPhysics.MeshCollisionAPI,PhysxSchema.PhysxCollisionAPI,
                           PhysxSchema.PhysxConvexDecompositionCollisionAPI,PhysxSchema.PhysxConvexHullCollisionAPI):
                if prim.HasAPI(schema):prim.RemoveAPI(schema)
            disabled_removed+=1
    result['disabled_collision_schemas_removed']=disabled_removed
    from home_floor_material import add_wood_floor
    result['home_floor']=add_wood_floor(stage,ROOT)
    prepared=OUT/'prepared_scene.usdc'
    stage.GetRootLayer().Export(str(prepared))
    omni.usd.get_context().open_stage(str(prepared));stage=omni.usd.get_context().get_stage()
    objprim=stage.GetPrimAtPath('/World/TestObject')
    objpos=plan['object_position'] if plan else [.72,.1,.002]
    yaw=np.radians(plan.get('object_yaw_deg',0.) if plan else 0.)
    oq0=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    obj=RigidPrim('/World/TestObject',name='object',position=np.array(objpos),orientation=oq0)
    obj.set_default_state(position=np.array(objpos),orientation=oq0)
    bodyapi=PhysxSchema.PhysxRigidBodyAPI.Apply(objprim);bodyapi.CreateSolverPositionIterationCountAttr(32);bodyapi.CreateSolverVelocityIterationCountAttr(8)
    world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30)
    if result['support_height_m']>0:
        from omni.isaac.core.objects import FixedCuboid
        data=np.load(ROOT/'object_assets'/(args.asset+'.npz'));extent=np.ptp(data['vertices'],axis=0)
        world.scene.add(FixedCuboid('/World/Support',name='support',position=np.array([objpos[0],objpos[1],result['support_height_m']/2]),scale=np.array([extent[0]+.04,extent[1]+.04,result['support_height_m']]),color=np.array([.3,.3,.3])))
    robot=world.scene.add(Articulation('/robot',name='g2'));world.scene.add(obj)
    manifest=json.loads((assets/'gripper_manifest.json').read_text())
    dual=plan is not None and plan.get('mode','').startswith('bimanual_')
    paths=[r['path'] for r in manifest if dual or r['name'].startswith('gripper_l_')]
    fingers=world.scene.add(RigidPrimView(prim_paths_expr=paths,name='finger_contacts',track_contact_forces=True,
        contact_filter_prim_paths_expr=[['/World/TestObject'] for _ in paths],max_contact_count=16384))
    robot.set_default_state(position=np.array([0,0,.04]),orientation=np.array([1.,0,0,0]))
    world.reset()
    import carb
    carb.settings.get_settings().set_bool('/app/runLoops/main/rateLimitEnabled',False)
    result['execution_runtime']={'carb_task_workers':16,'tbb_workers':16,'physx_workers':16,'wall_clock_rate_limit':False,'physics_dt':1/120}
    names=robot.dof_names
    def vector(q):
        values=[]
        for n in names:
            j=k.by_name[n];m=j['mimic']
            values.append(q.get(m['joint'],0)*float(m.get('multiplier',1))+float(m.get('offset',0)) if m else q.get(n,0))
        return np.array(values,dtype=float)
    robot.set_linear_velocity(np.zeros(3));robot.set_angular_velocity(np.zeros(3))
    robot.set_joint_velocities(np.zeros(robot.num_dof))
    # Object state is never reset after physics initialization.
    kp=np.array([0. if k.by_name[n]['mimic'] else 10.0 if 'gripper' in n else 100000. if 'body_joint' in n else 1000. for n in names])
    kd=np.array([0. if k.by_name[n]['mimic'] else 2.0 if 'gripper' in n else 5000. if 'body_joint' in n else 100. for n in names])
    robot.get_articulation_controller().set_gains(kps=kp,kds=kd)
    robot._articulation_view.set_max_efforts(np.array([[0. if k.by_name[n]['mimic'] else k.by_name[n]['effort'] for n in names]]))
    camera=Camera('/World/GraspReview',resolution=(640,480));camera.initialize()
    camera.set_focal_length(24.);camera.set_horizontal_aperture(20.955)
    train_camera=camera;train_camera._frequency=-1
    from head_pilot_recorder import HeadPilotRecorder
    head_pilot=HeadPilotRecorder(OUT,robot,recorder)
    set_camera_view(eye=np.array([2.5,1.8,1.65]),target=np.array([.45,0.,.48]),camera_prim_path='/World/GraspReview');recorder.bind(train_camera)
    set_camera_view(eye=np.array([2.5,1.8,1.65]),target=np.array([.45,0.,.48]),camera_prim_path='/World/GraspReview')
    writer=cv2.VideoWriter(str(OUT/'review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
    assert writer.isOpened()
    def loadmesh(n):
        folder=assets if n.startswith('gripper_') else ROOT/'object_assets'
        d=np.load(folder/(n+'.npz'));return trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    omesh=loadmesh(args.asset)
    fmeshes={r['name']:loadmesh(r['name']) for r in manifest if r['path'] in paths}
    def bvh(m):
        shape=fcl.BVHModel();shape.beginModel(len(m.vertices),len(m.faces));shape.addSubModel(m.vertices,m.faces.astype(np.int32));shape.endModel();return shape
    oshape=bvh(omesh);fshapes={n:bvh(m) for n,m in fmeshes.items()}
    omesh_mm=omesh.copy();omesh_mm.apply_scale(1000.)
    fmeshes_mm={n:m.copy() for n,m in fmeshes.items()}
    for m in fmeshes_mm.values():m.apply_scale(1000.)
    fname=[Path(p).name for p in fingers.prim_paths]
    sourcepath={n:fmeshes[n] for n in fname}
    def mat(pos,quat):
        t=np.eye(4);t[:3,:3]=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix();t[:3,3]=pos;return t
    maxwitness=0.;witness_count=0;settled_z=None;hold_start=None;failure=None;frames=0;step=0
    active=[i for i,n in enumerate(names) if not k.by_name[n]['mimic']]
    effort_limits=np.array([0. if k.by_name[n]['mimic'] else k.by_name[n]['effort'] for n in names])
    # Compare FK against simulator actual rigid link transforms before trusting IK.
    pp,qq=fingers.get_world_poses();base=mat(*robot.get_world_pose());fk=k.fk(qpre,base)
    fk_errors={n:float(np.linalg.norm(pp[i]-fk[n][:3,3])) for i,n in enumerate(fname)}
    result['fk_initial_errors_m']=fk_errors
    if max(fk_errors.values())>.002:raise RuntimeError('FK_SIMULATOR_FRAME_MISMATCH')
    def tick(target,phase):
        nonlocal_dummy=None
        global maxwitness,witness_count,settled_z,hold_start,failure,frames,step,last_frame
        target=recorder.command(target,phase,robot,world)
        head_pilot.capture(phase,world)
        vals=vector(target)
        signed=model_gravity.torques(dict(zip(names,robot.get_joint_positions())),mat(*robot.get_world_pose()))
        gravity=np.array([signed.get(n,0.) for n in names])
        compensation=np.clip(gravity,-effort_limits,effort_limits)
        # Budget feed-forward plus feedback inside the original URDF motor limit.
        robot._articulation_view.set_max_efforts((effort_limits-np.abs(compensation))[None,:])
        robot._articulation_view.set_joint_efforts(compensation[None,:])
        feedforward_velocity=np.array([globals().get('preparation_velocity',{}).get(n,0.) for n in names]) if phase=='prepare_home' else np.zeros(len(names))
        robot.apply_action(ArticulationAction(joint_positions=vals[active],joint_velocities=feedforward_velocity[active],joint_indices=np.array(active)))
        before_clock=world.current_time
        before_index=world.current_time_step_index
        world.step(render=False)
        if step%12==0 and recorder.ticks==0:world.render()
        if world.current_time_step_index-before_index!=1 or abs(world.current_time-before_clock-1/120)>1e-6:
            raise RuntimeError('PHYSICS_CLOCK_MISMATCH')
        op,oq=obj.get_world_pose();bp,bq=robot.get_world_pose();q=robot.get_joint_positions()
        if not np.isfinite(np.r_[op,oq,bp,bq,q]).all():raise TrialRejected('NUMERICAL_INSTABILITY')
        orn=Rotation.from_quat(oq[[1,2,3,0]]);tilt=float(np.degrees(np.arccos(np.clip(orn.as_matrix()[2,2],-1,1))))
        bt=Rotation.from_quat(bq[[1,2,3,0]]).as_matrix();btilt=float(np.degrees(np.arccos(np.clip(bt[2,2],-1,1))))
        forces=np.linalg.norm(fingers.get_contact_force_matrix(dt=1/120).reshape(len(fname),-1,3).sum(axis=1),axis=1)
        if step%12==0:
            pp,qq=fingers.get_world_poses();ft=[mat(pp[i],qq[i]) for i in range(len(fname))]
            ot=mat(op,oq)
            raw=fingers.get_contact_force_data(dt=1/120)
            force,points,normals,separation,counts,starts=map(np.asarray,raw)
            counts=counts.reshape(len(fname),-1);starts=starts.reshape(len(fname),-1)
            if int(counts.sum())>=len(points) and len(points)>0:raise RuntimeError('CONTACT_BUFFER_OVERFLOW')
            contact_witnesses=[]
            for i,n in enumerate(fname):
                if forces[i]<=.5:continue
                ob=fcl.CollisionObject(oshape,fcl.Transform(ot[:3,:3],ot[:3,3]))
                fb=fcl.CollisionObject(fshapes[n],fcl.Transform(ft[i][:3,:3],ft[i][:3,3]))
                gap=max(0.,float(fcl.distance(fb,ob,fcl.DistanceRequest(),fcl.DistanceResult())))
                if gap>.0015:
                    result.setdefault('loaded_gap_events',[]).append(dict(t=step/120.,phase=phase,link=n,gap_m=gap,force_n=float(forces[i])))
                    failure=failure or 'LOADED_VISIBLE_SURFACE_GAP'
                for col in range(counts.shape[1]):
                    start=int(starts[i,col]);count=int(counts[i,col])
                    if count==0:continue
                    pts=points[start:start+count];loads=np.abs(force[start:start+count].reshape(-1))
                    pts=pts[loads>.001]
                    if not len(pts):continue
                    oi=(pts-ot[:3,3])@ot[:3,:3];fi=(pts-ft[i][:3,3])@ft[i][:3,:3]
                    contact_witnesses.append(dict(link=n,points_world=pts.tolist(),points_object=oi.tolist(),normal_force_n=loads[loads>.001].tolist()))
                    _,od,_=trimesh.proximity.closest_point(omesh_mm,oi*1000.)
                    _,fd,_=trimesh.proximity.closest_point(fmeshes_mm[n],fi*1000.)
                    od/=1000.;fd/=1000.
                    err=float(max(od.max(),fd.max()));maxwitness=max(maxwitness,err);witness_count+=len(pts)
                    if err>.0025:
                        result.setdefault('surface_error_events',[]).append(dict(t=(step+1)/120,phase=phase,link=n,object_error_m=od.tolist(),finger_error_m=fd.tolist(),points_world=pts.tolist(),object_points=oi.tolist(),finger_points=fi.tolist(),separation=separation[start:start+count].tolist()))
                        failure=failure or 'CONTACT_POINT_OFF_VISIBLE_SURFACE'
            base=mat(bp,bq);qd=dict(zip(names,q));tcp=k.fk(qd,base)['arm_l_end_link']@np.array(plan['tcp']) if plan else np.eye(4)
            relative=(np.linalg.inv(tcp)@ot)[:3,3]
            tracking=float(max(abs(q[i]-vals[i]) for i in active if 'gripper' not in names[i] and 'chassis' not in names[i]))
            rec=dict(t=(step+1)/120,phase=phase,object_position=op.tolist(),object_quaternion=oq.tolist(),
                lift_m=float(op[2]-(settled_z if settled_z is not None else objpos[2])),tilt_deg=tilt,
                base_tilt_deg=btilt,base_position=bp.tolist(),max_tracking_error_rad=tracking,
                finger_forces=dict(zip(fname,forces.tolist())),object_relative_tcp=relative.tolist(),
                max_contact_surface_error_m=maxwitness,
                joint_positions=dict(zip(names,q.tolist())),joint_targets=dict(zip(names,vals.tolist())))
            rec['gravity_torques_nm']=dict(zip(names,gravity.tolist()))
            rec['max_mimic_error_rad']=float(max([abs(qd.get(n,0.)-(qd.get(j['mimic']['joint'],0.)*float(j['mimic'].get('multiplier',1))+float(j['mimic'].get('offset',0)))) for n,j in k.by_name.items() if j['mimic']]+[0.]))
            rec['contact_witnesses']=contact_witnesses
            measured=envelope.check(qd)
            rec['torso_tilt_deg']=measured['torso_tilt_deg'];rec['max_body_joint_deg']=measured['max_body_joint_deg']
            if measured['torso_tilt_deg']>20.5 or any(qd.get(n,0)<j['lower']-.01 or qd.get(n,0)>j['upper']+.01 for n,j in k.by_name.items() if 'body_joint' in n):failure=failure or 'TORSO_ENVELOPE_EXCEEDED'
            rec['frozen_joint_max_error_rad']=float(max((abs(qd.get(n,0.)-v) for n,v in frozen.items()),default=0.))
            rec['joint_positions']={n:float(v) for n,v in qd.items()}
            if rec['frozen_joint_max_error_rad']>.005:failure=failure or 'FROZEN_JOINT_MOVED'
            if nonhand_contacts and max(x['impulse'] for x in nonhand_contacts)>.001:failure=failure or 'NONHAND_OBJECT_CONTACT'
            if len(nonhand_contacts)>=500:failure=failure or 'NONHAND_CONTACT_BUFFER_FULL'
            rec['target_joint_positions']={n:float(target.get(n,0.)) for n in names if not k.by_name[n]['mimic']}
            rec['joint_velocities']={n:float(v) for n,v in zip(names,robot.get_joint_velocities())}
            rec['physics_time_s']=float(world.current_time)
            samples.append(rec)
            if step%120==0:(OUT/'live.json').write_text(json.dumps(dict(t=rec['t'],phase=phase,max_tracking_error_rad=rec['max_tracking_error_rad'],lift_m=rec['lift_m'])))
            if phase=='settle' and (tracking>.15 or btilt>5):raise TrialRejected('INITIAL_POSE_NOT_HELD')
            if phase!='settle' and btilt>10:failure=failure or 'BASE_UNSTABLE'
            if phase!='settle' and np.linalg.norm(bp[:2])>.05:failure=failure or 'BASE_TRANSLATION_EXCEEDED'
            if phase!='settle' and tracking>.25:failure=failure or 'JOINT_TRACKING_ERROR'
            if failure:raise TrialRejected(failure)
        if step%12==0:
            rgba=camera.get_rgba()
            if rgba is not None and rgba.size:
                frame=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)
                cv2.rectangle(frame,(0,0),(1280,100),(20,20,20),-1)
                cv2.putText(frame,f'G2 VARIED-PLACEMENT COLLECTION | {args.asset} | {phase} | t={step/120:.1f}s',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(frame,f'lift={(op[2]-(settled_z if settled_z is not None else objpos[2]))*100:.1f} cm | tilt={tilt:.1f} deg | contact={forces.sum():.1f} N',(18,60),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                if failure:cv2.putText(frame,f'FAIL: {failure}',(18,86),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,80,255),1)
                writer.write(frame);frames+=1;last_frame=frame.copy()
        step+=1
    def segment(a,b,seconds,phase):
        # Respect published URDF velocity limits with quintic max derivative 1.875.
        av=vector(a);bv=vector(b)
        need=max(1.875*abs(bv[i]-av[i])/max(.01,k.by_name[n]['velocity']) for i,n in enumerate(names))
        seconds=max(seconds,need)*(1.75 if phase!='hold' else 1.)
        for i in range(round(seconds*120)):
            u=(i+1)/round(seconds*120);s=10*u**3-15*u**4+6*u**5
            q={n:float(x) for n,x in zip(names,av+(bv-av)*s)};tick(q,phase)
    squeeze = {'l': 0., 'r': 0.}
    balance = {'l': 0., 'r': 0.}
    pinch_aperture={'l':0.,'r':0.};pinch_force_time={'l':None,'r':None}
    pinch_shift={'l':0.,'r':0.};pinch_axes={}
    grasp_fk=k.fk(plan['q_grasp'],np.array(plan['base_pose']))
    for side in ('l','r'):
        centers=[(grasp_fk[f'gripper_{side}_{finger}_link4']@np.r_[fmeshes[f'gripper_{side}_{finger}_link4'].vertices.mean(0),1.])[:3] for finger in ('inner','outer')]
        axis=grasp_fk[f'arm_{side}_end_link'][:3,:3].T@(centers[0]-centers[1]);pinch_axes[side]=axis/max(1e-9,np.linalg.norm(axis))

    reference_rotation = Rotation.from_euler('z', plan.get('object_yaw_deg',0.),degrees=True).as_matrix()
    reference_closed = k.fk(plan.get('q_close',plan['q_grasp']),np.array(plan['base_pose']))
    def retarget(nominal,command,extra=0.):
        actual=dict(zip(names,robot.get_joint_positions()))
        current_base=mat(*robot.get_world_pose());desired=k.fk(nominal,np.array(plan['base_pose']))
        corrected=command.copy();tcp=np.array(plan['tcp'])
        delta=np.eye(3)  # Maintain the planned upright support planes; avoid following object tipping
        for side in (('l','r') if dual else ('l',)):
            target=desired[f'arm_{side}_end_link']@tcp
            if plan.get('mode')=='bimanual_spread':
                center=np.array(plan['candidate']['contact_centers_local'][side])
                nominal_contact=(desired[f'arm_{side}_end_link']@np.r_[center,1.])[:3]
                initial_contact=(reference_closed[f'arm_{side}_end_link']@np.r_[center,1.])[:3]
                pivot=np.array(plan['object_position'])+nominal_contact-initial_contact
                # Read object orientation only: object position is never imposed or followed.
                target[:3,3]=pivot+delta@(target[:3,3]-pivot)
                target[:3,:3]=delta@target[:3,:3]
                latest=samples[-1]['finger_forces']
                loads=[float(np.mean([s['finger_forces'].get(f'gripper_{side}_{finger}_link4',0.) for s in samples[-3:]])) for finger in ('inner','outer')]
                if max(loads)<6.:squeeze[side]=min(.012,squeeze[side]+.0003)
                elif max(loads)>35.:squeeze[side]=max(0.,squeeze[side]-.0003)
                normal=delta@np.array([0.,-1. if side=='l' else 1.,0.])
                # Roll the wrist about its support center to load the weaker finger.
                if min(loads)<2. and max(loads)>4.:
                    balance[side]=float(np.clip(balance[side]+np.radians(.08)*np.sign(loads[1]-loads[0]),-np.radians(8),np.radians(8)))
                actual_fk=k.fk(actual,current_base)
                finger_centers=[]
                for finger in ('inner','outer'):
                    n=f'gripper_{side}_{finger}_link4'
                    finger_centers.append((actual_fk[n]@np.r_[fmeshes[n].vertices.mean(0),1.])[:3])
                axis=np.cross(finger_centers[0]-finger_centers[1],normal)
                axis/=max(1e-9,np.linalg.norm(axis))
                rr=Rotation.from_rotvec(balance[side]*axis).as_matrix()
                center_world=target[:3,3]+target[:3,:3]@center
                target[:3,3]=center_world+rr@(target[:3,3]-center_world)
                target[:3,:3]=rr@target[:3,:3]
                target[:3,3]+=squeeze[side]*normal
            elif plan.get('mode')=='bimanual_top':
                loads=[float(np.mean([sum(v for n,v in sample['finger_forces'].items() if n.startswith(f'gripper_{side}_{finger}_')) for sample in samples[-3:]])) for finger in ('inner','outer')]
                now=step/120.;elapsed=.25 if pinch_force_time[side] is None else min(.5,max(.01,now-pinch_force_time[side]));pinch_force_time[side]=now
                if min(loads)<3. and max(loads)>4.:
                    pinch_shift[side]=float(np.clip(pinch_shift[side]+.0005*elapsed*np.sign(loads[0]-loads[1]),-.008,.008))
                target[:3,3]+=pinch_shift[side]*(desired[f'arm_{side}_end_link'][:3,:3]@pinch_axes[side])
                total=sum(loads)
                rate=-.04 if max(loads)>30. else float(np.clip((20.-total)*.0015,-.03,.02)) if abs(total-20.)>3. else 0.
                pinch_aperture[side]=float(np.clip(pinch_aperture[side]+rate*elapsed,-.025,0.))
                master='idx31_gripper_l_inner_joint1' if side=='l' else 'idx71_gripper_r_inner_joint1'
                corrected[master]=pinch_aperture[side]
            solution,pe,re=k.solve(side,target,actual,current_base,tcp)
            if pe>.005 or re>.05:raise TrialRejected('FIXED_TORSO_ARM_IK_UNREACHABLE')
            for n,v in solution.items():
                if f'arm_{side}_joint' in n:corrected[n]=v
        if not envelope.check(corrected)['ok']:raise TrialRejected('FEEDBACK_STATIC_LOAD_LIMIT')
        return corrected
    def four_contact(sample,threshold):
        forces=sample['finger_forces']
        return all(sum(v for n,v in forces.items() if (n==f'gripper_{side}_{finger}_link4' if plan.get('mode')=='bimanual_spread' else n.startswith(f'gripper_{side}_{finger}_')))>threshold for side in ('l','r') for finger in ('inner','outer'))
    result['pinch_force_feedback']='basket top pinch: total20N/hand target with3N deadband; maxbranch>30N opens; joint offset[-0.025,0]rad, max0.04rad/s opening and0.02rad/s closure; original effort/material/geometry/contact acceptance retained'
    result['contact_definition']='spread: four distal fingers; rim/handle pinch: four finger branches; >0.5 N each during hold'
    result['initialization_scope']='CUSTOM arms-down symmetric simulation standby -> unchanged symmetric tucked common home -> per-object approach; no reset after default initialization'
    result['preparation_profile']='seventh-order endpoint-zero velocity/acceleration/jerk; >=20s, <=40% URDF velocity, target acceleration<=0.5rad/s2, target jerk<=2rad/s3'
    result['preparation_motion']=[]
    segment(qpre,qpre,.5,'default_settle')
    result['standby_label']='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL'
    actual_standby=dict(zip(names,robot.get_joint_positions()))
    result['standby_target_q']=qpre.copy()
    result['standby_actual_q']={n:float(v) for n,v in actual_standby.items()}
    result['standby_max_tracking_error_rad']=max(abs(actual_standby.get(n,0.)-qpre.get(n,0.)) for n in qpre)
    standby_fk=k.fk(actual_standby,mat(*robot.get_world_pose()))
    result['standby_link_positions_m']={n:standby_fk[n][:3,3].tolist() for side in ('l','r') for n in [f'arm_{side}_link2',f'arm_{side}_link4',f'arm_{side}_end_link']}
    visual_vertices={p.stem:np.load(p)['vertices'] for p in (ROOT/'robot_assets').glob('*.npz') if p.stem in k.links}
    result['standby_visual_y_bounds_m']={n:[float((v@standby_fk[n][:3,:3].T+standby_fk[n][:3,3])[:,1].min()),float((v@standby_fk[n][:3,:3].T+standby_fk[n][:3,3])[:,1].max())] for n,v in visual_vertices.items()}

    saved_standby=camera.get_world_pose(camera_axes='usd')
    for label,eye in [('standby_front',[3.8,0.,1.05]),('standby_oblique',[2.5,1.8,1.65])]:
        set_camera_view(eye=np.array(eye),target=np.array([0.,0.,.95]),camera_prim_path='/World/GraspReview')
        for _ in range(12):world.render()
        rgba=camera.get_rgba()
        if rgba is not None and rgba.size:cv2.imwrite(str(OUT/(label+'.png')),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
    camera.set_world_pose(position=saved_standby[0],orientation=saved_standby[1],camera_axes='usd')
    for _ in range(12):world.render()

    previous=qpre.copy()
    for waypoint in plan['q_prepare']:
        delta=max(abs(waypoint.get(n,0.)-previous.get(n,0.)) for n in names if not k.by_name[n]['mimic'])
        duration=max(20.,max(2.1875*abs(waypoint.get(n,0.)-previous.get(n,0.))/max(.01,.4*k.by_name[n]['velocity']) for n in names if not k.by_name[n]['mimic']),np.sqrt(7.514*delta/.5),(52.5*delta/2.)**(1/3))
        duration=np.ceil(duration*120)/120;steps=round(duration*120)
        result['preparation_motion'].append(dict(duration_s=duration,start=previous,end=waypoint))
        for i in range(steps):
            u=(i+1)/steps;v=35*u**4-84*u**5+70*u**6-20*u**7
            target={n:(1-v)*previous.get(n,0.)+v*waypoint.get(n,0.) for n in set(previous)|set(waypoint)}
            dv=(140*u**3-420*u**4+420*u**5-140*u**6)/duration
            preparation_velocity={n:dv*(waypoint.get(n,0.)-previous.get(n,0.)) for n in target}
            tick(target,'prepare_home')
        previous=waypoint.copy()
    qpre=plan['q_start'].copy()
    segment(qpre,qpre,.5,'initial_settle')
    actual=dict(zip(names,robot.get_joint_positions()))
    result['prepared_max_tracking_error_rad']=max(abs(actual.get(n,0.)-qpre.get(n,0.)) for n in qpre)
    if result['prepared_max_tracking_error_rad']>.03:raise TrialRejected('PREPARATION_NOT_SETTLED')
    live_stage=omni.usd.get_context().get_stage()
    live_joints={prim.GetName():prim for prim in live_stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
    for name,prim in live_joints.items():
        if 'body_joint' not in name and 'head_joint' not in name:continue
        value=qpre[name];frozen[name]=value
        joint=UsdPhysics.RevoluteJoint(prim);joint.CreateLowerLimitAttr(float(np.degrees(value)));joint.CreateUpperLimitAttr(float(np.degrees(value)))
    result['frozen_joint_targets_rad']=frozen.copy()
    # Capture the actual new home without stepping physics or changing the robot.
    saved=camera.get_world_pose(camera_axes='usd')
    camera.set_world_pose(position=np.array([.15,0.,3.2]),orientation=np.array([np.sqrt(.5),0.,0.,-np.sqrt(.5)]),camera_axes='usd')
    for _ in range(12):world.render()
    rgba=camera.get_rgba()
    if rgba is not None and rgba.size:cv2.imwrite(str(OUT/'initial_topview.png'),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
    camera.set_world_pose(position=saved[0],orientation=saved[1],camera_axes='usd')
    for _ in range(12):world.render()
    segment(qpre,qpre,.5,'settle')
    settled_z=float(obj.get_world_pose()[0][2])
    if plan and 'q_start' in plan and not args.pose_only:
        last=qpre
        for waypoint in plan['q_approach']:
            segment(last,waypoint,6.,'common_approach');last=waypoint
        segment(q_grasp_pre,q_grasp_pre,.75,'pregrasp_hold')
        if failure:raise TrialRejected(failure)
        qpre=q_grasp_pre
    if args.pose_only:
        segment(qpre,qpre,5.,'pose_hold')
    elif plan:
        if samples[-1]['max_tracking_error_rad']>.25:failure=failure or 'INITIAL_POSE_NOT_HELD'
        if failure is None:
            segment(qpre,plan['q_grasp'],3.,'approach')
        closed=plan.get('q_close',{**plan['q_grasp'],'idx31_gripper_l_inner_joint1':0.})
        if failure is None:segment(plan['q_grasp'],closed,2.,'close')
        nominal_closed=closed.copy()
        grip_extra=0.
        stable=0
        for refine in range(96):
            fixed=retarget(nominal_closed,closed)
            segment(closed,fixed,.25,'contact_refine');closed=fixed
            forces=samples[-1]['finger_forces']
            if plan.get('mode')=='bimanual_spread':
                achieved=all(forces.get(f'gripper_{side}_{finger}_link4',0.)>1. for side in ('l','r') for finger in ('inner','outer'))
            else:
                achieved=four_contact(samples[-1],1.)
            stable=stable+1 if achieved else 0
            if stable>=3:break
        result['preload_stable_updates']=stable
        result['squeeze_m']=squeeze.copy();result['balance_deg']={s:float(np.degrees(v)) for s,v in balance.items()}
        if stable<3:raise TrialRejected('FOUR_FINGER_PRELOAD_NOT_ESTABLISHED')
        result['adaptive_grip_extra_m']=grip_extra
        latest=samples[-1]['finger_forces']
        side_a=sum(v for n,v in latest.items() if ('gripper_l_' if dual else 'inner') in n)
        side_b=sum(v for n,v in latest.items() if ('gripper_r_' if dual else 'outer') in n)
        if min(side_a,side_b)<=.5:raise TrialRejected('NO_BILATERAL_CONTACT_AFTER_CLOSE')
        if failure is None:
            last=closed
            previous_nominal=nominal_closed.copy()
            for q in plan['q_lift']:
                q={**q,'idx31_gripper_l_inner_joint1':plan.get('hold_gripper_q',0.),'idx71_gripper_r_inner_joint1':plan.get('hold_gripper_q',0.)}
                for fraction in np.linspace(.2,1.,5):
                    intermediate={n:(1-fraction)*previous_nominal.get(n,0.)+fraction*q.get(n,0.) for n in set(previous_nominal)|set(q)}
                    corrected=retarget(intermediate,last,grip_extra if dual else 0.)
                    segment(last,corrected,.2,'lift');last=corrected
                previous_nominal=q.copy()
                if failure:break
            if failure is None:
                settled_updates=0
                for _ in range(60):
                    corrected=retarget(q,last)
                    segment(last,corrected,.2,'contact_settle');last=corrected
                    recent=samples[-1];previous=samples[-2]
                    velocity=np.linalg.norm(np.array(recent['object_position'])-previous['object_position'])/max(.001,recent['t']-previous['t'])
                    forces=recent['finger_forces']
                    ready=four_contact(recent,1.) and velocity<.015 and recent['lift_m']>=.15
                    settled_updates=settled_updates+1 if ready else 0
                    if settled_updates>=4:break
                result['settled_contact_updates']=settled_updates
                if settled_updates<4:raise TrialRejected('NO_SETTLED_FOUR_FINGER_GRASP')
                segment(last,last,3.,'hold')
    else:
        failure='NO_COLLISION_FREE_REACHABLE_SURFACE_GRASP'
        segment(qpre,qpre,3.,'planning_rejected')
    result['collection_timing_profile']='shorter static settles only; original free-space/contact/lift durations and original velocity bound retained'
    result['collection_initialization']=plan.get('collection_initialization','full_standby_preparation')
    result['max_frozen_joint_error_rad']=max((s['frozen_joint_max_error_rad'] for s in samples),default=None)
    if failure:
        current=dict(zip(names,robot.get_joint_positions()));segment(current,current,1.,'rejected')
    hold=[s for s in samples if s['phase']=='hold']
    liftmax=max((s['lift_m'] for s in samples),default=0.)
    inner=lambda s:sum(v for n,v in s['finger_forces'].items() if ('gripper_l_' if dual else 'inner') in n)
    outer=lambda s:sum(v for n,v in s['finger_forces'].items() if ('gripper_r_' if dual else 'outer') in n)
    drift=max((np.linalg.norm(np.array(s['object_relative_tcp'])-hold[0]['object_relative_tcp']) for s in hold),default=999.)
    bilateral=sum(inner(s)>.5 and outer(s)>.5 for s in hold)/max(1,len(hold))
    four=lambda s:four_contact(s,.5)
    four_fraction=sum(four(s) for s in hold)/max(1,len(hold))
    result['hold_four_finger_contact_fraction']=four_fraction
    passed=bool(four_fraction>=.9 and failure is None and len(hold)>=29 and min(s['lift_m'] for s in hold)>=.15 and max(s['tilt_deg'] for s in hold)<=20 and drift<=.02 and bilateral>=.9 and witness_count>0)
    if args.pose_only:
        held=[s for s in samples if s['phase']=='pose_hold']
        passed=bool(failure is None and len(held)>=49 and max(s['max_tracking_error_rad'] for s in held)<.08 and max(s['base_tilt_deg'] for s in held)<2.)
        result['pose_hold_max_tracking_rad']=max((s['max_tracking_error_rad'] for s in held),default=None)
        result['pose_hold_max_torso_tilt_deg']=max((s['torso_tilt_deg'] for s in held),default=None)
    result.update(success=passed,failure=failure or (None if passed else 'LIFT_HOLD_CRITERIA_NOT_MET'),
        max_lift_m=liftmax,min_hold_lift_m=min((s['lift_m'] for s in hold),default=None),
        max_hold_tilt_deg=max((s['tilt_deg'] for s in hold),default=None),hold_relative_drift_m=float(drift) if hold else None,
        hold_bilateral_contact_fraction=bilateral,contact_witness_points=witness_count,max_contact_surface_error_m=maxwitness,
        frames=frames,simulated_seconds=step/120,plan=plan)
except TrialRejected as error:
    result.update(failure=str(error),max_lift_m=max((s['lift_m'] for s in samples),default=0.),
        simulated_seconds=step/120,frames=frames,contact_witness_points=globals().get('witness_count',0),
        max_contact_surface_error_m=globals().get('maxwitness',0.),
        grasp_attempted=any(s['phase'] in ('approach','close','lift','hold') for s in samples),
        plan=plan)
    if writer is not None and last_frame is not None:
        frozen=last_frame.copy();cv2.rectangle(frozen,(0,70),(1280,105),(20,20,20),-1)
        cv2.putText(frozen,'STOPPED: '+str(error)+' (last valid frame)',(18,93),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,80,255),1)
        for _ in range(60):writer.write(frozen)
        result['frozen_tail_frames']=60
except Exception:
    result['failure']='HARNESS_ERROR';result['error']=traceback.format_exc()
finally:
    if writer is not None:writer.release()
    (OUT/'samples.json').write_text(json.dumps(samples,indent=2))
    (OUT/'self_contacts.json').write_text(json.dumps(self_contacts,indent=2))
    (OUT/'nonhand_contacts.json').write_text(json.dumps(nonhand_contacts,indent=2))
    recorder.finish(result)
    if head_pilot is not None:head_pilot.finish(result)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print('G2_TRIAL_RESULT '+json.dumps({k:v for k,v in result.items() if k!='plan'}),flush=True)
    app.close()

