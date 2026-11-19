from pathlib import Path
import re,ast
R=Path(__file__).resolve().parent
for label,src in [('side','grasp_trial_strategy15_fixedclock.py'),('deadband','grasp_trial_strategy15_deadbandclock.py'),('support','grasp_trial_strategy15_supportbalance.py')]:
 s=(R/src).read_text()
 s=re.sub(r"p.add_argument\('--plan-dir'.*?\);", "p.add_argument('--plan-dir',default='fixedpose_0_validated_plans');",s,count=1)
 s=re.sub(r"OUT=ROOT/[^\n]+",f"OUT=ROOT/'fixedpose_{label}_trials'/args.plan_dir/args.asset/f'candidate{{args.candidate}}_repeat{{args.repeat}}'",s,count=1)
 s=s.replace('robot_fixed_to_world=False','robot_fixed_to_world=True')
 needle="    objprim=stage.DefinePrim('/World/TestObject','Xform')"
 insertion="""    # Training isolation: lock all non-arm/non-gripper joints, and anchor root.
    frozen={}
    for name,prim in joints.items():
        j=k.by_name.get(name)
        if not j or 'arm_' in name or 'gripper' in name:continue
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
    result['training_scope']='fixed root and non-arm joint limits; arms and grippers only; prepose initialization'
"""
 assert needle in s;s=s.replace(needle,insertion+needle)
 # Set object rotation BEFORE reset, to avoid stale collision initialization.
 needle="    obj=RigidPrim('/World/TestObject',name='object',position=np.array(objpos))"
 assert needle in s
 s=s.replace(needle,"    yaw=np.radians(plan.get('object_yaw_deg',0.) if plan else 0.)\n    oq0=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])\n    obj=RigidPrim('/World/TestObject',name='object',position=np.array(objpos),orientation=oq0)\n    obj.set_default_state(position=np.array(objpos),orientation=oq0)")
 needle="    yaw=np.radians(plan.get('object_yaw_deg',0.) if plan else 0.);obj.set_world_pose(np.array(objpos),np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]));obj.set_linear_velocity(np.zeros(3));obj.set_angular_velocity(np.zeros(3))"
 assert needle in s;s=s.replace(needle,'    # Object state is never reset after physics initialization.')
 s=s.replace('            samples.append(rec)',"            rec['frozen_joint_max_error_rad']=float(max((abs(qd.get(n,0.)-v) for n,v in frozen.items()),default=0.))\n            rec['joint_positions']=qd\n            if rec['frozen_joint_max_error_rad']>.005:failure=failure or 'FROZEN_JOINT_MOVED'\n            samples.append(rec)")
 s=s.replace("    if failure:\n        current=dict", "    result['max_frozen_joint_error_rad']=max((s['frozen_joint_max_error_rad'] for s in samples),default=None)\n    if failure:\n        current=dict")
 ast.parse(s);(R/f'fixedpose_{label}_trial.py').write_text(s)
 print('BUILT',label)
