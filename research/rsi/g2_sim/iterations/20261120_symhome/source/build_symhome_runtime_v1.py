from pathlib import Path
import json,re,ast
R=Path(__file__).resolve().parent;cfg=json.loads((R/'ground_lift_training_config.json').read_text())
for a,b in cfg['validated_simulation_baselines'].items():
 s=(R/b['harness']).read_text();s=re.sub(r"OUT=ROOT/'[^']+'", "OUT=ROOT/'symhome_trials'",s)
 old="    qpre=plan.get('q_start',qpre) if plan else qpre";assert old in s;s=s.replace(old,"    qpre=plan.get('q_default',plan.get('q_start',qpre)) if plan else qpre")
 old="        if not j or 'arm_' in name or 'gripper' in name:continue";assert old in s;s=s.replace(old,"        if not j or 'arm_' in name or 'gripper' in name or 'body_joint' in name:continue")
 s=s.replace('            samples.append(rec)',"            rec['target_joint_positions']={n:float(target.get(n,0.)) for n in names if not k.by_name[n]['mimic']}\n            rec['joint_velocities']={n:float(v) for n,v in zip(names,robot.get_joint_velocities())}\n            samples.append(rec)")
 # Insert before the original home settling: move torso through original limits, then lock it at the low work posture.
 needle="    segment(qpre,qpre,2.,'settle')";assert needle in s
 prep="""    result['initialization_scope']='official flat G2 op scene defaults -> symmetric tucked common home -> per-object approach; no reset after default initialization'
    result['preparation_profile']='seventh-order endpoint-zero velocity/acceleration/jerk; >=12s, <=40% URDF velocity, target acceleration<=0.5rad/s2, target jerk<=2rad/s3'
    result['preparation_motion']=[]
    segment(qpre,qpre,2.,'default_settle')
    previous=qpre.copy()
    for waypoint in plan['q_prepare']:
        delta=max(abs(waypoint.get(n,0.)-previous.get(n,0.)) for n in names if not k.by_name[n]['mimic'])
        duration=max(12.,max(2.1875*abs(waypoint.get(n,0.)-previous.get(n,0.))/max(.01,.4*k.by_name[n]['velocity']) for n in names if not k.by_name[n]['mimic']),np.sqrt(7.514*delta/.5),(52.5*delta/2.)**(1/3))
        duration=np.ceil(duration*120)/120;steps=round(duration*120)
        result['preparation_motion'].append(dict(duration_s=duration,start=previous,end=waypoint))
        for i in range(steps):
            u=(i+1)/steps;v=35*u**4-84*u**5+70*u**6-20*u**7
            target={n:(1-v)*previous.get(n,0.)+v*waypoint.get(n,0.) for n in set(previous)|set(waypoint)}
            tick(target,'prepare_home')
        previous=waypoint.copy()
    qpre=plan['q_start'].copy()
    segment(qpre,qpre,2.,'initial_settle')
    actual=dict(zip(names,robot.get_joint_positions()))
    result['prepared_max_tracking_error_rad']=max(abs(actual.get(n,0.)-qpre.get(n,0.)) for n in qpre)
    if result['prepared_max_tracking_error_rad']>.03:raise TrialRejected('PREPARATION_NOT_SETTLED')
    for name,prim in joints.items():
        if 'body_joint' not in name:continue
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
"""
 s=s.replace(needle,prep+needle)
 ast.parse(s);(R/f'symhome_{a}_trial.py').write_text(s)
print('BUILT',len(cfg['validated_assets']))
