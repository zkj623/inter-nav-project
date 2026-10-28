"""Physical contact calibration only; never counted as a navigation task.

Use the original Go2 asset/policy, actuator and a dynamic rectangular carton.
The fixture deliberately removes online planning to isolate contact control.
"""
import argparse,json,math,shutil,hashlib
from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.configs.objects import FixedCubeCfg,DynamicCubeCfg

p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',default='explicit_dc',choices=['explicit_dc','implicit_pd'])
p.add_argument('--suite',default='comparison',choices=['comparison','perturbed','forward_only','contact_latched','forward_steer'])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,a.output/'source.py')
(a.output/'source_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
cases=[dict(name='current_abort18',start=.30,ramp=.15,guard=18.,yaw_gain=3.),
       dict(name='current_observe30',start=.30,ramp=.15,guard=30.,yaw_gain=3.),
       dict(name='slow_ramp_observe30',start=.30,ramp=.05625,guard=30.,yaw_gain=1.8),
       dict(name='steady040_observe30',start=.40,ramp=0.,guard=30.,yaw_gain=1.8),
       dict(name='steady050_observe30',start=.50,ramp=0.,guard=30.,yaw_gain=1.8)]
if a.suite=='perturbed':
    cases=[dict(name=name,start=.30,ramp=.05625,guard=18.,yaw_gain=1.8,initial_heading=heading,initial_lateral=lateral)
           for name,heading,lateral in [('nominal',0.,0.),('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
if a.suite=='forward_only':
    cases=[dict(name=name,start=.30,ramp=.05625,guard=18.,yaw_gain=0.,body_forward=True,initial_heading=heading,initial_lateral=lateral)
           for name,heading,lateral in [('nominal',0.,0.),('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
if a.suite=='contact_latched':
    cases=[dict(name=name,start=speed,ramp=.05625,guard=18.,yaw_gain=1.8,forward_until_contact=True,initial_heading=heading,initial_lateral=lateral)
           for name,speed,heading,lateral in [('nominal030',.30,0.,0.),('nominal040',.40,0.,0.),('yaw_plus',.40,.05,0.),('yaw_minus',.40,-.05,0.),('lateral_plus',.40,0.,.03),('lateral_minus',.40,0.,-.03)]]
if a.suite=='forward_steer':
    cases=[dict(name=name,start=.30,ramp=.05625,guard=18.,yaw_gain=1.8,body_forward_steer=True,initial_heading=heading,initial_lateral=lateral)
           for name,heading,lateral in [('nominal',0.,0.),('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
profile=programmatic_go2_profile(environment_length=12.,environment_width=8.,start=(0.,0.,.41),goal=(3.,0.,.41))
run=runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,actuator_mode=a.mode)
objects=[FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(12.,8.,.01),visible=True,color=(.4,.4,.4)),
         DynamicCubeCfg(name='probe_carton',prim_path='/World/env_0/objects/probe_carton',position=(1.155,0.,.485),scale=(.85,.55,.95),color=(.65,.45,.25))]
cfg=runner.build_go2_navigation_config(profile,run,objects=objects)
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39
cfg.task_config.episodes[0].robots[0].sensors=[]
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu)
env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'))
    env=runner.Env(runtime);obs,_=env.reset()
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.materials import PhysicsMaterial
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdShade,UsdPhysics
    robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()))
    ctrl=robot._go2_locomotion_control;art=robot.isaac_robot
    box=RigidPrim('/World/env_0/objects/probe_carton',name='contact_calibration_box');box.initialize();box.set_mass(6.)
    material=PhysicsMaterial('/World/CalibrationCardboard',static_friction=.6,dynamic_friction=.5,restitution=0.)
    prim=get_current_stage().GetPrimAtPath('/World/env_0/objects/probe_carton')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(material.prim),materialPurpose='physics')
    runner._apply_high_friction_material('/World/env_0/objects/grscene_go2_floor')
    (a.output/'contract.json').write_text(json.dumps({'diagnostic_only':True,'navigation_task':False,
        'mode':a.mode,'physics_dt':.005,'policy_hz':50,'floor_top_m':.01,
        'box_dimensions_m':[.85,.55,.95],'box_mass_kg':float(box.get_mass()),
        'box_static_friction':.6,'box_dynamic_friction':.5,'robot_policy':ctrl.policy_name,
        'removed_sensors_for_control_only_diagnostic':True,'cases':cases},indent=2))
    reports=[]
    for case in cases:
        root,_=art.get_world_pose();base,_=robot.get_world_pose()
        heading0=case.get('initial_heading',0.);lateral0=case.get('initial_lateral',0.)
        art.set_world_pose(position=root+np.array([0.,lateral0,.41])-base,
                           orientation=np.array([math.cos(heading0/2),0.,0.,math.sin(heading0/2)]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        box.set_world_pose(position=np.array([1.155,0.,.485]),orientation=np.array([1.,0.,0.,0.]))
        box.set_linear_velocity(np.zeros(3));box.set_angular_velocity(np.zeros(3))
        limiter=VelocitySlewLimiter();rows=[];engage_time=None;end_reason=None;stop_step=None;body_contact=False
        with (a.output/(case['name']+'.jsonl')).open('w',buffering=1) as trace:
            for step in range(4640):
                now=float(ctrl._simulation_time());pos,q=robot.get_world_pose();q=np.asarray(q);q=q/np.linalg.norm(q)
                tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
                heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
                bp,_=box.get_world_pose();gap=float(bp[0]-pos[0]-.33*math.cos(heading)-.425)
                if step>=640 and end_reason is None:
                    if tilt>=case['guard']:end_reason='attitude_guard'
                    elif abs(heading)>.18:end_reason='heading_drift'
                    elif abs(bp[1]-pos[1]-.33*math.sin(heading))>.075:end_reason='contact_offset'
                    elif abs(pos[0])>4. or abs(pos[1])>3.:end_reason='fixture_boundary'
                    elif pos[2]<.21:end_reason='height_guard'
                    elif bp[0]-1.155>=.85:end_reason='parked_distance'
                    elif abs(bp[1])>.15:end_reason='box_cross_track'
                    if end_reason is not None:stop_step=step
                if step<640 or end_reason is not None:request=np.zeros(3)
                else:
                    if gap<=.19 and engage_time is None:engage_time=now
                    elapsed=0. if engage_time is None else now-engage_time
                    speed=min(.95,case['start']+case['ramp']*elapsed)
                    c,s=math.cos(heading),math.sin(heading);rot=np.array([[c,-s],[s,c]])
                    world_v=rot@ctrl.body_lin_vel[:2]
                    lateral=bp[1]-(pos[1]+.33*s)
                    vy=float(np.clip(3.*lateral-.6*world_v[1],-.20,.20))
                    yaw=float(np.clip(-case['yaw_gain']*heading-(.35*ctrl.body_ang_vel[2] if case['yaw_gain']==3. else 0.),-.55,.55))
                    if gap<=.02:body_contact=True
                    only_forward=case.get('body_forward',False) or (case.get('forward_until_contact',False) and not body_contact)
                    request=np.array([speed,0.,0.]) if only_forward else np.array([*(rot.T@np.array([speed,vy])),yaw])
                    if case.get('body_forward_steer',False):request=np.array([speed,0.,yaw])
                command=(np.zeros(3) if end_reason=='attitude_guard' else limiter.preview(request,now))
                limiter.accept(command,now);obs,*_=env.step({'move_by_speed':command.tolist()})
                q=np.asarray(obs['orientation']);q=q/np.linalg.norm(q);bp,_=box.get_world_pose()
                row={'step':step,'time_s':float(ctrl._simulation_time()),'command':command.tolist(),
                    'position':list(map(float,obs['position'])),'orientation':q.tolist(),
                    'velocity':ctrl.body_lin_vel.tolist(),'angular':ctrl.body_ang_vel.tolist(),
                    'tilt':math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))),
                    'box_position':list(map(float,bp)),'gap_before_step':gap,'end_reason':end_reason}
                rows.append(row)
                if step%4==0:trace.write(json.dumps(row)+'\n')
                if stop_step is not None and step-stop_step>=400:break
        active=rows[640:];report={**case,'stop_reason':end_reason or 'time_limit','physical_steps':len(rows),
            'maximum_tilt':max(x['tilt'] for x in active),'healthy':all(x['tilt']<30 and x['position'][2]>.21 for x in active),
            'box_shift_m':float(rows[-1]['box_position'][0]-1.155),'box_cross_m':float(rows[-1]['box_position'][1]),
            'minimum_height_m':min(x['position'][2] for x in active)}
        reports.append(report);(a.output/'results.json').write_text(json.dumps({'diagnostic_only':True,'complete':False,'cases':reports},indent=2))
        print(json.dumps(report),flush=True)
    (a.output/'results.json').write_text(json.dumps({'diagnostic_only':True,'complete':True,'cases':reports},indent=2))
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
