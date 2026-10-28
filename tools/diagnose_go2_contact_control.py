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
p.add_argument('--suite',default='comparison',choices=['comparison','perturbed','forward_only','contact_latched','forward_steer','contact_guard','approach_caps','contact_frequency','native_contract','skill_contact'])
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
if a.suite=='contact_guard':
    cases=[dict(name=name+'_guard'+str(int(guard)),start=.30,ramp=.15,guard=guard,yaw_gain=3.,initial_heading=heading,initial_lateral=lateral)
           for guard in [18.,30.] for name,heading,lateral in [('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
if a.suite=='approach_caps':
    cases=[dict(name=name+'_cap'+str(limit),start=.30,ramp=.15,guard=18.,yaw_gain=3.,initial_heading=heading,initial_lateral=lateral,engage_lateral_limit=limit,engage_yaw_limit=.15)
           for limit in [.05,.03] for name,heading,lateral in [('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
if a.suite=='contact_frequency':
    cases=[dict(name=name+'_hz'+str(200//decimation),start=.30,ramp=.15,guard=18.,yaw_gain=3.,initial_heading=heading,initial_lateral=lateral,policy_decimation=decimation)
           for decimation in [8,5] for name,heading,lateral in [('nominal',0.,0.),('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
if a.suite in ['native_contract','skill_contact']:
    cases=[dict(name=name,start=.30,ramp=.15,guard=18.,yaw_gain=3.,initial_heading=heading,initial_lateral=lateral)
           for name,heading,lateral in [('nominal',0.,0.),('yaw_plus',.05,0.),('yaw_minus',-.05,0.),('lateral_plus',0.,.03),('lateral_minus',0.,-.03)]]
if a.suite=='skill_contact':
    from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideConfig
    cases=[{**case,'start':PushAsideConfig().push_start_speed,'ramp':PushAsideConfig().push_acceleration} for case in cases]
profile=programmatic_go2_profile(environment_length=12.,environment_width=8.,start=(0.,0.,.41),goal=(3.,0.,.41))
run=runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,actuator_mode=a.mode)
objects=[FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(12.,8.,.01),visible=True,color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.)]
cfg=runner.build_go2_navigation_config(profile,run,objects=objects)
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39
cfg.task_config.episodes[0].robots[0].sensors=[]
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu)
env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'))
    env=runner.Env(runtime)
    from omni.isaac.core.utils.stage import get_current_stage
    from grutopia_extension.interactive_navigation.interactive_box_scenarios import author_box
    author_box(get_current_stage(), '/World/ContactProbeBox',
        dict(id='Probe',position=[1.155,0.,.485],yaw_degrees=0.,mass_kg=6.,
             dimensions=[.85,.55,.95],tape=[.8,.7,.3],interactions=['push']),
        material_root='/World/ContactProbeMaterials')
    obs,_=env.reset()
    import grutopia_extension.controllers.isaac_go2_ctrl as locomotion
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.materials import PhysicsMaterial
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdShade,UsdPhysics
    robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()))
    ctrl=robot._go2_locomotion_control;art=robot.isaac_robot
    box=RigidPrim('/World/ContactProbeBox',name='contact_calibration_box');box.initialize()
    runner._apply_high_friction_material('/World/env_0/objects/grscene_go2_floor')
    (a.output/'contract.json').write_text(json.dumps({'diagnostic_only':True,'navigation_task':False,
        'mode':a.mode,'physics_dt':.005,'policy_hz':'case_specific' if a.suite=='contact_frequency' else 50,'floor_top_m':.01,
        'box_dimensions_m':[.85,.55,.95],'box_mass_kg':float(box.get_mass()),
        'box_authoring':'interactive_box_scenarios.author_box_before_reset',
        'box_inertia':box._rigid_prim_view._physics_view.get_inertias().tolist(),
        'box_static_friction':.6,'box_dynamic_friction':.5,'robot_policy':ctrl.policy_name,
        'native_velocity_limits':art._articulation_view._physics_view.get_dof_max_velocities().tolist(),
        'native_max_forces':art._articulation_view._physics_view.get_dof_max_forces().tolist(),
        'removed_sensors_for_control_only_diagnostic':True,'cases':cases},indent=2))
    from types import SimpleNamespace
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
    from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
    from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
    from grutopia_extension.interactive_navigation.go2_interaction_motion import checked_skill_brake
    reports=[]
    for case in cases:
        locomotion.POLICY_DECIMATION=case.get('policy_decimation',4)
        root,_=art.get_world_pose();base,_=robot.get_world_pose()
        heading0=case.get('initial_heading',0.);lateral0=case.get('initial_lateral',0.)
        art.set_world_pose(position=root+np.array([0.,lateral0,.41])-base,
                           orientation=np.array([math.cos(heading0/2),0.,0.,math.sin(heading0/2)]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        box.set_world_pose(position=np.array([1.155,0.,.485]),orientation=np.array([1.,0.,0.,0.]))
        box.set_linear_velocity(np.zeros(3));box.set_angular_velocity(np.zeros(3))
        limiter=VelocitySlewLimiter();rows=[];engage_time=None;end_reason=None;stop_step=None;body_contact=False;feedback_contact=False
        skill=None;skill_state=None
        with (a.output/(case['name']+'.jsonl')).open('w',buffering=1) as trace:
            for step in range(4640):
                now=float(ctrl._simulation_time());pos,q=robot.get_world_pose();q=np.asarray(q);q=q/np.linalg.norm(q)
                tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
                heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
                bp,_=box.get_world_pose();gap=float(bp[0]-pos[0]-.33*math.cos(heading)-.425)
                if step>=640 and end_reason is None and a.suite!='skill_contact':
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
                    if gap<=.03 or bp[0]-1.155>.015:feedback_contact=True
                    if 'engage_lateral_limit' in case and not feedback_contact:
                        vy=float(np.clip(vy,-case['engage_lateral_limit'],case['engage_lateral_limit']))
                        yaw=float(np.clip(yaw,-case['engage_yaw_limit'],case['engage_yaw_limit']))
                    only_forward=case.get('body_forward',False) or (case.get('forward_until_contact',False) and not body_contact)
                    request=np.array([speed,0.,0.]) if only_forward else np.array([*(rot.T@np.array([speed,vy])),yaw])
                    if case.get('body_forward_steer',False):request=np.array([speed,0.,yaw])
                command=(np.zeros(3) if end_reason=='attitude_guard' else limiter.preview(request,now))
                if a.suite=='skill_contact' and step>=640 and end_reason is None:
                    if skill is None:
                        # Known empty local fixture ONLY, never used by online
                        # navigation. Isolate contact/stop after initialization.
                        planner=SidePushPlanner(np.ones((160,240),dtype=bool),[(-6.,6.),(-4.,4.)],resolution=.05)
                        planner.full_navigation_footprint=True
                        planner.active_passages['Probe']={'center':[1.155,0.],'axis':[0.,1.]}
                        permission=InteractionPermission('Probe','box','push_aside',(),(),now,
                            source='ground_truth_layout',source_reference=str(a.output/'source.py'),
                            source_digest=hashlib.sha256((a.output/'source.py').read_bytes()).hexdigest())
                        plan=SimpleNamespace(box_id='Probe',direction=(1.,0.),push_distance=.85)
                        skill=PushAsideSkill(permission,plan,(0.,0.),(3.,0.))
                        skill.state='engage';skill.engage_box_position=np.asarray(bp[:2]).copy()
                    tracked=BoxState('Probe',tuple(bp[:2]),(.85,.55,.95),mass_kg=6.)
                    context=SkillContext(now,np.asarray(pos),heading,ctrl.body_lin_vel,
                        float(ctrl.body_ang_vel[2]),tilt,tilt<30. and pos[2]>.21,
                        tracked,[tracked],planner,None,limiter,body_angular_velocity=ctrl.body_ang_vel)
                    answer=skill.step(context);skill_state=answer.state
                    command=checked_skill_brake(answer,context)
                    if answer.state=='retreat' and skill.release_reason=='side_parked':
                        end_reason='parked_settled';stop_step=step
                    elif answer.status!='running' or answer.state in ['brake_for_recovery','prepare']:
                        end_reason='skill_'+answer.state+':'+(answer.reason or str(answer.events));stop_step=step
                limiter.accept(command,now);obs,*_=env.step({'move_by_speed':command.tolist()})
                limiter.observe([*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],float(ctrl._simulation_time()))
                q=np.asarray(obs['orientation']);q=q/np.linalg.norm(q);bp,_=box.get_world_pose()
                row={'step':step,'time_s':float(ctrl._simulation_time()),'command':command.tolist(),
                    'position':list(map(float,obs['position'])),'orientation':q.tolist(),
                    'velocity':ctrl.body_lin_vel.tolist(),'angular':ctrl.body_ang_vel.tolist(),
                    'tilt':math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))),
                    'box_position':list(map(float,bp)),'gap_before_step':gap,'end_reason':end_reason,
                    'skill_state':skill_state,'policy_inferences':ctrl.policy_inferences,'last_inference_time':ctrl._last_inference_time}
                rows.append(row)
                if step%4==0:trace.write(json.dumps(row)+'\n')
                if stop_step is not None and step-stop_step>=400:break
        active=rows[640:];report={**case,'stop_reason':end_reason or 'time_limit','physical_steps':len(rows),
            'maximum_tilt':max(x['tilt'] for x in active),'healthy':all(x['tilt']<30 and x['position'][2]>.21 for x in active),
            'box_shift_m':float(rows[-1]['box_position'][0]-1.155),'box_cross_m':float(rows[-1]['box_position'][1]),
            'minimum_height_m':min(x['position'][2] for x in active)}
        report['contact_phase_accepted']=bool(a.suite=='skill_contact' and end_reason=='parked_settled' and report['healthy'])
        report['skill_report']=None if skill is None else skill.report()
        reports.append(report);(a.output/'results.json').write_text(json.dumps({'diagnostic_only':True,'complete':False,'cases':reports},indent=2))
        print(json.dumps(report),flush=True)
    (a.output/'results.json').write_text(json.dumps({'diagnostic_only':True,'complete':True,'cases':reports},indent=2))
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
