"""Flat-floor stop-response diagnosis; never a substitute for MV7 navigation."""
import argparse,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_stop_feedback import ForwardStopFeedback
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.configs.objects import FixedCubeCfg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--mode',choices=['passive','forward_feedback'],default='passive')
p.add_argument('--replay-prefix',type=Path,help='Optional diagnostic-only NPZ command prefix')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.output/'source.py')
commands=None;initial_heading=0.;prefix_hash=None
if a.replay_prefix:
    with np.load(a.replay_prefix,allow_pickle=False) as data:
        commands=data['commands'];initial_heading=float(data['heading'])
    prefix_hash=hashlib.sha256(a.replay_prefix.read_bytes()).hexdigest()
    shutil.copy2(a.replay_prefix,a.output/'replay_prefix.npz')
    if commands.ndim!=2 or commands.shape[1]!=3 or not np.isfinite(commands).all():raise ValueError('Invalid command prefix')
cases=[('recorded_arrival',None,len(commands))] if commands is not None else [
    ('forward',np.array([.5,0.,0.]),640),('lateral',np.array([0.,.35,0.]),400),('yaw',np.array([0.,0.,1.]),400)]
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=(0.,0.,.41),goal=(3.,0.,.41))
cfg=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,actuator_mode='explicit_dc'),
    objects=[FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(16.,12.,.01),
        color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.)])
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39;cfg.task_config.episodes[0].robots[0].sensors=[]
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu);env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'));env=runner.Env(runtime);obs,_=env.reset()
    robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()));art=robot.isaac_robot;ctrl=robot._go2_locomotion_control
    contract={'diagnostic_only':True,'navigation_task':False,'policy':ctrl.policy_name,'physics_hz':200,'policy_hz':50,
        'floor_top_m':.01,'mode':a.mode,'replay_prefix_sha256':prefix_hash,'standby_before_motion_s':3.2,'observation_after_stop_s':12.}
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2))
    reports=[]
    for name,cruise,motion_steps in cases:
        root,_=art.get_world_pose();base,_=robot.get_world_pose()
        art.set_world_pose(position=root+np.array([0.,0.,.41])-base,orientation=np.array([math.cos(initial_heading/2),0.,0.,math.sin(initial_heading/2)]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        limiter=VelocitySlewLimiter();servo=ForwardStopFeedback();gate=MeasuredStopGate();handed_off=False;handoff_time=None
        planner=SidePushPlanner(np.ones((240,320),bool),[(-8.,8.),(-6.,6.)],resolution=.05)
        planner.navigation_half_width=.22;planner.full_navigation_footprint=True
        stop_tick=640+motion_steps;rows=[];origin=None;late_origin=None;maximum=0.;minimum=float('inf');reason='completed'
        with (a.output/f'{name}.jsonl').open('w',buffering=1) as trace:
            for step in range(stop_tick+2400):
                now=float(ctrl._simulation_time());pos,q=robot.get_world_pose();q=np.asarray(q)/np.linalg.norm(q)
                tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
                heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
                if step>=640:maximum=max(maximum,tilt);minimum=min(minimum,float(pos[2]))
                if step==stop_tick:origin=np.asarray(pos[:2]).copy()
                if step==stop_tick+400:late_origin=np.asarray(pos[:2]).copy()
                moving=640<=step<stop_tick
                request=cruise if moving and commands is None else np.zeros(3)
                requested=np.asarray(commands[step-640],float) if moving and commands is not None else limiter.preview(request,now)
                limiter.accept(requested,now)
                if step>=stop_tick and not handed_off and gate.update(pos,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt,now):
                    handed_off=True;handoff_time=(step-stop_tick)*.005
                def checked(target,feedback_limiter,when):
                    return checked_contact_turn_command(target,pos,heading,planner,[],feedback_limiter,when)
                sent=(servo.apply(requested,[*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],now,
                    stop_requested=step>=stop_tick and not handed_off,checker=checked) if a.mode=='forward_feedback' else requested)
                if tilt>=30 or pos[2]<.21:reason='health_guard';sent=np.zeros(3)
                obs,*_=env.step({'move_by_speed':sent.tolist()})
                limiter.observe([*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],float(ctrl._simulation_time()))
                row={'step':step,'time_s':now,'seconds_after_stop':(step-stop_tick)*.005,'position':list(map(float,pos)),
                    'orientation':q.tolist(),'tilt':tilt,'requested':requested.tolist(),'sent':sent.tolist(),
                    'feedback_active':servo.active,'body_velocity':ctrl.body_lin_vel.tolist(),'angular_velocity':ctrl.body_ang_vel.tolist(),
                    'joint_velocity':art.get_joint_velocities().tolist()}
                rows.append(row)
                if step%4==0:trace.write(json.dumps(row)+'\n')
                if reason!='completed':break
        late=[r for r in rows if r['step']>=stop_tick+400]
        result={'case':name,'reason':reason,'maximum_tilt':maximum,'minimum_height':minimum,
            'healthy':reason!='health_guard' and maximum<30 and minimum>=.21,
            'total_stop_displacement_m':None if origin is None else float(np.linalg.norm(np.asarray(rows[-1]['position'][:2])-origin)),
            'late_drift_after_two_seconds_m':None if not late else max(float(np.linalg.norm(np.asarray(r['position'][:2])-late_origin)) for r in late),
            'first_measured_stop_time_s':handoff_time,'feedback_active_after_handoff':servo.active,
            'final_body_velocity':rows[-1]['body_velocity'],'final_angular_velocity':rows[-1]['angular_velocity']}
        reports.append(result);(a.output/'result.json').write_text(json.dumps({'diagnostic_only':True,'complete':False,'cases':reports},indent=2))
        print(json.dumps(result),flush=True)
    (a.output/'result.json').write_text(json.dumps({'diagnostic_only':True,'complete':True,'cases':reports},indent=2))
except Exception:
    import traceback;traceback.print_exc();raise
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
