"""Calibrate settled heading trim with the original Go2 policy (not a task result)."""
import argparse,json,math,hashlib,shutil
from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_control import ContactYawServo
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command,ContactTurnControl
from grutopia_extension.interactive_navigation.go2_actuation import physics_steps_per_command
from grutopia_extension.configs.objects import FixedCubeCfg
from grutopia_extension.interactive_navigation.go2_heading_trim import HeadingTrimControl
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adaptive',action='store_true');p.add_argument('--acceptance',type=float,default=.05);p.add_argument('--coarse-first',action='store_true');args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,args.output/'source.py')
(args.output/'source_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
cases=[(mode,error) for mode in (['adaptive_arc'] if args.adaptive else ['yaw_only','backward_arc']) for error in [.12,-.12,.25,-.25]]
shutil.copy2(Path(__file__).resolve().parents[1]/'grutopia_extension/interactive_navigation/go2_heading_trim.py',args.output/'heading_trim_source.py')
profile=programmatic_go2_profile(environment_length=12.,environment_width=10.,start=(0.,0.,.41),goal=(3.,0.,.41))
run=runner.Go2NavigationRunConfig(gpu=args.gpu,ground_height=.01,actuator_mode='explicit_dc')
floor=FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(12.,10.,.01),color=(.4,.4,.4))
cfg=runner.build_go2_navigation_config(profile,run,objects=[floor]);cfg.task_config.episodes[0].robots[0].sensors=[]
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=args.gpu,physics_gpu=args.gpu);env=None
try:
 runner.import_extensions(('controllers','objects','robots','sensors','tasks'));env=runner.Env(runtime);obs,_=env.reset()
 robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()));ctrl=robot._go2_locomotion_control;art=robot.isaac_robot
 runner._apply_high_friction_material('/World/env_0/objects/grscene_go2_floor')
 planner=SidePushPlanner(np.ones((200,240),bool),[[-6.,6.],[-5.,5.]],resolution=.05);planner.navigation_half_width=.22
 reports=[]
 for mode,error0 in cases:
  root,_=art.get_world_pose();base,_=robot.get_world_pose();art.set_world_pose(position=root+np.array([0.,0.,.41])-base,orientation=np.array([1.,0.,0.,0.]))
  art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
  robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
  limiter=VelocitySlewLimiter();servo=ContactYawServo(.05);target=None;origin=None;ready_since=None;done_step=None;failure=None;next_update=0;logical=0;command=np.zeros(3);rows=[];trim=None;coarse=None;phase_start=640
  label=mode+('_plus' if error0>0 else '_minus')+str(abs(error0))
  with (args.output/(label+'.jsonl')).open('w',buffering=1) as trace:
   for step in range(8840 if args.coarse_first else 4840):
    now=float(ctrl._simulation_time());pos,q=robot.get_world_pose();q=np.asarray(q);q/=np.linalg.norm(q)
    heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
    if step>=640 and target is None:
     target=heading+error0;origin=np.asarray(pos[:2]).copy();trim=HeadingTrimControl(target,tolerance=.05,acceptance_tolerance=args.acceptance)
     if args.coarse_first and abs(error0)>.12:coarse=ContactTurnControl(pos,heading,target,planner,[],now)
    error=0. if target is None else (target-heading+math.pi)%(2*math.pi)-math.pi
    if step>=640 and (tilt>=18. or pos[2]<.21):failure='attitude_guard'
    if step>=phase_start+4000 and done_step is None:failure='phase_timeout'
    if origin is not None and coarse is None and np.linalg.norm(pos[:2]-origin)>.60:failure='fixture_translation_boundary'
    if step>=next_update:
     limiter.observe([*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],now)
     if mode=='adaptive_arc':
      if target is None or done_step is not None or failure is not None:
       command,_=checked_contact_turn_command([0.,0.,0.],pos,heading,planner,[],limiter,now)
      elif coarse is not None:
       answer=coarse.update(pos,heading,planner,[],limiter,now)
       if coarse.failed:failure='coarse_turn_failed'
       if answer is None:
        coarse=None;trim=HeadingTrimControl(target,tolerance=.05,acceptance_tolerance=args.acceptance)
        origin=np.asarray(pos[:2]).copy();phase_start=step
        command,_=checked_contact_turn_command([0.,0.,0.],pos,heading,planner,[],limiter,now)
       else:command=answer[0]
      else:
       command,done,detail=trim.update(pos,heading,planner,[],limiter,now,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt)
       if done:done_step=step
       if trim.failed:failure=trim.failed
      if failure=='attitude_guard':command=np.zeros(3)
     else:
      request=np.zeros(3)
      if target is not None and done_step is None and failure is None:
       quiet=(np.linalg.norm(ctrl.body_lin_vel)<.08 and np.max(np.abs(ctrl.body_ang_vel))<.10
              and np.linalg.norm(limiter.command[:2])<.03 and abs(limiter.command[2])<.06)
       if abs(error)<.05 and quiet:
        if ready_since is None:ready_since=now
        if now-ready_since>=.30:done_step=step
       else:ready_since=None
       if abs(error)>=.05:
        rate=servo.update(error,float(ctrl.body_ang_vel[2]),now)
        request=np.array([-.20 if mode=='backward_arc' and abs(rate)>.01 else 0.,0.,rate])
      command,_=checked_contact_turn_command(request,pos,heading,planner,[],limiter,now)
      if failure=='attitude_guard':command=np.zeros(3)
     limiter.accept(command,now);next_update=step+physics_steps_per_command(logical,'explicit_dc');logical+=1
    obs,*_=env.step({'move_by_speed':command.tolist()})
    row={'step':step,'time_s':float(ctrl._simulation_time()),'position':list(map(float,obs['position'])),'heading_before_step':heading,'error_before_step':error,'tilt_before_step':tilt,'command':command.tolist(),'velocity':ctrl.body_lin_vel.tolist(),'angular':ctrl.body_ang_vel.tolist(),'failure':failure}
    rows.append(row)
    if step%4==0:trace.write(json.dumps(row)+'\n')
    if (done_step is not None and step-done_step>=200) or (failure is not None and step>=640+400 and np.linalg.norm(command)<.01):break
  active=rows[640:];final_error=abs(active[-1]['error_before_step'])
  report={'mode':mode,'coarse_first':args.coarse_first,'initial_error':error0,'aiming_tolerance':.05,'settled_tolerance':args.acceptance,'reached_and_held':bool(done_step is not None and failure is None and final_error<args.acceptance),'final_error':final_error,'maximum_tilt':max(x['tilt_before_step'] for x in active),'translation_m':float(np.linalg.norm(np.asarray(active[-1]['position'][:2])-origin)),'steps':len(rows),'failure':failure,'transient_reached_step':done_step}
  reports.append(report);(args.output/'results.json').write_text(json.dumps({'diagnostic_only':True,'complete':False,'cases':reports},indent=2));print(json.dumps(report),flush=True)
 (args.output/'results.json').write_text(json.dumps({'diagnostic_only':True,'complete':True,'cases':reports},indent=2))
finally:
 if env is not None:env.close()
 else:runtime.simulation_app.close()
