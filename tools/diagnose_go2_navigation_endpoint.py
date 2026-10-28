"""Flat-floor endpoint-turn regression; never a substitute for MV7 navigation."""
import argparse,hashlib,importlib.util,json,math,shutil
from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker,VelocitySlewLimiter
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.configs.objects import FixedCubeCfg

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--tracker-source',type=Path);p.add_argument('--mirror',type=float,choices=[-1.,1.],default=1.);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,args.output/'source.py')
source=args.tracker_source or Path('grutopia_extension/interactive_navigation/smooth_go2_navigation.py')
shutil.copy2(source,args.output/'tracker_source.py')
if args.tracker_source:
 spec=importlib.util.spec_from_file_location('baseline_tracker',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 SmoothExplorationTracker=module.SmoothExplorationTracker
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=(0.,0.,.41),goal=(-3.35,1.95*args.mirror,.41))
config=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=args.gpu,ground_height=.01,actuator_mode='explicit_dc'),
 objects=[FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(16.,12.,.01),
                      color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.)])
config.simulator.rendering_dt=.005;config.simulator.rendering_interval=39
config.task_config.episodes[0].robots[0].sensors=[]
runtime=runner.SimulatorRuntime(config_class=config,headless=True,active_gpu=args.gpu,physics_gpu=args.gpu)
env=None
try:
 runner.import_extensions(('controllers','objects','robots','sensors','tasks'))
 env=runner.Env(runtime);obs,_=env.reset()
 robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()));art=robot.isaac_robot;ctrl=robot._go2_locomotion_control
 root,_=art.get_world_pose();base,_=robot.get_world_pose()
 art.set_world_pose(position=root+np.array([0.,0.,.41])-base,orientation=np.array([0.,0.,0.,1.]))
 art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
 robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
 planner=SidePushPlanner(np.ones((240,320),bool),[(-8.,8.),(-6.,6.)],resolution=.05)
 planner.full_navigation_footprint=True;planner.navigation_half_width=.22
 vertices=[np.array([0.,0.]),np.array([-4.,0.]),np.array([-4.,1.95*args.mirror]),np.array([-3.35,1.95*args.mirror])]
 path=[]
 for first,last in zip(vertices,vertices[1:]):
  count=int(math.ceil(np.linalg.norm(last-first)/.3))
  path.extend([tuple(first+(last-first)*i/count) for i in range(1,count+1)])
 goal=vertices[-1];limiter=VelocitySlewLimiter();tracker=None;stop_gate=MeasuredStopGate();goal_braking=False;active_path=path
 phase='warmup';reason=None;stopped_at=None;maximum=0.;minimum_height=float('inf');trace=(args.output/'trace.jsonl').open('w',buffering=1)
 (args.output/'contract.json').write_text(json.dumps({'diagnostic_only':True,'navigation_task':False,'scene':'flat_16x12_fixture',
     'original_mv7_task':False,'policy':ctrl.policy_name,'actuator_mode':'explicit_dc','physics_hz':200,'policy_hz':50,
     'floor_top':.01,'path':path,'goal':goal.tolist(),'initial_heading':math.pi,'tracker_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
 for step in range(12640):
  now=float(ctrl._simulation_time());pos,q=robot.get_world_pose();q=np.asarray(q)/np.linalg.norm(q)
  tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
  heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
  planner.control_time=now;command=np.zeros(3);detail={}
  if step>=640:
   maximum=max(maximum,tilt);minimum_height=min(minimum_height,float(pos[2]))
   if reason is None and (tilt>=30. or pos[2]<.21):reason='robot_unstable';stopped_at=step
   if reason is None:
    if tracker is None:tracker=SmoothExplorationTracker(pos[:2],active_path,limiter)
    error=float(np.linalg.norm(pos[:2]-goal))
    if error<.20:goal_braking=True
    if goal_braking:
     phase='goal_braking';command,_=checked_contact_turn_command([0.,0.,0.],pos,heading,planner,[],limiter,now)
     if stop_gate.update(pos,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt,now):
      if error<.25:reason='goal_settled';stopped_at=step
      else:goal_braking=False;stop_gate.reset();tracker=None
    else:
     action,detail=tracker.action(pos[:2],heading,planner,[]);command=np.asarray(action['move_by_speed'])
     phase=detail.get('staging_phase','navigate')
     if detail.get('staging_completed'):
      active_path=tracker._planned_path[tracker._waypoint_index:];tracker=None
  limiter.accept(command,now);obs,*_=env.step({'move_by_speed':command.tolist()})
  limiter.observe([*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],float(ctrl._simulation_time()))
  if step%4==0:
   trace.write(json.dumps({'step':step,'time_s':now,'phase':phase,'command':command.tolist(),'position':list(map(float,pos)),
       'heading':heading,'tilt':tilt,'velocity':ctrl.body_lin_vel.tolist(),'angular':ctrl.body_ang_vel.tolist(),
       'target':detail.get('target'),'endpoint_turn_braking':detail.get('endpoint_turn_braking',False)})+'\n')
  if stopped_at is not None and step-stopped_at>=400:break
 pos,_=robot.get_world_pose();error=float(np.linalg.norm(pos[:2]-goal))
 result={'diagnostic_only':True,'reason':reason or 'time_limit','physical_steps':step+1,'maximum_tilt':maximum,
         'goal_error_m':error,'minimum_height':minimum_height,'healthy':maximum<30. and minimum_height>=.21,
         'goal_accepted':reason=='goal_settled' and maximum<30 and minimum_height>=.21 and error<.25}
 (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);trace.close()
except Exception:
 import traceback;traceback.print_exc();raise
finally:
 if env is not None:env.close()
 else:runtime.simulation_app.close()
