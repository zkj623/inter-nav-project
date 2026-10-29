"""Physical release recovery in a raster fixture reconstructed from an online snapshot.

This is NOT MV7: static cells become0.5m blocks; the box is fixed because
the test starts after pushing and stops after withdrawal. No navigation success.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,json,math,shutil
import numpy as np
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.go2_interaction_motion import checked_skill_brake,MeasuredStopGate
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear
from grutopia_extension.configs.objects import FixedCubeCfg
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--fixture',type=Path,required=True);p.add_argument('--spawn-offset',type=float,nargs=2,default=(0.,0.));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.output/'source.py');shutil.copy2(a.fixture,a.output/'fixture.npz')
with np.load(a.fixture,allow_pickle=False) as z:
    free=z['local_free'].copy();observed=z['observed'].copy();box_free=z['box_free'].copy();limits=z['limits'].copy()
    origin=z['position'].copy();heading0=float(z['heading']);boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
shift=np.array([10.,0.]);origin[:2]-=shift;limits-=shift[:,None]
boxes=[BoxState(b.id,tuple(np.array(b.xy)-shift),b.dimensions,b.yaw,b.mass_kg) for b in boxes];box=next(b for b in boxes if b.id=='B06')
planner=SidePushPlanner(free,limits.tolist(),resolution=.05);planner.box_free=box_free;planner.planning_observed=observed
planner.navigation_half_width=.22;planner.full_navigation_footprint=True;planner.layered_local_control=True;planner.planning_scope='local';planner.navigation_route_radius=.30
objects=[FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(16.,12.,.01),
    color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.)]
objects.append(FixedCubeCfg(name='released_box',prim_path='/World/env_0/objects/released_box',position=(*box.xy,.01+box.dimensions[2]/2),scale=box.dimensions,color=(.6,.4,.2)))
occupied=observed&~box_free
# Exact row runs preserve every free cell; no bounding-box filling of L shapes.
for row in range(occupied.shape[0]):
    y=limits[1,0]+(row+.5)*.05
    if abs(y-origin[1])>1.2:continue
    cols=np.flatnonzero(occupied[row]);near=[c for c in cols if abs(limits[0,0]+(c+.5)*.05-origin[0])<=1.2]
    for group in np.split(near,np.flatnonzero(np.diff(near)>1)+1):
        if not len(group):continue
        x=limits[0,0]+(float(group[0])+float(group[-1])+1)*.025
        name='observed_static_'+str(len(objects))
        objects.append(FixedCubeCfg(name=name,prim_path='/World/env_0/objects/'+name,position=(x,y,.26),scale=(len(group)*.05,.05,.5),color=(.4,.4,.4)))
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=tuple(origin),goal=(.3,1.75,.41))
cfg=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,actuator_mode='explicit_dc'),objects=objects)
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39;cfg.task_config.episodes[0].robots[0].sensors=[]
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu);env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'));env=runner.Env(runtime);env.reset()
    robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()));art=robot.isaac_robot;ctrl=robot._go2_locomotion_control
    spawn=origin.copy();spawn[:2]+=a.spawn_offset
    root,_=art.get_world_pose();base,_=robot.get_world_pose();art.set_world_pose(position=root+spawn-base,orientation=np.array([math.cos(heading0/2),0.,0.,math.sin(heading0/2)]))
    art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof));robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
    for _ in range(640):env.step({'move_by_speed':[0.,0.,0.]})
    permission=InteractionPermission('B06','box','push_aside',(),(),0.,source='ground_truth_layout',source_reference='raster_fixture',source_digest='a'*64)
    skill=PushAsideSkill(permission,NS(box_id='B06',direction=(-1.,0.),push_distance=.85),(-7.89,-.78),(.3,1.75))
    skill.initial_box=np.array(box.xy)+np.array([.620680809,0.]);skill.engage_box_position=skill.initial_box.copy()
    skill.started=float(ctrl._simulation_time());skill.state='settle_released_corridor';skill.state_since=skill.started
    limiter=VelocitySlewLimiter();gate=MeasuredStopGate();start=skill.started;maximum=0.;reason='diagnostic_time_limit';begun=False;rows=0
    with (a.output/'trace.jsonl').open('w',buffering=1) as trace:
        for step in range(4000):
            now=float(ctrl._simulation_time());pos,q=robot.get_world_pose();q=np.asarray(q)/np.linalg.norm(q)
            heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)));maximum=max(maximum,tilt)
            limiter.observe([*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],now)
            ctx=SkillContext(now,np.asarray(pos),heading,ctrl.body_lin_vel.copy(),float(ctrl.body_ang_vel[2]),tilt,tilt<30 and pos[2]>=.21,box,boxes,planner,None,limiter,body_angular_velocity=ctrl.body_ang_vel.copy())
            if not begun:
                if not skill._begin_release_pose_recovery(ctx,.620680809):reason='initial_fixture_pose_has_no_admitted_recovery';break
                begun=True
            if skill.state=='settle_released_corridor':
                sent=np.zeros(3)
                if gate.update(pos,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt,now) and body_motion_clear(planner,boxes,pos,heading,0.,0.,horizon=0.):reason='body_recovered_and_measured_stopped';break
            else:
                command=skill.step(ctx);sent=checked_skill_brake(command,ctx)
                for event in command.events:print(json.dumps(event),flush=True)
                if command.status!='running':reason=command.reason;break
            if not ctx.healthy:reason='health_guard';break
            if step%4==0:
                trace.write(json.dumps({'elapsed_s':now-start,'position':pos.tolist(),'heading':heading,'tilt':tilt,'state':skill.state,'sent':sent.tolist(),'velocity':ctrl.body_lin_vel.tolist(),'angular':ctrl.body_ang_vel.tolist()})+'\n');rows+=1
            limiter.accept(sent,now);env.step({'move_by_speed':sent.tolist()})
    result={'diagnostic_only':True,'navigation_task':False,'rasterized_static_scene':True,'physical_objects':len(objects),'policy':ctrl.policy_name,
        'physics_hz':200,'policy_hz':50,'spawn_offset':a.spawn_offset,'reason':reason,'elapsed_s':now-start,'maximum_tilt':maximum,'final_position':pos.tolist(),
        'final_body_free':body_motion_clear(planner,boxes,pos,heading,0.,0.,horizon=0.),'final_velocity':ctrl.body_lin_vel.tolist(),
        'final_angular_velocity':ctrl.body_ang_vel.tolist(),'trace_rows':rows,'original_task_success':False}
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
