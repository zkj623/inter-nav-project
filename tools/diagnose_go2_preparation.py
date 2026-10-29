"""Isolated physical preparation test; never MV7 navigation acceptance."""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,json,math,shutil,subprocess
import numpy as np
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl
from grutopia_extension.interactive_navigation.go2_interaction_motion import checked_skill_brake
from grutopia_extension.configs.objects import FixedCubeCfg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);shutil.copy2(__file__,a.output/'source.py')
box=BoxState('B1',(1.,0.),(.85,.65,.7),mass_kg=5.)
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=(-.21,-.44,.41),goal=(3.,0.,.41))
cfg=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,actuator_mode='explicit_dc'),objects=[
    FixedCubeCfg(name='grscene_go2_floor',prim_path='/World/env_0/objects/grscene_go2_floor',position=(0.,0.,.005),scale=(16.,12.,.01),
        color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.),
    FixedCubeCfg(name='preparation_box',prim_path='/World/env_0/objects/preparation_box',position=(1.,0.,.36),scale=box.dimensions,color=(.6,.4,.2))])
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39;cfg.task_config.episodes[0].robots[0].sensors=[]
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu);env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'));env=runner.Env(runtime);obs,_=env.reset()
    robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()));art=robot.isaac_robot;ctrl=robot._go2_locomotion_control
    contract={'diagnostic_only':True,'navigation_task':False,'no_mv7_scene':True,'map':'known_free_flat_fixture_with_fixed_box',
        'policy':ctrl.policy_name,'physics_hz':200,'policy_hz':50,'floor_top_m':.01,'box':vars(box),
        'purpose':'compare old duplicate standoff dwell with reuse of completed ContactTurnControl stop; stop before deliberate contact',
        'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_diff':subprocess.check_output(['git','diff','--','grutopia_extension/interactive_navigation/push_aside_skill.py'],text=True)}
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2));reports=[]
    for variant in ['baseline_duplicate_dwell','reuse_turn_stop']:
        root,_=art.get_world_pose();base,_=robot.get_world_pose();initial_heading=.30
        art.set_world_pose(position=root+np.array([-.21,-.44,.41])-base,orientation=np.array([math.cos(initial_heading/2),0.,0.,math.sin(initial_heading/2)]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        for _ in range(640):env.step({'move_by_speed':[0.,0.,0.]})
        limiter=VelocitySlewLimiter();planner=SidePushPlanner(np.ones((240,320),bool),[(-8.,8.),(-6.,6.)],resolution=.05)
        planner.navigation_half_width=.22;planner.full_navigation_footprint=True
        permission=InteractionPermission('B1','box','push_aside',(),(),0.,source='ground_truth_layout',source_reference='flat_fixture',source_digest='a'*64)
        skill=PushAsideSkill(permission,NS(box_id='B1',direction=(1.,0.),push_distance=.85),(0.,-2.),(0.,2.))
        now=float(ctrl._simulation_time());position,q=robot.get_world_pose()
        heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        skill.started=now;skill.state_since=now;skill.state='turn_at_stage';skill.initial_box=np.asarray(box.xy)
        skill.anchor=np.array([1.-.425-.33-.35,0.]);skill.stage=np.asarray(position[:2]).copy()
        skill.turn=ContactTurnControl(position,heading,0.,planner,[box],now)
        origin=now;maximum=0.;lateral_started=None;turn_handoff=None;reason='diagnostic_time_limit';events=[]
        with (a.output/(variant+'.jsonl')).open('w',buffering=1) as trace:
            for step in range(8000):
                now=float(ctrl._simulation_time());position,q=robot.get_world_pose();q=np.asarray(q)/np.linalg.norm(q)
                heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
                tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)));maximum=max(maximum,tilt)
                limiter.observe([*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],now)
                ctx=SkillContext(now,np.asarray(position),heading,ctrl.body_lin_vel.copy(),float(ctrl.body_ang_vel[2]),tilt,
                    tilt<30 and position[2]>=.21,box,[box],planner,None,limiter,body_angular_velocity=ctrl.body_ang_vel.copy())
                command=skill.step(ctx)
                if variant=='baseline_duplicate_dwell' and any(e.get('reused_turn_stop_for_standoff') for e in command.events):
                    skill.state='align_longitudinal';command.state='align_longitudinal'
                    for event in command.events:
                        if event.get('reused_turn_stop_for_standoff'):
                            event.update(to='align_longitudinal',reused_turn_stop_for_standoff=False,diagnostic_baseline_override=True)
                for event in command.events:
                    events.append(event)
                    if event.get('reason')=='coarse_heading_ready_for_centering':turn_handoff=now-origin
                    print(json.dumps({'variant':variant,**event}),flush=True)
                sent=checked_skill_brake(command,ctx)
                if command.state=='align_lateral' and abs(sent[1])>.05 and lateral_started is None:lateral_started=now-origin
                row={'step':step,'elapsed_s':now-origin,'state':command.state,'position':position.tolist(),'heading':heading,'tilt':tilt,
                    'velocity':ctrl.body_lin_vel.tolist(),'angular_velocity':ctrl.body_ang_vel.tolist(),'sent':sent.tolist(),'retry':skill.retries,
                    'nose_gap_lateral_yaw':skill._box_frame(ctx)}
                if step%4==0:trace.write(json.dumps(row)+'\n')
                if command.state=='engage':reason='contact_preparation_ready';break
                if command.status!='running':reason=command.reason;break
                if command.state in ['brake_for_recovery','navigate_to_stage','recover_preparation']:
                    reason='preparation_recovery:'+str(skill.release_reason);break
                if not ctx.healthy:reason='health_guard';break
                limiter.accept(sent,now);env.step({'move_by_speed':sent.tolist()})
        result={'variant':variant,'reason':reason,'maximum_tilt':maximum,'elapsed_s':now-origin,
            'turn_handoff_s':turn_handoff,'first_lateral_command_s':lateral_started,'final_state':skill.state,'retries':skill.retries,
            'events':events,'final_position':position.tolist(),'final_nose_gap_lateral_yaw':skill._box_frame(ctx)}
        reports.append(result);(a.output/'result.json').write_text(json.dumps({'diagnostic_only':True,'complete':False,'cases':reports},indent=2))
        print(json.dumps(result),flush=True)
    (a.output/'result.json').write_text(json.dumps({'diagnostic_only':True,'complete':True,'cases':reports},indent=2))
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
