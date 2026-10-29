"""Physical Go2 staging against saved online geometry; diagnostic, not MV7 acceptance."""
from pathlib import Path
import argparse
import json
import hashlib
import math
import traceback
import numpy as np
import torch
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.configs.objects import FixedCubeCfg
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.interactive_navigation.go2_actuation import physics_steps_per_command

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--policy',type=Path,required=True);p.add_argument('--robot-asset',type=Path,required=True)
p.add_argument('--label',choices=['no_hold','with_hold'],required=True)
p.add_argument('--command-profile',default='slew');a=p.parse_args();a.output.mkdir(exist_ok=False)
torch.set_num_threads(1);root=Path.cwd()
with np.load(root/'tests/data/t05_staging_boundary_v119.npz') as data:
    free=data['free'].copy();limits=data['limits'].tolist();position=data['position'].copy();q=data['orientation'].copy()
heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=(0,0,.41),goal=(7,4,.41))
cfg=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,
    actuator_mode='explicit_dc',policy_path=str(a.policy),stationary_joint_hold=a.label=='with_hold'),objects=[FixedCubeCfg(name='grscene_go2_floor',
    prim_path='/World/env_0/objects/grscene_go2_floor',position=(0,0,.005),scale=(16,12,.01),
    color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.)])
rcfg=cfg.task_config.episodes[0].robots[0];rcfg.usd_path=str(a.robot_asset);rcfg.physics_profile='isaaclab_training';rcfg.sensors=[]
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu);env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'))
    env=runner.Env(runtime);env.reset();robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()))
    art=robot.isaac_robot;ctrl=robot._go2_locomotion_control
    from grutopia_extension.interactive_navigation.go2_model_contract import collect_model_contract,validate_training_model_contract
    contract=validate_training_model_contract(collect_model_contract(robot,str(a.policy),str(a.robot_asset),'isaaclab_training'))
    contract.update(diagnostic_only=True,physical_environment='flat floor; saved map constrains commands; no MV7 prior injected into a task',
        staging_source_sha256=hashlib.sha256((root/'grutopia_extension/interactive_navigation/local_staging_control.py').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256((root/'tests/data/t05_staging_boundary_v119.npz').read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diagnostic_profile=a.label)
    (a.output/'robot_model_contract.json').write_text(json.dumps(contract,indent=2))
    assert contract['stationary_joint_hold']==(a.label=='with_hold')
    from grutopia_extension.interactive_navigation.go2_stationary_hold import stationary_hold_permitted
    from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery
    contract['production_navigation_attitude_recovery']=True
    contract['stationary_hold_source_sha256']=hashlib.sha256((root/'grutopia_extension/interactive_navigation/go2_stationary_hold.py').read_bytes()).hexdigest()
    contract['controller_source_sha256']=hashlib.sha256((root/'grutopia_extension/controllers/isaac_go2_ctrl.py').read_bytes()).hexdigest()
    (a.output/'robot_model_contract.json').write_text(json.dumps(contract,indent=2))
    reports=[]
    for name,goal,contact_permission_test in [('normal_restart',.7064224145543709,False),('contact_permission_restart',-1.0864211847023988,True)]:
        art_root,_=art.get_world_pose();body_root,_=robot.get_world_pose()
        art.set_world_pose(position=art_root+np.array([*position[:2],.41])-body_root,
            orientation=np.array([math.cos(heading/2),0.,0.,math.sin(heading/2)]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        planner=SidePushPlanner(free.copy(),limits,resolution=.05);planner.navigation_route_radius=.30
        planner.navigation_half_width=.22;planner.full_navigation_footprint=True
        limiter=VelocitySlewLimiter();turn=None;stop=MeasuredStopGate(.4);done_at=None;rows=[];maximum=0.;minimum=1e9
        started=float(ctrl._simulation_time());reason='stage_timeout';healthy=True;settled=False;staging_start=None
        stage_completed=False;walk_origin=None;walk_end=None;contact_violation=False;physical_steps=0
        attitude=NavigationAttitudeRecovery();attitude_events=[]
        for tick in range(7000):
            if (a.output.parent/'STOP').exists():reason='resource_stop';break
            now=float(ctrl._simulation_time());elapsed=now-started
            pos,quat=robot.get_world_pose();quat=np.asarray(quat)
            yaw=math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
            state='warmup';command=np.zeros(3);allowed=True;age=None
            recovery=attitude.update(pos,quat,enabled=elapsed>=3.2 and done_at is None)
            if recovery['started'] or recovery['state'] in ['released','failed']:
                attitude_events.append({'time_s':now-started,**recovery})
            if recovery['state']=='failed':reason='navigation_attitude_recovery_timeout';break
            if recovery['state']=='hold':state='attitude_recovery';command=np.zeros(3)
            elif recovery['state']=='released':
                # Same restart semantics as main: discard the old maneuver
                # after recovery and plan from the measured settled pose.
                turn=None;state='recovery_release';command=np.zeros(3)
            elif elapsed>=3.2 and done_at is None:
                if turn is None:
                    turn=ContactTurnControl(pos,yaw,goal,planner,[],now);staging_start=np.asarray(pos[:2]).copy()
                result=turn.update(pos,yaw,planner,[],limiter,now)
                if result is None:done_at=now;state='stage_completed';stage_completed=True;reason='completed'
                elif turn.failed:done_at=now;state='stage_failed';reason='staging_failed'
                else:command,detail=result;state=detail['staging_phase']
            elif done_at is not None:
                age=now-done_at
                if not stage_completed:
                    state='failed_stop';command=limiter.preview(np.zeros(3),now)
                elif age<8.:
                    state='initial_idle';command=limiter.preview(np.zeros(3),now)
                elif age<11.:
                    if walk_origin is None:walk_origin=np.asarray(pos[:2]).copy()
                    state='restart_motion'
                    command=limiter.preview([.4,0.,-.25 if contact_permission_test else .25],now)
                else:
                    if walk_end is None:walk_end=np.asarray(pos[:2]).copy()
                    state='final_idle';command=limiter.preview(np.zeros(3),now)
                # This exercises the production contact-permission boundary
                # on open floor, not a claim of physical box pushing.
                if contact_permission_test and 8.<=age<11.6:
                    allowed=stationary_hold_permitted('skill','push' if age<11. else 'settle_push_progress')
                else:allowed=stationary_hold_permitted('navigate')
            ctrl.stationary_hold_allowed=allowed
            limiter.accept(command,now)
            for _ in range(physics_steps_per_command(tick,'explicit_dc')):
                obs,*_=env.step({'move_by_speed':command.tolist()});physical_steps+=1
                quat=np.asarray(obs['orientation']);quat=quat/np.linalg.norm(quat)
                tilt=math.degrees(math.acos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1)))
                maximum=max(maximum,tilt);minimum=min(minimum,float(obs['position'][2]))
                if tilt>=30. or obs['position'][2]<.21:healthy=False
            diagnostic=ctrl.get_obs();now=float(ctrl._simulation_time());hold=diagnostic['stationary_hold']
            limiter.observe([*diagnostic['body_lin_vel'][:2],diagnostic['body_ang_vel'][2]],now,
                body_state=(obs['position'],diagnostic['body_lin_vel'],diagnostic['body_ang_vel'],tilt))
            if contact_permission_test and age is not None and 8.025<=age<11.6 and hold and hold['state']=='holding':contact_violation=True
            rows.append({'time_s':now-started,'phase':state,'position':list(map(float,obs['position'])),
                'command':command.tolist(),'tilt_deg':tilt,'body_velocity':diagnostic['body_lin_vel'].tolist(),
                'body_angular_velocity':diagnostic['body_ang_vel'].tolist(),'stationary_hold':hold,
                'hold_allowed':allowed,'motor_joint_target':diagnostic['motor_joint_target'].tolist(),
                'raw_policy_action':diagnostic['raw_policy_action'].tolist(),'actual_action':diagnostic['policy_action'].tolist()})
            if state=='final_idle' and np.max(np.abs(command))<=1e-8:
                settled=stop.update(obs['position'],diagnostic['body_lin_vel'],diagnostic['body_ang_vel'],tilt,now)
            else:stop.reset();settled=False
            if tick%100==0:(a.output/'live.json').write_text(json.dumps({'case':name,'tick':tick,'elapsed_s':now-started,'phase':state,'max_tilt_deg':maximum,'hold':hold}))
            if not healthy:reason='health_guard';break
            if done_at is not None and ((not stage_completed and now-done_at>=3.) or (stage_completed and now-done_at>=21.)):break
        assert abs(ctrl.policy_inferences-(physical_steps+3)//4)<=1,'Policy inference cadence changed'
        walking=0. if walk_origin is None or walk_end is None else float(np.linalg.norm(walk_end-walk_origin))
        hold=ctrl.get_obs()['stationary_hold']
        held_and_resumed=(a.label=='no_hold' or (hold['capture_count']>=2 and hold['release_count']>=2))
        accepted=bool(healthy and stage_completed and settled and walking>.25 and held_and_resumed and not contact_violation)
        report={'case':name,'reason':reason,'healthy':healthy,'stage_completed':stage_completed,'final_stop_valid':bool(settled),
            'maximum_tilt_deg':maximum,'minimum_height_m':minimum,'restart_displacement_m':walking,
            'stationary_hold_report':hold,'contact_permission_violation':contact_violation,'physical_steps':physical_steps,
            'policy_inferences':ctrl.policy_inferences,'navigation_attitude_recovery':attitude.report(),'recovery_events':attitude_events,'all_checks_passed':accepted,'diagnostic_only':True}
        reports.append(report);(a.output/(name+'.json')).write_text(json.dumps(rows));print(json.dumps(report),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'diagnostic_only':True,'navigation_accepted':False,'label':a.label,
        'cases':reports,'all_cases_healthy_and_stopped':all(r['all_checks_passed'] for r in reports)},indent=2))
except BaseException as error:
    failure={'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    (a.output/'failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure),flush=True);raise
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
