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
p.add_argument('--label',choices=['legacy_margin025','selected_margin'],required=True)
p.add_argument('--command-profile',default='slew');a=p.parse_args();a.output.mkdir(exist_ok=False)
torch.set_num_threads(1);root=Path.cwd()
with np.load(root/'tests/data/t05_staging_boundary_v119.npz') as data:
    free=data['free'].copy();limits=data['limits'].tolist();position=data['position'].copy();q=data['orientation'].copy()
heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=(0,0,.41),goal=(7,4,.41))
cfg=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,
    actuator_mode='explicit_dc',policy_path=str(a.policy)),objects=[FixedCubeCfg(name='grscene_go2_floor',
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
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diagnostic_profile=a.label,idle_target_hold_experiment=True,hold_has_no_rollout_pose_edits=True)
    (a.output/'robot_model_contract.json').write_text(json.dumps(contract,indent=2))
    if a.label=='legacy_margin025':
        from grutopia_extension.interactive_navigation import local_staging_control as module
        original=module.body_rotation_clear
        def legacy(*args,margin=0.):return original(*args,margin=.025)
        module.body_rotation_clear=legacy
    # Diagnostic-only intervention: keep actor inference/cadence and native
    # DC motor torque computation, but hold the last supported joint target
    # once the complete measured-stop gate has passed during final idle.
    hold_target=None
    original_infer=ctrl._infer_policy
    def infer_with_supported_idle_target():
        original_infer()
        if hold_target is not None:
            ctrl.applied_joint_positions=hold_target.copy()
            ctrl.last_action=((hold_target-ctrl.default_joint_positions)/.25).astype(np.float32)
    ctrl._infer_policy=infer_with_supported_idle_target
    reports=[]
    for name,goal in [('original_frontier',.7064224145543709),('reselected_frontier',-1.0864211847023988)]:
        art_root,_=art.get_world_pose();body_root,_=robot.get_world_pose()
        art.set_world_pose(position=art_root+np.array([*position[:2],.41])-body_root,
            orientation=np.array([math.cos(heading/2),0.,0.,math.sin(heading/2)]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        planner=SidePushPlanner(free.copy(),limits,resolution=.05);planner.navigation_route_radius=.30
        planner.navigation_half_width=.22;planner.full_navigation_footprint=True
        limiter=VelocitySlewLimiter();turn=None;stop=MeasuredStopGate(.4);done_at=None;rows=[];maximum=0.;minimum=1e9
        started=float(ctrl._simulation_time());reason='stage_timeout';healthy=True;settled=False;staging_start=None;handoff=None;hold_target=None;hold_started=None
        for tick in range(8000):
            if (a.output.parent/'STOP').exists():reason='resource_stop';break
            now=float(ctrl._simulation_time());elapsed=now-started
            pos,quat=robot.get_world_pose();quat=np.asarray(quat);yaw=math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
            state='warmup';command=np.zeros(3)
            if elapsed>=3.2 and done_at is None:
                if turn is None:
                    turn=ContactTurnControl(pos,yaw,goal,planner,[],now);staging_start=np.asarray(pos[:2]).copy()
                result=turn.update(pos,yaw,planner,[],limiter,now)
                if result is None:
                    done_at=now;state='finished';reason='completed'
                    diagnostic=ctrl.get_obs()
                    handoff={'time_s':now-started,'position':list(map(float,pos)),
                        'body_velocity':diagnostic['body_lin_vel'].tolist(),
                        'body_angular_velocity':diagnostic['body_ang_vel'].tolist(),
                        'orientation':list(map(float,quat))}

                elif turn.failed:done_at=now;state='failed';reason='staging_failed'
                else:command,detail=result;state=detail['staging_phase']
            elif done_at is not None:state='final_stop'
            limiter.accept(command,now)
            for _ in range(physics_steps_per_command(tick,'explicit_dc')):
                obs,*_=env.step({'move_by_speed':command.tolist()})
                quat=np.asarray(obs['orientation']);quat=quat/np.linalg.norm(quat)
                tilt=math.degrees(math.acos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1)))
                maximum=max(maximum,tilt);minimum=min(minimum,float(obs['position'][2]))
                if tilt>=30. or obs['position'][2]<.21:healthy=False
            diagnostic=ctrl.get_obs();now=float(ctrl._simulation_time())
            limiter.observe([*diagnostic['body_lin_vel'][:2],diagnostic['body_ang_vel'][2]],now,
                body_state=(obs['position'],diagnostic['body_lin_vel'],diagnostic['body_ang_vel'],tilt))
            rows.append({'time_s':now-started,'phase':state,'position':list(map(float,obs['position'])),
                'body_velocity':diagnostic['body_lin_vel'].tolist(),'body_angular_velocity':diagnostic['body_ang_vel'].tolist(),'orientation':list(map(float,quat)),'command':command.tolist(),'tilt_deg':tilt,'margin':None if turn is None else turn.stage.selector.stage_margin})
            if tick%100==0:(a.output/'live.json').write_text(json.dumps({'case':name,'tick':tick,'elapsed_s':now-started,'phase':state,'max_tilt_deg':maximum}))
            if not healthy:reason='health_guard';break
            if done_at is not None:
                settled=stop.update(obs['position'],diagnostic['body_lin_vel'],diagnostic['body_ang_vel'],tilt,now)
                if settled and hold_target is None:
                    hold_target=ctrl.applied_joint_positions.copy();hold_started=now-started
                if now-done_at>=20.:break
        report={'case':name,'reason':reason,'healthy':healthy,'stage_completed':reason=='completed',
            'final_stop_valid':bool(settled),'maximum_tilt_deg':maximum,'minimum_height_m':minimum,
            'staging_displacement_m':None if staging_start is None else float(np.linalg.norm(np.asarray(obs['position'][:2])-staging_start)),
            'diagnostic_idle_target_hold':True,'hold_started_s':hold_started,'held_joint_target':None if hold_target is None else hold_target.tolist(),'handoff':handoff,'final_body_velocity':diagnostic['body_lin_vel'].tolist(),'final_body_angular_velocity':diagnostic['body_ang_vel'].tolist(),'final_position':list(map(float,obs['position']))}
        reports.append(report);(a.output/(name+'.json')).write_text(json.dumps(rows))
        print(json.dumps(report),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'diagnostic_only':True,'navigation_accepted':False,'label':a.label,
        'cases':reports,'all_cases_healthy_and_stopped':all(r['healthy'] and r['stage_completed'] and r['final_stop_valid'] for r in reports)},indent=2))
except BaseException as error:
    failure={'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    (a.output/'failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure),flush=True);raise
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
