"""Same-plant policy benchmark; independent resets, not navigation acceptance."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import traceback
import numpy as np
import torch
from grutopia_extension.interactive_navigation import go2_navigation_runner as runner
from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.configs.objects import FixedCubeCfg
from grutopia_extension.configs.robots.go2 import GO2_JOINT_NAMES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
p.add_argument('--command-profile',choices=['slew','direct'],default='slew')
p.add_argument('--robot-asset',type=Path)
p.add_argument('--physics-profile',choices=['legacy','isaaclab_training'],default='isaaclab_training')
p.add_argument('--physics-backend',choices=['default','gpu_dynamics'],default='default')
p.add_argument('--case-ids',help='Comma-separated subset; default runs all ten scripts')
a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
cases={
    'forward_stop':[(3.,(0,0,0)),(4.,(.5,0,0)),(3.,(0,0,0))],
    'reverse_stop':[(3.,(0,0,0)),(3.,(-.35,0,0)),(3.,(0,0,0))],
    'left_stop':[(3.,(0,0,0)),(4.,(0,.35,0)),(3.,(0,0,0))],
    'right_stop':[(3.,(0,0,0)),(4.,(0,-.35,0)),(3.,(0,0,0))],
    'yaw_left_stop':[(3.,(0,0,0)),(4.,(0,0,.6)),(3.,(0,0,0))],
    'yaw_right_stop':[(3.,(0,0,0)),(4.,(0,0,-.6)),(3.,(0,0,0))],
    'fast_yaw_left_stop':[(3.,(0,0,0)),(3.,(0,0,1.)),(3.,(0,0,0))],
    'fast_yaw_right_stop':[(3.,(0,0,0)),(3.,(0,0,-1.)),(3.,(0,0,0))],
    'small_adjustments':[(3.,(0,0,0)),(1.,(.2,0,0)),(1.,(0,0,0)),(1.,(0,.2,0)),(1.,(0,0,0)),(1.,(0,-.2,0)),(3.,(0,0,0))],
    'mode_switches':[(3.,(0,0,0)),(3.,(.5,0,0)),(1.,(0,0,0)),(2.,(0,-.35,0)),(1.,(0,0,0)),(3.,(0,0,.6)),(3.,(0,0,0))],
}
if a.case_ids:
    selected=a.case_ids.split(',')
    if any(name not in cases for name in selected):p.error('Unknown case identifier')
    cases={name:cases[name] for name in selected}
profile=programmatic_go2_profile(environment_length=16.,environment_width=12.,start=(0,0,.41),goal=(7,4,.41))
cfg=runner.build_go2_navigation_config(profile,runner.Go2NavigationRunConfig(gpu=a.gpu,ground_height=.01,
    actuator_mode='explicit_dc',policy_path=str(a.policy)),objects=[FixedCubeCfg(name='grscene_go2_floor',
    prim_path='/World/env_0/objects/grscene_go2_floor',position=(0,0,.005),scale=(16,12,.01),
    color=(.4,.4,.4),static_friction=1.,dynamic_friction=1.,torsional_patch_radius=0.,min_torsional_patch_radius=0.)])
if a.robot_asset is not None:
    assert a.robot_asset.is_file()
    cfg.task_config.episodes[0].robots[0].usd_path=str(a.robot_asset)
cfg.task_config.episodes[0].robots[0].physics_profile=a.physics_profile
cfg.simulator.rendering_dt=.005;cfg.simulator.rendering_interval=39;cfg.task_config.episodes[0].robots[0].sensors=[]
robot_asset=Path(cfg.task_config.episodes[0].robots[0].usd_path)
runtime=runner.SimulatorRuntime(config_class=cfg,headless=True,active_gpu=a.gpu,physics_gpu=a.gpu);env=None
try:
    runner.import_extensions(('controllers','objects','robots','sensors','tasks'))
    from grutopia_extension.interactive_navigation.go2_physics_backend import configure_go2_physics_backend,collect_go2_physics_contract
    env=runner.Env(runtime)
    physics_context=env.runner._world.get_physics_context()
    configure_go2_physics_backend(physics_context,a.physics_backend)
    env.reset()
    physics_contract=collect_go2_physics_contract(physics_context,a.physics_backend)
    (a.output/'physics_contract.json').write_text(json.dumps(physics_contract,indent=2))
    robot=next(iter(next(iter(env.runner.current_tasks.values())).robots.values()));art=robot.isaac_robot;ctrl=robot._go2_locomotion_control
    view=art._articulation_view
    from grutopia_extension.interactive_navigation.go2_model_contract import collect_model_contract,validate_training_model_contract
    native=collect_model_contract(robot,str(a.policy),str(robot_asset),a.physics_profile)
    if a.physics_profile=='isaaclab_training':validate_training_model_contract(native)
    native['scene_physics']=physics_contract
    (a.output/'robot_model_contract.json').write_text(json.dumps(native,indent=2,allow_nan=False))
    actual_assets=native['actual_asset_references']
    dependency=robot_asset.parent/'Props/instanceable_meshes.usd'
    contract={'diagnostic_only':True,'navigation_accepted':False,'label':a.label,'policy':str(a.policy),
        'policy_sha256':hashlib.sha256(a.policy.read_bytes()).hexdigest(),'observation_dim':ctrl.observation_dim,
        'actor_inference_hz':50,'physics_hz':200,'floor_top_m':.01,'actuator_mode':'explicit_dc',
        'command_profile':a.command_profile,'command_slew_acceleration':[.65,1.8] if a.command_profile=='slew' else None,
        'native_actuation':native,'joint_names':GO2_JOINT_NAMES,'robot_asset':str(robot_asset),
        'robot_asset_sha256':hashlib.sha256(robot_asset.read_bytes()).hexdigest(),
        'robot_asset_dependency_sha256':hashlib.sha256(dependency.read_bytes()).hexdigest() if dependency.exists() else None,
        'physics_profile':a.physics_profile,'actual_asset_references':actual_assets,
        'training_contact_profile':a.physics_profile=='isaaclab_training',
        'training_contact_profile_scope':'Only Go2 rigid speed limits and calf/foot material(.8/.6,multiply); floor and head/box materials unchanged' if a.physics_profile=='isaaclab_training' else None,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'default_joint_positions':ctrl.default_joint_positions.tolist(),
        'independent_case_resets':True,'cases':cases,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'No MV7/map/obstacles/contact success claim; identical command scripts and plant for all policies.'}
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2));reports=[]
    for name,segments in cases.items():
        art_root,_=art.get_world_pose();body_root,_=robot.get_world_pose()
        art.set_world_pose(position=art_root+np.array([0,0,.41])-body_root,orientation=np.array([1.,0,0,0]))
        art.set_joint_positions(art.get_joints_default_state().positions);art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3));ctrl.reset()
        limiter=VelocitySlewLimiter();stop=MeasuredStopGate(.4)
        ends=np.cumsum([int(round(s[0]/.005)) for s in segments]);total=int(ends[-1]);rows=[]
        maximum=0.;minimum=1e9;failure_tick=None;stop_start=None;stop_origin=None;stop_done=None;stop_records=[];was_moving=False;last_stop_valid=False
        previous_position=np.asarray(robot.get_world_pose()[0],float).copy()
        reason='completed';simulation_times=[]
        for step in range(total):
            if (a.output.parent/'STOP').exists():reason='resource_stop';break
            segment=int(np.searchsorted(ends,step,side='right'));target=np.asarray(segments[segment][1],float)
            now=float(ctrl._simulation_time());command=limiter.preview(target,now) if a.command_profile=='slew' else target.copy();limiter.accept(command,now)
            if failure_tick is not None:command=np.zeros(3)
            obs,*_=env.step({'move_by_speed':command.tolist()})
            q=np.asarray(obs['orientation'],float);q/=np.linalg.norm(q);position=np.asarray(obs['position'],float)
            diagnostic=ctrl.get_obs();linear=np.asarray(diagnostic['body_lin_vel']);angular=np.asarray(diagnostic['body_ang_vel'])
            current=float(ctrl._simulation_time());simulation_times.append(current)
            tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)));maximum=max(maximum,tilt);minimum=min(minimum,position[2])
            moving=bool(np.any(target))
            if was_moving and not moving:
                stop_start=step*.005;stop_origin=previous_position.copy();stop_done=None;last_stop_valid=False;stop.reset()
            if stop_start is not None:
                if not moving:
                    if np.linalg.norm(command[:2])>=.03 or abs(command[2])>=.06:stop.reset();last_stop_valid=False
                    else:last_stop_valid=stop.update(position,linear,angular,tilt,current)
                    if last_stop_valid and stop_done is None:stop_done=step*.005-stop_start
                if moving or step==total-1 or (failure_tick is not None and step-failure_tick>=256):
                    stop_records.append({'start_s':stop_start,'settled_after_s':stop_done,
                        'still_settled_at_end':bool(last_stop_valid),
                        'displacement_m':float(np.linalg.norm(position[:2]-stop_origin[:2]))});stop_start=None
            was_moving=moving
            previous_position=position.copy()
            if not np.isfinite(position).all() or tilt>=30. or position[2]<.21:
                if failure_tick is None:failure_tick=step;reason='health_guard'
            if step%4==0:
                rows.append([step*.005,*target,*command,*position,*q,*linear,*angular,tilt,
                    *ctrl.joint_subset.get_joint_positions(),*ctrl.joint_subset.get_joint_velocities()])
            if step%200==0:
                tmp=a.output/'live.tmp';tmp.write_text(json.dumps({'case':name,'step':step,'total_steps':total,'maximum_tilt_deg':maximum,'failed':failure_tick is not None}));tmp.replace(a.output/'live.json')
            if failure_tick is not None and step-failure_tick>=256:break
        data=np.asarray(rows,float).reshape(-1,45)
        np.savez_compressed(a.output/(name+'.npz'),trace=data,
            columns=np.array(['time','target_vx','target_vy','target_wz','command_vx','command_vy','command_wz','x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz','tilt']+
                ['position_'+name for name in GO2_JOINT_NAMES]+['velocity_'+name for name in GO2_JOINT_NAMES]))
        intervals=np.diff(simulation_times)
        assert not intervals.size or np.allclose(intervals,.005,rtol=0,atol=1e-6),'Physics cadence changed'
        assert abs(ctrl.policy_inferences-(len(simulation_times)+3)//4)<=1,'Policy cadence changed'
        report={'case':name,'reason':reason,'completed_script':len(simulation_times)==total,'healthy':failure_tick is None and reason=='completed',
            'first_failure_s':None if failure_tick is None else failure_tick*.005,'maximum_tilt_deg':maximum,
            'minimum_base_height_m':minimum if simulation_times else None,'physics_steps':len(simulation_times),'policy_inferences':ctrl.policy_inferences,
            'linear_command_rmse_mps':float(np.sqrt(np.mean((data[:,4:6]-data[:,14:16])**2))) if rows else None,
            'yaw_command_rmse_rad_s':float(np.sqrt(np.mean((data[:,6]-data[:,19])**2))) if rows else None,
            'stops':stop_records,'final_stop_valid':bool(stop_records and stop_records[-1]['still_settled_at_end'])}
        reports.append(report);(a.output/'results.json').write_text(json.dumps(reports,indent=2));print(json.dumps(report),flush=True)
        if reason=='resource_stop':break
    (a.output/'summary.json').write_text(json.dumps({'diagnostic_only':True,'navigation_accepted':False,
        'label':a.label,'cases':reports,'all_cases_healthy_and_stopped':len(reports)==len(cases) and all(r['healthy'] and r['completed_script'] and r['final_stop_valid'] for r in reports)},indent=2))
except BaseException as error:
    failure={'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
        'case':locals().get('name'),'step':locals().get('step'),'diagnostic_only':True,'navigation_accepted':False}
    (a.output/'failure.json').write_text(json.dumps(failure,indent=2))
    print(json.dumps(failure),flush=True)
    raise
finally:
    if env is not None:env.close()
    else:runtime.simulation_app.close()
