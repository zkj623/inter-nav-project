"""Compare native Unitree observations/actions with the production Go2 adapter."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--policy', type=Path, required=True)
p.add_argument('--robot-asset', type=Path, required=True)
p.add_argument('--label', default='unitree45-native-interface')
p.add_argument('--command-profile', default='slew')
a = p.parse_args()
a.output.mkdir(exist_ok=False)
for extension in ['omni.isaac.lab', 'omni.isaac.lab_assets', 'omni.isaac.lab_tasks']:
    sys.path.insert(0, str(BASE / 'native-reference/IsaacLab-v1.4.0/source/extensions' / extension))
sys.path.insert(0, str(ROOT))
from omni.isaac.lab.app import AppLauncher
app = None
env = None
try:
    app = AppLauncher(headless=True, device=f'cuda:{a.gpu}').app
    import numpy as np
    import torch
    from pxr import UsdGeom, PhysxSchema
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
    from omni.isaac.lab.utils import configclass
    import omni.isaac.lab.envs.mdp as mdp
    from go2_official_training.env_cfg import RobotEnvCfg
    from grutopia_extension.controllers.go2_policy import load_go2_actor, build_go2_observation, rotate_vector_inverse_wxyz
    from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
    torch.set_num_threads(1)
    actor = load_go2_actor(str(a.policy))
    assert actor[0].in_features == 45
    contract = actor.go2_deployment_contract
    assert contract['asset_sha256'] == hashlib.sha256(a.robot_asset.read_bytes()).hexdigest()
    cfg = RobotEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 3.
    cfg.sim.device = f'cuda:{a.gpu}'
    cfg.sim.dt = .005
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.episode_length_s = 10000.
    cfg.scene.robot.spawn.usd_path = str(a.robot_asset)
    cfg.scene.robot.init_state.pos = (0., 0., .41)
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.height_scanner = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.commands.base_velocity.resampling_time_range = (10000., 10000.)
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.debug_vis = False
    @configclass
    class EvaluationEvents:
        reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode='reset')
        leg_material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode='startup', params={
            'asset_cfg': SceneEntityCfg('robot', body_names=['.*_calf', '.*_foot']),
            'static_friction_range': (.8, .8), 'dynamic_friction_range': (.6, .6),
            'restitution_range': (0., 0.), 'num_buckets': 1})
    @configclass
    class EvaluationTerminations:
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    cfg.events = EvaluationEvents()
    cfg.terminations = EvaluationTerminations()
    env = ManagerBasedRLEnv(cfg=cfg)
    floor = env.sim.stage.GetPrimAtPath('/World/ground')
    UsdGeom.XformCommonAPI(floor).SetTranslate((0., 0., .01))
    for prim in env.sim.stage.Traverse():
        if str(prim.GetPath()).startswith('/World/ground'):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateTorsionalPatchRadiusAttr(0.)
            api.CreateMinTorsionalPatchRadiusAttr(0.)
    env.reset()
    robot = env.scene['robot']
    view = robot.root_physx_view
    assert list(robot.joint_names) == contract['joint_names']
    assert len(robot.body_names) == 19
    np.testing.assert_allclose(robot.data.default_joint_pos[0].cpu().numpy(), contract['default_joint_positions'], atol=1e-6)
    pc = env.sim.get_physics_context()
    assert pc.is_gpu_dynamics_enabled() and pc.get_broadphase_type() == 'GPU'
    masses = view.get_masses().cpu().numpy()
    assert abs(float(masses.sum()) - 15.019) < .01
    expected_obs_terms = contract['observation_order']
    assert env.observation_manager.active_terms['policy'] == expected_obs_terms
    actual_step = env.sim.step
    all_physics = []
    current_case = 'startup'
    class HealthStop(Exception):
        pass
    def checked_step(*args, **kwargs):
        result = actual_step(*args, **kwargs)
        transform = view.get_root_transforms()[0].cpu().numpy()
        q = transform[3:][[3, 0, 1, 2]]
        q = q / np.linalg.norm(q)
        tilt = math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1))))
        row = {'sample': len(all_physics), 'case': current_case, 'tilt_deg': tilt,
               'position': transform[:3].tolist(), 'orientation': q.tolist()}
        all_physics.append(row)
        if tilt >= 30. or transform[2] < .21:
            raise HealthStop('all_physics_health_guard')
        return result
    env.sim.step = checked_step
    segments = [
        ('startup', 5., [0., 0., 0.]), ('forward', 4., [.5, 0., 0.]),
        ('forward_stop', 10., [0., 0., 0.]), ('left', 3., [0., .35, 0.]),
        ('left_stop', 10., [0., 0., 0.]), ('positive_yaw', 1., [0., 0., .6]),
        ('positive_yaw_stop', 10., [0., 0., 0.]), ('negative_yaw', 1., [0., 0., -.6]),
        ('negative_yaw_stop', 10., [0., 0., 0.]),
    ]
    limiter = VelocitySlewLimiter()
    command_term = env.command_manager.get_term('base_velocity')
    rows = []
    max_obs = 0.
    max_action = 0.
    reason = 'completed'
    tick = 0
    try:
        for current_case, duration, target in segments:
            for _ in range(round(duration * 50)):
                if (a.output.parent / 'STOP').exists():
                    raise InterruptedError('resource_stop')
                now = tick * .02
                command = limiter.preview(target, now)
                limiter.accept(command, now)
                command_term.vel_command_b[0] = torch.tensor(command, device=env.device)
                native_obs = env.observation_manager.compute()['policy'][0].cpu().numpy()
                transform = view.get_root_transforms()[0].cpu().numpy()
                velocity = view.get_root_velocities()[0].cpu().numpy()
                q = transform[3:][[3, 0, 1, 2]]
                angular = rotate_vector_inverse_wxyz(q, velocity[3:])
                gravity = rotate_vector_inverse_wxyz(q, [0., 0., -1.])
                packed = build_go2_observation(rotate_vector_inverse_wxyz(q, velocity[:3]), angular, gravity, command,
                    view.get_dof_positions()[0].cpu().numpy(), contract['default_joint_positions'],
                    view.get_dof_velocities()[0].cpu().numpy(), env.action_manager.action[0].cpu().numpy(), 45)
                obs_error = float(np.max(np.abs(native_obs-packed)))
                with torch.inference_mode():
                    native_action = actor(torch.from_numpy(native_obs).unsqueeze(0))
                    packed_action = actor(torch.from_numpy(packed).unsqueeze(0))
                action_error = float(torch.max(torch.abs(native_action-packed_action)))
                max_obs = max(max_obs, obs_error)
                max_action = max(max_action, action_error)
                assert obs_error <= 2e-5 and action_error <= 2e-5, (obs_error, action_error)
                obs, _, terminated, truncated, _ = env.step(native_action.to(env.device).clamp(-100., 100.))
                assert not terminated.any() and not truncated.any(), 'Unexpected automatic reset in evaluation'
                rows.append({'time_s': (tick+1)*.02, 'case': current_case, 'command': command.tolist(),
                    'native_observation': native_obs.tolist(), 'packed_observation': packed.tolist(),
                    'native_action': native_action[0].tolist(), 'observation_max_difference': obs_error,
                    'action_max_difference': action_error, 'position': robot.data.root_pos_w[0].cpu().tolist(),
                    'orientation': robot.data.root_quat_w[0].cpu().tolist(),
                    'linear_body': robot.data.root_lin_vel_b[0].cpu().tolist(),
                    'angular_body': robot.data.root_ang_vel_b[0].cpu().tolist()})
                tick += 1
                if tick % 50 == 0:
                    (a.output/'live.json').write_text(json.dumps({'case': current_case, 'time_s': tick*.02,
                        'max_observation_difference': max_obs, 'max_action_difference': max_action,
                        'max_tilt_deg': max(row['tilt_deg'] for row in all_physics)}))
    except (HealthStop, InterruptedError) as error:
        reason = str(error)
    finally:
        env.sim.step = actual_step
        (a.output/'control_trace.json').write_text(json.dumps(rows))
        (a.output/'all_physics_trace.json').write_text(json.dumps(all_physics))
    startup_max = max((row['tilt_deg'] for row in all_physics if row['case']=='startup'), default=999.)
    report = {'diagnostic_only': True, 'original_task_accepted': False, 'reason': reason,
        'interface_matches_recorded_states': max_obs<=2e-5 and max_action<=2e-5 and bool(rows),
        'complete_interface_probe_passed': reason=='completed' and startup_max<15.,
        'maximum_observation_difference': max_obs, 'maximum_action_difference': max_action,
        'all_physics_max_tilt_deg': max((row['tilt_deg'] for row in all_physics), default=999.),
        'startup_max_tilt_deg': startup_max, 'policy_steps': tick, 'all_physics_samples': len(all_physics),
        'native_joint_names': list(robot.joint_names), 'body_mass_kg': float(masses.sum()),
        'automatic_rollout_resets': 0, 'policy_sha256': hashlib.sha256(a.policy.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'One native flat-floor run checks deployment observation/action equivalence and health. Not original T04/T05 or contact acceptance.'}
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
except BaseException as error:
    (a.output/'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error),
                                                   'traceback': traceback.format_exc()}, indent=2))
    raise
finally:
    if env is not None:
        env.close()
    if app is not None:
        app.close()
