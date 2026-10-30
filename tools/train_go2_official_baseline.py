"""Train an isolated Go2 candidate adapted from Unitree RL Lab; not task acceptance."""
from pathlib import Path
import argparse
import copy
import faulthandler
import signal
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--num-envs', type=int, default=32)
p.add_argument('--iterations', type=int, default=5)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--native-lab', type=Path, required=True)
p.add_argument('--robot-asset', type=Path, required=True)
p.add_argument('--resume', type=Path)
a = p.parse_args()
faulthandler.register(signal.SIGUSR1, all_threads=True)
assert a.num_envs >= 16 and a.iterations > 0
a.output.mkdir(parents=True, exist_ok=False)
for extension in ['omni.isaac.lab', 'omni.isaac.lab_assets', 'omni.isaac.lab_tasks']:
    sys.path.insert(0, str(a.native_lab / 'source/extensions' / extension))

def write_json(name, value):
    path = a.output / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)

asset_sha = hashlib.sha256(a.robot_asset.read_bytes()).hexdigest()
assert asset_sha == '75958c8286d2dfdf7617108a685110ac831dfc47f2546e71a8cc2ccaecbd83a5'
source_root = Path(__file__).resolve().parent
source_files = [Path(__file__).resolve(), *sorted((source_root / 'go2_official_training').rglob('*.py')),
                source_root / 'go2_official_training/upstream_manifest.json']
write_json('source_manifest.json', {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_files})
from omni.isaac.lab.app import AppLauncher
app = None
env = None
started = time.monotonic()
try:
    launcher = AppLauncher(headless=True, device=f'cuda:{a.gpu}')
    app = launcher.app
    import torch
    import numpy as np
    from pxr import UsdGeom, PhysxSchema
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    from omni.isaac.lab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner
    from go2_official_training.env_cfg import RobotEnvCfg
    torch.set_num_threads(1)
    cfg = RobotEnvCfg()
    cfg.seed = a.seed
    cfg.scene.num_envs = a.num_envs
    cfg.scene.env_spacing = 3.0
    cfg.scene.replicate_physics = True
    cfg.sim.device = f'cuda:{a.gpu}'
    cfg.sim.dt = .005
    cfg.decimation = 4
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.robot.spawn.usd_path = str(a.robot_asset)
    cfg.scene.robot.init_state.pos = (0., 0., .41)
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.height_scanner = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.curriculum.terrain_levels = None
    cfg.terminations.bad_orientation.params['limit_angle'] = math.radians(30.)
    cfg.commands.base_velocity.debug_vis = False
    # The official actor and critic do not consume the height scanner.
    assert not hasattr(cfg.observations.policy, 'height_scan')
    dump_yaml(str(a.output / 'environment.yaml'), cfg)
    env = ManagerBasedRLEnv(cfg=cfg)
    floor = env.sim.stage.GetPrimAtPath('/World/ground')
    assert floor.IsValid()
    UsdGeom.XformCommonAPI(floor).SetTranslate((0., 0., .01))
    for prim in env.sim.stage.Traverse():
        if str(prim.GetPath()).startswith('/World/ground'):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateTorsionalPatchRadiusAttr(0.)
            api.CreateMinTorsionalPatchRadiusAttr(0.)
    class ClippedWrapper(RslRlVecEnvWrapper):
        def step(self, actions):
            if (a.output.parent / 'STOP').exists():
                raise InterruptedError('Resource supervisor requested a checkpoint stop')
            if not torch.isfinite(actions).all():
                raise FloatingPointError('Non-finite policy action')
            result = super().step(actions.clamp(-100., 100.))
            if not torch.isfinite(result[0]).all() or not torch.isfinite(result[1]).all():
                raise FloatingPointError('Non-finite training observation or reward')
            return result
    wrapped = ClippedWrapper(env)
    obs, extras = wrapped.get_observations()
    assert obs.shape == (a.num_envs, 45), obs.shape
    assert extras['observations']['critic'].shape == (a.num_envs, 60)
    assert wrapped.num_actions == 12
    robot = env.scene['robot']
    pc = env.sim.get_physics_context()
    assert pc.is_gpu_dynamics_enabled() and pc.get_broadphase_type() == 'GPU'
    native_joint_names = list(robot.joint_names)
    actuator = robot.actuators['base_legs']
    nominal = robot.data.default_joint_pos[0].detach().cpu().tolist()
    names_expected = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                      'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
                      'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    assert native_joint_names == names_expected, native_joint_names
    assert len(robot.body_names) == 19
    assert abs(float(actuator.stiffness.flatten()[0]) - 25.) < 1e-6
    assert abs(float(actuator.damping.flatten()[0]) - .5) < 1e-6
    assert abs(float(actuator.effort_limit.flatten()[0]) - 23.5) < 1e-6
    assert abs(float(actuator.velocity_limit.flatten()[0]) - 30.) < 1e-6
    contract = {
        'training_only': True, 'original_task_accepted': False,
        'official_pretrained_policy': False, 'upstream_commit': '4960b84732b0c2ec593dccbfe963fda1bcd7b1e3',
        'isaacsim_version': importlib.metadata.version('isaacsim'),
        'native_lab_version': '1.4.0', 'native_lab_commit': 'dd044a4977722d85e966c5b63949093c9b14667f',
        'rsl_rl_version': importlib.metadata.version('rsl-rl-lib'),
        'actor_dimension': 45, 'critic_dimension': 60, 'action_dimension': 12,
        'observation_terms': env.observation_manager.active_terms,
        'joint_names': native_joint_names, 'default_joint_positions': nominal,
        'body_names': list(robot.body_names), 'asset_sha256': asset_sha,
        'floor_top_m': .01, 'actor_hz': 50, 'physics_hz': 200,
        'gpu_dynamics': pc.is_gpu_dynamics_enabled(), 'broadphase': pc.get_broadphase_type(),
        'action_scale': .25, 'kp': 25., 'kd': .5, 'saturation_torque_nm': 23.5,
        'velocity_limit_rad_s': 30., 'training_mass_randomization_kg': [-1., 3.],
        'training_termination_tilt_deg': 30.,
        'training_reset_scope': 'independent RL episodes only; not navigation rollouts',
    }
    write_json('training_contract.json', contract)
    train_cfg = {
        'seed': a.seed, 'num_steps_per_env': 24, 'save_interval': 100,
        'empirical_normalization': False, 'logger': 'tensorboard',
        'policy': {'class_name': 'ActorCritic', 'init_noise_std': 1.0,
                   'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128],
                   'activation': 'elu'},
        'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1.0, 'use_clipped_value_loss': True,
                      'clip_param': .2, 'entropy_coef': .01, 'num_learning_epochs': 5,
                      'num_mini_batches': 4, 'learning_rate': .001, 'schedule': 'adaptive',
                      'gamma': .99, 'lam': .95, 'desired_kl': .01, 'max_grad_norm': 1.},
    }
    write_json('ppo_config.json', train_cfg)
    class ReportingRunner(OnPolicyRunner):
        def log(self, locs, width=80, pad=35):
            super().log(locs, width, pad)
            metrics = {'iteration': int(locs['it']), 'total_transitions': int(self.tot_timesteps),
                'wall_seconds': time.monotonic() - started,
                'iteration_seconds': locs['collection_time'] + locs['learn_time'],
                'mean_reward': statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
                'mean_episode_steps': statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None,
                'losses': {k: float(v) for k, v in locs['loss_dict'].items()},
                'episode_metrics': {key: float(torch.mean(torch.cat([
                    torch.as_tensor(item[key], device=self.device).reshape(-1)
                    for item in locs['ep_infos'] if key in item])))
                    for key in (locs['ep_infos'][0] if locs['ep_infos'] else [])},
                'command_range': dict(vars(env.command_manager.get_term('base_velocity').cfg.ranges)),
                'training_only': True, 'original_task_accepted': False}
            assert all(math.isfinite(v) for v in metrics['losses'].values())
            write_json('live.json', metrics)
            with (a.output / 'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, default=str) + '\n')
    runner = ReportingRunner(wrapped, copy.deepcopy(train_cfg), log_dir=str(a.output / 'checkpoints'), device=f'cuda:{a.gpu}')
    if a.resume:
        runner.load(str(a.resume))
    before = torch.cat([p.detach().flatten().cpu() for p in runner.alg.policy.actor.parameters()])
    reason = 'iterations_completed'
    try:
        runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
    except InterruptedError:
        reason = 'resource_stop'
        runner.save(str(a.output / 'checkpoints/model_resource_stop.pt'))
    after = torch.cat([p.detach().flatten().cpu() for p in runner.alg.policy.actor.parameters()])
    parameter_change = float(torch.max(torch.abs(after - before)))
    assert parameter_change > 0
    export = torch.jit.script(copy.deepcopy(runner.alg.policy.actor).cpu())
    export_path = a.output / 'actor.pt'
    export.save(str(export_path))
    # Plain deterministic actor checkpoint for the existing navigation loader.
    deployment_checkpoint = a.output / 'actor_checkpoint.pt'
    torch.save({'model_state_dict': {'actor.' + key: value.detach().cpu()
                for key, value in runner.alg.policy.actor.state_dict().items()},
                'observation_profile': 'unitree_go2_45', 'empirical_normalization': False}, deployment_checkpoint)
    contract['actor_sha256'] = hashlib.sha256(export_path.read_bytes()).hexdigest()
    contract['actor_checkpoint_sha256'] = hashlib.sha256(deployment_checkpoint.read_bytes()).hexdigest()
    contract['observation_profile'] = 'unitree_go2_45'
    contract['observation_order'] = ['base_ang_vel', 'projected_gravity', 'velocity_commands',
                                      'joint_pos_rel', 'joint_vel_rel', 'last_action']
    contract['observation_scales'] = [.2, 1., 1., 1., .05, 1.]
    contract['observation_clip_before_scale'] = [-100., 100.]
    contract['empirical_normalization'] = False
    write_json('deployment_contract.json', contract)
    with torch.inference_mode():
        test = torch.zeros((2, 45))
        error = float(torch.max(torch.abs(export(test) - runner.alg.policy.actor(test.to(runner.device)).cpu())))
    assert error < 1e-5
    write_json('summary.json', {'training_only': True, 'original_task_accepted': False,
        'reason': reason, 'training_runtime_passed': reason == 'iterations_completed',
        'requested_iterations': a.iterations, 'num_envs': a.num_envs,
        'max_actor_parameter_change': parameter_change, 'export_equivalence_error': error,
        'actor_sha256': hashlib.sha256(export_path.read_bytes()).hexdigest(),
        'wall_seconds': time.monotonic() - started, 'trained_policy_deployment_accepted': False})
except BaseException as error:
    write_json('failure.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None:
        env.close()
    if app is not None:
        app.close()
