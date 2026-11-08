"""Run an upstream mandatory-clearance task through a bounded Python skill."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import random
import traceback


def snapshot_runtime_sources(root, output):
    from grutopia_extension.interaction_clearance.benchmark import runtime_fingerprint

    expected = runtime_fingerprint(root)['files']
    sources = {root / relative for relative in expected}
    sources.update((root / 'grutopia_extension/interaction_clearance').rglob('*.py'))
    manifest = {}
    for source in sorted(sources):
        relative = source.relative_to(root)
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    if any(manifest.get(relative) != checksum for relative, checksum in expected.items()):
        raise RuntimeError('Runtime source snapshot differs from the frozen dependency manifest')
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', type=Path)
    parser.add_argument('--skill', type=Path)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--object-xy', type=float, nargs=2)
    parser.add_argument('--box-mass', type=float)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from grutopia_extension.interaction_clearance.task import ClearanceTask
    from grutopia_extension.interaction_clearance.skill_runtime import Outcome, SkillContext, load_skill

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Use a new empty output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(args.task.read_text(encoding='utf-8')) if args.task else ClearanceTask().to_dict()
        data['seed'] = args.seed
        if args.object_xy:
            data['object_xy'] = args.object_xy
            data['place_xy'] = [args.object_xy[0], data['place_xy'][1]]
        if args.box_mass is not None:
            data['mass_kg'] = args.box_mass
        task = ClearanceTask.from_dict(data)
        skill_path = args.skill or root / 'grutopia_extension/interaction_clearance/skills/clear_blockage.py'
        skill, skill_hash = load_skill(skill_path)
    except (ValueError, TypeError, KeyError, SyntaxError, OSError) as error:
        rejected = {'success': False, 'reason': 'PREFLIGHT_REJECTED', 'error': str(error),
                    'collision_audit_passed': False, 'training_performed': False}
        (args.output / 'result.json').write_text(json.dumps(rejected, indent=2))
        print(json.dumps(rejected), flush=True)
        return 1
    if args.validate_only:
        report = {'preflight_passed': True, 'task': task.to_dict(), 'skill_sha256': skill_hash,
                  'physics_executed': False, 'training_performed': False}
        (args.output / 'preflight.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return 0
    manifest = snapshot_runtime_sources(root, args.output)
    shutil.copy2(skill_path, args.output / 'executed_skill.py')
    random.seed(task.seed)
    import numpy as np
    np.random.seed(task.seed)
    from isaacsim import SimulationApp

    app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': args.gpu, 'multi_gpu': False})
    environment = None
    recorder = None
    result = {'success': False, 'collision_audit_passed': False, 'training_performed': False}
    try:
        from grutopia_extension.interaction_clearance.environment import ClearanceEnvironment
        from grutopia_extension.interaction_clearance.policies import PickPlacePolicy
        from grutopia_extension.interaction_clearance.primitives import primitive_registry, VERSION
        from grutopia_extension.interaction_clearance.recording import EpisodeRecorder

        environment = ClearanceEnvironment(args.output, task)
        recorder = EpisodeRecorder(args.output, task, skill_hash)
        environment.recorder = recorder
        recorder.snapshot(environment, 'start')
        initial = environment.evaluate()
        recorder.event('episode_begin', {'task': task.to_dict(), 'initial_metrics': initial,
                                         'skill_sha256': skill_hash, 'initialization_physics_samples': len(environment.physics_audit)})

        def verify():
            metrics = environment.evaluate()
            from grutopia_extension.interaction_clearance.motion_geometry import angle_error, quaternion_yaw
            position, quaternion = environment.base.get_world_pose()
            protocol_required = task.return_to_start
            protocol_complete = (not protocol_required or (environment.interaction_state == 'released'
                                 and np.linalg.norm(position[:2] - environment.start_base_position[:2]) < 0.025
                                 and abs(angle_error(quaternion_yaw(quaternion), quaternion_yaw(environment.start_base_quaternion))) < 0.035
                                 and np.linalg.norm(environment.base.get_linear_velocity()[:2]) < 0.02))
            metrics['protocol_required'] = protocol_required
            metrics['protocol_complete'] = bool(protocol_complete)
            metrics.update(environment.robot_passage_metrics())
            success = bool(metrics['success'] and protocol_complete and metrics['robot_clear_of_passage']
                           and environment.interaction_state == 'released'
                           and np.linalg.norm(environment.base.get_linear_velocity()[:2]) < 0.02)
            return Outcome(success, 'ROAD_CLEARED' if success else 'CLEARANCE_NOT_COMPLETE', metrics)

        registry = {'pick_place': PickPlacePolicy(environment), **primitive_registry(environment)}
        context = SkillContext(task, registry, verify,
                               environment.observe, recorder.event, wait=environment.wait)
        outcome = skill(context)
        if not isinstance(outcome, Outcome):
            raise ValueError('Skill must return a policy or verification Outcome')
        final_check = verify()
        metrics = final_check.details
        recorder.snapshot(environment, 'finish')
        collision = environment.save_collision_audit()
        result = {'success': bool(outcome.success and final_check.success and collision['passed']),
                  'reason': outcome.reason, 'skill_details': outcome.details, 'metrics': metrics,
                  'initial_metrics': initial, 'collision_audit_passed': collision['passed'],
                  'minimum_noncontact_clearance_m': collision['minimum_noncontact_clearance_m'],
                  'minimum_wall_clearance_m': collision['minimum_wall_clearance_m'],
                  'maximum_finger_penetration_m': collision['maximum_finger_penetration_m'],
                  'frames': len(recorder.actions), 'physics_samples': len(environment.physics_audit),
                  'task': task.to_dict(), 'skill_sha256': skill_hash, 'policy_version': VERSION,
                  'training_performed': False, 'object_attachment': False, 'object_teleport_during_rollout': False,
                  'scope': 'mandatory_interaction_clearance_only', 'navigation_decision_performed': False,
                  'observation_source': 'simulation_ground_truth', 'continuous_time_guarantee': False}
        result['simulated_seconds'] = len(recorder.actions) * environment.control_dt
        result['actual_physics_seconds'] = float(environment.world.current_time) - environment.control_start_physics_time
        result['physics_clock_schema_version'] = 1
        result['maximum_episode_physics_seconds'] = environment.maximum_episode_physics_seconds
        result['policy_sha256'] = manifest['grutopia_extension/interaction_clearance/policies.py']
        result['primitive_sha256'] = manifest['grutopia_extension/interaction_clearance/primitives.py']
        result['initialization'] = environment.initialization
        result['interaction_state'] = environment.interaction_state
        result['executed_policy_ids'] = list(context.executed_policy_ids)
        recorder.event('episode_end', result)
    except Exception as error:
        traceback.print_exc()
        result.update({'reason': 'PIPELINE_EXCEPTION', 'error': str(error), 'task': task.to_dict()})
        if environment is not None:
            environment.base_twist[:] = 0.0
            environment.save_collision_audit()
    finally:
        if recorder is not None:
            recorder.close(result)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        app.close()
    print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
