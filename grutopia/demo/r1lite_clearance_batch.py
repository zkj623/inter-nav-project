"""Evaluate a frozen skill/policy version on reproducible clearance configurations."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from grutopia_extension.interaction_clearance.task import ClearanceTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skill', type=Path)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes > 100:
        raise SystemExit('Episode count must be between 1 and 100')
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Use a new empty batch directory')
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('r1lite_interaction_clearance.py')
    results = []
    for index in range(args.episodes):
        seed = args.seed + index
        generator = random.Random(seed)
        data = ClearanceTask(seed=seed).to_dict()
        data['object_xy'] = [0.65 + generator.uniform(-0.0075, 0.0075), generator.uniform(-0.005, 0.005)]
        data['place_xy'] = [data['object_xy'][0], 0.70]
        data['mass_kg'] = generator.uniform(1.2, 1.5)
        task_path = args.output / f'task_{index:03d}.json'
        task_path.write_text(json.dumps(data, indent=2))
        destination = args.output / f'episode_{index:03d}'
        command = [sys.executable, str(runner), '--task', str(task_path), '--output', str(destination),
                   '--seed', str(seed), '--gpu', str(args.gpu)]
        if args.skill:
            command.extend(['--skill', str(args.skill.resolve())])
        with (args.output / f'episode_{index:03d}.log').open('wb') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        path = destination / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {'success': False, 'reason': 'MISSING_RESULT'}
        results.append({'seed': seed, 'task': data, 'run': str(destination), 'returncode': completed.returncode,
                        'success': bool(result.get('success') and completed.returncode == 0),
                        'reason': result.get('reason'), 'collision_audit_passed': result.get('collision_audit_passed', False),
                        'skill_sha256': result.get('skill_sha256'), 'policy_version': result.get('policy_version')})
        summary = {'episodes': len(results), 'successes': sum(item['success'] for item in results),
                   'success_rate': sum(item['success'] for item in results) / len(results),
                   'results': results, 'scope': 'small_parameter_variations_not_generalization_certification'}
        (args.output / 'batch_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(results[-1]), flush=True)
    return 0 if all(item['success'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
