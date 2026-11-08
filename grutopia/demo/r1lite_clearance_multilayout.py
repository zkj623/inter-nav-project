"""Evaluate one immutable skill on every case in a fixed split, with GPU lanes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from grutopia_extension.interaction_clearance.benchmark import aggregate_results, digest, runtime_fingerprint, suite_cases, validate_suite
from grutopia_extension.interaction_clearance.skill_runtime import load_skill


def run_process(command, log, timeout):
    with Path(log).open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return process.wait(), True
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return process.returncode, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--split', choices=('train', 'validation', 'heldout'), default='train')
    parser.add_argument('--skill', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[7])
    parser.add_argument('--episode-timeout', type=int, default=900)
    parser.add_argument('--fingerprint-only', action='store_true')
    parser.add_argument('--expected-runtime')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    frozen = runtime_fingerprint(root)
    if args.fingerprint_only:
        print(json.dumps(frozen), flush=True)
        return 0
    if not args.suite or not args.skill or not args.output:
        parser.error('--suite, --skill, and --output are required')
    if args.expected_runtime and args.expected_runtime != frozen['sha256']:
        raise SystemExit('Runtime changed before evaluation; refusing incomparable episodes')
    if len(set(args.gpus)) != len(args.gpus) or len(args.gpus) > 4 or any(gpu < 0 for gpu in args.gpus):
        parser.error('Use 1..4 distinct GPU IDs')
    if not 30 <= args.episode_timeout <= 3600:
        parser.error('Episode timeout must be 30..3600 seconds')
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Use a new empty evaluation directory')
    suite = validate_suite(json.loads(args.suite.read_text(encoding='utf-8')))
    cases = suite_cases(suite, args.split)
    _, skill_hash = load_skill(args.skill)
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = args.output / 'candidate.py'
    candidate.write_bytes(args.skill.read_bytes())
    (args.output / 'runtime_manifest.json').write_text(json.dumps(frozen, indent=2))
    runner = Path(__file__).with_name('r1lite_interaction_clearance.py')
    auditor = Path(__file__).with_name('audit_r1lite_clearance.py')

    def lane(gpu, jobs):
        completed_jobs = []
        for case in jobs:
            if runtime_fingerprint(root) != frozen:
                raise RuntimeError('Runtime changed while evaluating the frozen candidate')
            run = args.output / case['case_id']
            task_path = args.output / f'{case["case_id"]}.json'
            task_path.write_text(json.dumps(case['task'], indent=2))
            command = [sys.executable, str(runner), '--task', str(task_path), '--skill', str(candidate),
                       '--output', str(run), '--seed', str(case['task']['seed']), '--gpu', str(gpu)]
            returncode, timed_out = run_process(command, args.output / f'{case["case_id"]}.log', args.episode_timeout)
            path = run / 'result.json'
            try:
                result = json.loads(path.read_text()) if path.exists() else {}
            except (OSError, ValueError):
                result = {}
            audit = {}
            if returncode == 0 and result.get('success') and not timed_out:
                audit_code, audit_timeout = run_process([sys.executable, str(auditor), '--run', str(run)],
                                                       args.output / f'{case["case_id"]}_audit.log', args.episode_timeout)
                if audit_code == 0 and not audit_timeout and (run / 'offline_reaudit.json').exists():
                    audit = json.loads((run / 'offline_reaudit.json').read_text())
            from grutopia_extension.interaction_clearance.task import ClearanceTask
            task_matches = digest(result.get('task')) == digest(ClearanceTask.from_dict(case['task']).to_dict())
            skill_matches = result.get('skill_sha256') == skill_hash and hashlib.sha256(candidate.read_bytes()).hexdigest() == skill_hash
            policy_matches = result.get('policy_sha256') == frozen['files']['grutopia_extension/interaction_clearance/policies.py']
            primitive_matches = result.get('primitive_sha256') == frozen['files']['grutopia_extension/interaction_clearance/primitives.py']
            snapshot_matches = False
            if (run / 'source_manifest.json').exists():
                source_manifest = json.loads((run / 'source_manifest.json').read_text())
                snapshot_matches = all(source_manifest.get(name) == expected for name, expected in frozen['files'].items())
            success = bool(returncode == 0 and not timed_out and result.get('success') and result.get('collision_audit_passed')
                           and audit.get('passed') and task_matches and skill_matches and policy_matches and primitive_matches and snapshot_matches)
            safety_failure = bool(timed_out or not result.get('collision_audit_passed') or
                                  (result.get('success') and not audit.get('passed')))
            reason = ('EPISODE_TIMEOUT' if timed_out else result.get('reason', 'MISSING_RESULT'))
            if result.get('success') and not (task_matches and skill_matches and policy_matches and primitive_matches and snapshot_matches):
                reason, safety_failure = 'EVALUATION_IDENTITY_MISMATCH', True
            elif result.get('success') and not audit.get('passed'):
                reason = 'OFFLINE_AUDIT_REJECTED'
            events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()] if (run / 'events.jsonl').exists() else []
            record = {'case_id': case['case_id'], 'layout_id': case['layout_id'], 'repeat': case['repeat'],
                      'success': success, 'reason': reason, 'safety_failure': safety_failure, 'returncode': returncode,
                      'simulated_seconds': result.get('simulated_seconds', 0.0),
                      'policy_calls': sum(event.get('event') == 'policy_begin' for event in events),
                      'metrics': result.get('metrics', {}), 'policy_details': result.get('skill_details', {}),
                      'offline_audit_passed': bool(audit.get('passed')), 'task_matches': task_matches,
                      'skill_matches': skill_matches, 'policy_matches': policy_matches, 'source_snapshot_matches': snapshot_matches,
                      'primitive_matches': primitive_matches, 'initialization': result.get('initialization'),
                      'minimum_noncontact_clearance_m': result.get('minimum_noncontact_clearance_m'),
                      'minimum_wall_clearance_m': result.get('minimum_wall_clearance_m'), 'gpu': gpu, 'run': str(run)}
            (args.output / f'{case["case_id"]}_summary.json').write_text(json.dumps(record, indent=2))
            completed_jobs.append(record)
            print(json.dumps({'event': 'case_complete', **record}), flush=True)
        return completed_jobs

    results = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(lane, gpu, cases[index::len(args.gpus)]) for index, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            results.extend(future.result())
    if runtime_fingerprint(root) != frozen:
        raise RuntimeError('Runtime changed; refusing candidate selection')
    results.sort(key=lambda item: item['case_id'])
    summary = {'suite_id': suite['suite_id'], 'suite_sha256': digest(suite), 'split': args.split,
               'skill_sha256': skill_hash, 'runtime_sha256': frozen['sha256'], 'backend': 'isaac_physx',
               'metrics': aggregate_results(cases, results), 'results': results, 'complete': True}
    (args.output / 'evaluation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'event': 'evaluation_complete', 'metrics': summary['metrics']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
