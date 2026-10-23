import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import shutil


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--launch-wrapper', type=Path)
    parser.add_argument('--max-startup-retries', type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.max_startup_retries <= 5:
        raise ValueError('Startup retry budget must be within 0..5')
    sys.path.insert(0, str(args.runtime))
    from grutopia_extension.interaction_clearance.benchmark import digest, runtime_fingerprint
    from grutopia_extension.interaction_clearance.task import ClearanceTask
    from grutopia.demo.r1lite_clearance_multilayout import run_process

    expected = read_json(args.runtime.parent / 'runtime_manifest.json')
    assets = read_json(args.runtime.parent / 'robot_asset_manifest.json')
    skill = args.runtime / 'grutopia_extension/interaction_clearance/skills/composable_clear_blockage.py'
    skill_hash = checksum(skill)

    def assert_frozen():
        assert runtime_fingerprint(args.runtime) == expected
        assert all(checksum(args.runtime.parent / 'robot_snapshot' / relative) == value for relative, value in assets.items())
        assert checksum(skill) == skill_hash

    assert_frozen()
    original = read_json(args.suite)
    cases = [dict(case, task=ClearanceTask.from_dict(case['task']).to_dict()) for case in original['cases']
             if not args.cases or case['case_id'] in args.cases]
    if not cases or args.cases and len(cases) != len(args.cases):
        raise ValueError('Requested case ids missing from suite')
    if args.cases:
        cases.sort(key=lambda case: args.cases.index(case['case_id']))
    args.output.mkdir(parents=True, exist_ok=True)
    suite = {'purpose': original.get('purpose', 'baseline_regression'), 'cases': cases,
             'runtime_sha256': expected['sha256'], 'skill_sha256': skill_hash, 'success_contract': 'clearance_and_robot_outside_passage'}
    suite_path = args.output / 'suite.json'
    if suite_path.exists():
        assert read_json(suite_path) == suite
    else:
        suite_path.write_text(json.dumps(suite, indent=2))
    completed, lock = [], threading.Lock()

    def lane(gpu, jobs):
        for case in jobs:
            assert_frozen()
            case_id = case['case_id']
            summary = args.output / f'{case_id}_summary.json'
            if summary.exists():
                row = read_json(summary)
            else:
                run = args.output / case_id
                if run.exists():
                    raise RuntimeError(f'Incomplete run preserved; use a new output directory: {run}')
                task_file = args.output / f'{case_id}.json'
                task_file.write_text(json.dumps(case['task'], indent=2))
                print(json.dumps({'event': 'episode_begin', 'case_id': case_id, 'gpu': gpu}), flush=True)
                command = [sys.executable, str(args.runtime / 'grutopia/demo/r1lite_interaction_clearance.py'),
                           '--task', str(task_file), '--skill', str(skill), '--output', str(run),
                           '--seed', str(case['task']['seed']), '--gpu', str(gpu)]
                if args.launch_wrapper:
                    command = [sys.executable, str(args.launch_wrapper), *command[1:]]
                startup_attempts = []
                for startup_index in range(args.max_startup_retries + 1):
                    attempt_log = args.output / f'{case_id}_attempt_{startup_index:02d}.log'
                    code, timeout = run_process(command, attempt_log, 1200)
                    startup_attempts.append({'attempt': startup_index, 'exit_code': code, 'timeout': timeout,
                                             'log': str(attempt_log), 'pre_physics_ipc_collision': code == 75})
                    if code != 75 or not args.launch_wrapper:
                        break
                    if run.exists():
                        raise RuntimeError('Pre-physics launcher must not create a physical episode directory')
                shutil.copy2(attempt_log, args.output / f'{case_id}.log')
                result = read_json(run / 'result.json') if (run / 'result.json').exists() else {}
                audit = {}
                if (run / 'physics_collision_samples.json').exists() and (run / 'demonstration.npz').exists():
                    run_process([sys.executable, str(args.runtime / 'grutopia/demo/audit_r1lite_clearance.py'),
                                 '--run', str(run), '--robot-urdf', str(args.runtime.parent / 'robot_snapshot/r1lite_navigation.urdf')],
                                args.output / f'{case_id}_audit.log', 1200)
                    if (run / 'offline_reaudit.json').exists():
                        audit = read_json(run / 'offline_reaudit.json')
                source = read_json(run / 'source_manifest.json') if (run / 'source_manifest.json').exists() else {}
                identity = (digest(result.get('task')) == digest(case['task']) and result.get('skill_sha256') == skill_hash
                            and all(source.get(relative) == value for relative, value in expected['files'].items()))
                infrastructure_failure = not result or code == 75
                row = {'case_id': case_id, 'family': case['family'], 'repeat': case['repeat'],
                       'success': bool(code == 0 and not timeout and result.get('success') and audit.get('passed') and identity),
                       'reason': result.get('reason', 'SIMULATOR_STARTUP_OR_NATIVE_FAILURE'),
                       'details': result.get('skill_details', result.get('error')),
                       'simulated_seconds': result.get('simulated_seconds'), 'metrics': result.get('metrics'),
                       'offline_audit_passed': audit.get('passed', False), 'identity_matches': identity,
                       'collision_violations': audit.get('violations', []), 'timeout': timeout, 'run': str(run),
                       'exit_code': code, 'infrastructure_failure': infrastructure_failure,
                       'startup_attempts': startup_attempts}
                summary.write_text(json.dumps(row, indent=2))
            with lock:
                completed.append(row)
                report = {'completed': len(completed), 'total': len(cases),
                          'successes': sum(item['success'] for item in completed), 'results': completed}
                temporary = args.output / 'progress.tmp'
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(args.output / 'progress.json')
            print(json.dumps({'event': 'episode_complete', 'case_id': case_id, 'success': row['success'],
                              'reason': row['reason'], 'simulated_seconds': row['simulated_seconds']}), flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(lane, gpu, cases[index::len(args.gpus)]) for index, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()
    assert_frozen()
    report = {'completed': len(completed), 'total': len(cases), 'successes': sum(item['success'] for item in completed)}
    report['runtime_sha256'] = expected['sha256']
    report['skill_sha256'] = skill_hash
    report['results'] = sorted(completed, key=lambda row: row['case_id'])
    report['continuous_time_guarantee'] = False
    (args.output / 'round_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'round_complete', 'completed': len(cases), 'successes': report['successes']}), flush=True)


if __name__ == '__main__':
    main()
