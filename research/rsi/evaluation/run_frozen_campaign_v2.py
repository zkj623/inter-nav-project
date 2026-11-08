import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--round', required=True)
    parser.add_argument('--wait-for', type=Path)
    parser.add_argument('--target-cases', nargs='+', default=['left_wide_turn_01', 'long_offset_home_01'])
    parser.add_argument('--development-suite', type=Path)
    parser.add_argument('--validation-suite', type=Path)
    args = parser.parse_args()
    if '/' in args.round or '\\' in args.round or args.round.startswith('.'):
        raise ValueError('Round must be a directory name')
    root = args.root
    round_root = root / args.round
    runtime = round_root / 'runtime'
    driver = round_root / 'evaluation_sources/run_stability_round_v3.py'
    wrapper = round_root / 'evaluation_sources/sdk_launch_guard.py'
    sys.path.insert(0, str(runtime))
    from grutopia_extension.interaction_clearance.task import ClearanceTask

    def status(stage, state, **details):
        row = {'stage': stage, 'state': state, 'details': details,
               'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        temporary = round_root / 'evaluation_status.tmp'
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(round_root / 'evaluation_status.json')
        print(json.dumps(row), flush=True)

    if args.wait_for:
        status('waiting_for_previous_round', 'running', report=str(args.wait_for))
        while not args.wait_for.exists():
            time.sleep(30)
    development = args.development_suite or root / 'suites/development_suite.json'
    validation = args.validation_suite or root / 'suites/validation_suite.json'
    stages = [('targeted_regression', root / 'baseline_suite.json', round_root / 'targeted-regression', args.target_cases),
              ('full_regression', root / 'baseline_suite.json', round_root / 'full-regression', None),
              ('development', development, round_root / 'development', None),
              ('validation', validation, round_root / 'validation', None)]
    summaries = []
    for stage, suite, output, selected in stages:
        status(stage, 'running')
        command = [sys.executable, str(driver), '--runtime', str(runtime), '--suite', str(suite),
                   '--output', str(output), '--gpus', '0', '7', '--launch-wrapper', str(wrapper)]
        if selected:
            command.extend(['--cases', *selected])
        with (round_root / f'{stage}.log').open('a') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        report_path = output / 'round_report.json'
        if code or not report_path.exists():
            status(stage, 'infrastructure_failure', exit_code=code)
            return 1
        report = read_json(report_path)
        summaries.append({'stage': stage, 'report': str(report_path), 'successes': report['successes'], 'total': report['total']})
        (round_root / 'campaign_progress.json').write_text(json.dumps(summaries, indent=2))
        if report['successes'] != report['total']:
            failure_state = 'infrastructure_failure' if any(row.get('infrastructure_failure') for row in report['results']) else 'needs_policy_iteration'
            status(stage, failure_state, successes=report['successes'], total=report['total'],
                   validation_must_be_replaced_if_used_for_optimization=stage == 'validation')
            return 1
        status(stage, 'passed', successes=report['successes'], total=report['total'])
    validation_data = read_json(validation)
    families = sorted({case['family'] for case in validation_data['cases']})
    cases = []
    for family_index, family in enumerate(families):
        reference = next(case for case in validation_data['cases'] if case['family'] == family)
        for repetition in range(3):
            task = replace(ClearanceTask.from_dict(reference['task']), task_id=f'{family}_restart_{repetition:02d}',
                           seed=9400000 + family_index * 1000 + repetition)
            cases.append({'case_id': task.task_id, 'family': family, 'repeat': repetition,
                          'task': task.to_dict(), 'reference_case_id': reference['case_id']})
    restart_suite = round_root / 'restart_suite.json'
    restart_suite.write_text(json.dumps({'purpose': 'independent_simulator_restarts_same_poses', 'cases': cases}, indent=2))
    status('repeatability', 'running')
    with (round_root / 'repeatability.log').open('a') as log:
        code = subprocess.run([sys.executable, str(driver), '--runtime', str(runtime), '--suite', str(restart_suite),
                               '--output', str(round_root / 'repeatability'), '--gpus', '0', '7', '--launch-wrapper', str(wrapper)],
                              stdout=log, stderr=subprocess.STDOUT).returncode
    report_path = round_root / 'repeatability/round_report.json'
    report = read_json(report_path) if report_path.exists() else {}
    if code or report.get('successes') != len(cases):
        status('repeatability', 'needs_policy_iteration', successes=report.get('successes'), total=len(cases))
        return 1
    status('all_physical_evaluation_stages', 'passed_pending_artifact_review',
           baseline=12, development=30, validation=60, restarts=18, summaries=summaries,
           real_robot_or_unbounded_safety_guarantee=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
