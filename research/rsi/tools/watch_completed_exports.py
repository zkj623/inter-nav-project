import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from research.rsi.paths import default_results_root


ROOT = default_results_root()
TOOLS = Path(__file__).resolve().parent
HOST = 'zkj@162.105.195.166'
REMOTE = '/data20t/embodied/zkj/inter-nav-results/r1lite-stability-v4'
BATCHES = ('targeted-regression', 'full-regression', 'development', 'validation', 'repeatability')


def fetch(relative):
    path = f'{REMOTE}/{relative}'
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST,
                             f'if test -f {path}; then cat {path}; fi'], check=True, capture_output=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.results_root.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    if not args.round.replace('-', '').isalnum():
        raise ValueError('Invalid round name')
    local = ROOT / args.round
    local.mkdir(exist_ok=True)
    last_status = None
    while True:
        try:
            current = fetch(f'{args.round}/evaluation_status.json')
            if current is not None:
                (local / 'evaluation_status.json').write_text(json.dumps(current, indent=2))
                identifier = (current['stage'], current['state'])
                if identifier != last_status:
                    print(json.dumps({'event': 'evaluation_status', **current}), flush=True)
                    last_status = identifier
            for batch in BATCHES:
                progress = fetch(f'{args.round}/{batch}/progress.json')
                if progress is None:
                    continue
                validations = local / batch / 'videos'
                downloaded = {path.name.removesuffix('.validation.json').removesuffix('.unverified.json').removesuffix('.no_action.json')
                              for path in validations.glob('*.json')
                              if path.name.endswith(('.validation.json', '.unverified.json', '.no_action.json'))} if validations.exists() else set()
                needed = [row for row in progress['results'] if row['case_id'] not in downloaded]
                if needed:
                    if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
                        raise RuntimeError('Local E drive reserve below 5 GiB; existing data preserved')
                    subprocess.run([sys.executable, str(TOOLS / 'export_verified_cases.py'),
                                    '--batch', f'{args.round}/{batch}', '--results-root', str(ROOT)], check=True)
                report = fetch(f'{args.round}/{batch}/round_report.json')
                if report is not None:
                    (local / batch / 'round_report.json').write_text(json.dumps(report, indent=2))
            if current and current['state'] in ('needs_policy_iteration', 'infrastructure_failure', 'passed_pending_artifact_review'):
                print(json.dumps({'event': 'export_watch_finished', 'stage': current['stage'], 'state': current['state']}), flush=True)
                return
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            print(json.dumps({'event': 'export_retry', 'error': str(error)}), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
