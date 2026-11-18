"""Prepare or replay the 2026-11-18 G2 iteration without changing old snapshots."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]


def verify():
    for name, expected in json.loads((HERE / 'manifest.json').read_text()).items():
        path = HERE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Iteration file mismatch: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'prepare', 'run'])
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--assets-from', type=Path)
    parser.add_argument('--collision-deps', type=Path)
    parser.add_argument('--python')
    parser.add_argument('--asset')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    verify()
    if args.action == 'verify':
        print('Iteration SHA256 manifest OK')
        return
    spec = importlib.util.spec_from_file_location('g2_workspace', PACKAGE / 'workspace.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    if args.workspace is None:
        parser.error('--workspace is required')
    if args.action == 'prepare':
        if args.assets_from is None or args.collision_deps is None:
            parser.error('prepare requires --assets-from and --collision-deps')
        base.prepare(args.workspace, args.assets_from, args.collision_deps)
        for src in (HERE / 'source').glob('*.py'):
            target = args.workspace / src.name
            if target.exists():
                raise ValueError('Refusing to replace frozen source: ' + src.name)
            shutil.copy2(src, target)
        for src in (HERE / 'plans').iterdir():
            shutil.copytree(src, args.workspace / src.name)
        print('Prepared', args.workspace)
        return
    if not args.python or not args.asset:
        parser.error('run requires --python and --asset')
    baselines = json.loads((HERE / 'baselines.json').read_text())
    if args.asset not in baselines:
        parser.error('No repeated passing baseline for this asset in this iteration')
    for folder in ['source', 'plans']:
        for src in (HERE / folder).rglob('*'):
            if not src.is_file():
                continue
            target = args.workspace / src.relative_to(HERE / folder)
            if not target.is_file() or base.digest(src) != base.digest(target):
                raise ValueError('Prepared iteration changed: ' + str(target))
    baseline = baselines[args.asset]
    output = args.workspace / baseline['result_directory'] / baseline['plan_dir'] / args.asset / f"candidate{baseline['candidate']}_repeat{args.repeat}"
    if output.exists():
        raise ValueError('Use an unused repeat number: ' + str(output))
    command = [args.python, str(args.workspace / baseline['harness']), '--asset', args.asset,
               '--plan-dir', baseline['plan_dir'], '--candidate', str(baseline['candidate']),
               '--repeat', str(args.repeat), '--gpu', str(args.gpu)]
    subprocess.run(command, cwd=args.workspace, env=base.environment(args.workspace), check=True)
    result = json.loads((output / 'result.json').read_text())
    print(json.dumps({key: result.get(key) for key in ['asset', 'success', 'failure', 'min_hold_lift_m', 'max_hold_tilt_deg']}))
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
