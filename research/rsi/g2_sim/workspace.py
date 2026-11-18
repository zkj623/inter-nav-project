"""Materialize the frozen G2 experiment in a fresh, external workspace."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
RESOURCES = ('generated', 'robot_assets', 'object_assets',
             'strategy15_surface_assets', 'strategy15_surface_proxies', 'vendor')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_snapshot():
    manifest = read(PACKAGE / 'snapshot_manifest.json')
    for name, expected in manifest.items():
        path = PACKAGE / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Snapshot mismatch: ' + name)
    return manifest

def prepare(destination, assets, collision_deps):
    verify_snapshot()
    assets = assets.resolve(strict=True)
    collision_deps = collision_deps.resolve(strict=True)
    missing = [name for name in RESOURCES if not (assets / name).is_dir()]
    if missing:
        raise ValueError('Missing prepared asset directories: ' + ', '.join(missing))
    destination = destination.absolute()
    if PACKAGE.parent.parent.parent in destination.resolve().parents:
        raise ValueError('Create the experiment workspace outside the repository')
    # Refuse reuse: historical result directories must never satisfy a new run.
    destination.mkdir(parents=True, exist_ok=False)
    for path in (PACKAGE / 'snapshot').glob('*.py'):
        shutil.copy2(path, destination / path.name)
    for path in (PACKAGE / 'fixtures').iterdir():
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    for name in RESOURCES:
        (destination / name).symlink_to(assets / name, target_is_directory=True)
    (destination / 'strategy15_logs').mkdir()
    state = {'package': str(PACKAGE), 'assets': str(assets),
             'collision_deps': str(collision_deps),
             'snapshot_sha256': digest(PACKAGE / 'snapshot_manifest.json')}
    (destination / 'g2_workspace.json').write_text(json.dumps(state, indent=2))
    return destination

def environment(workspace):
    state = read(workspace / 'g2_workspace.json')
    if state['snapshot_sha256'] != digest(PACKAGE / 'snapshot_manifest.json'):
        raise ValueError('Workspace belongs to another snapshot; create a fresh workspace')
    for path in (PACKAGE / 'snapshot').glob('*.py'):
        if digest(workspace / path.name) != digest(path):
            raise ValueError('Workspace source changed: ' + path.name)
    catalog = PACKAGE.parent / 'household_catalog'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = os.pathsep.join([str(workspace), state['collision_deps'],
                                      str(catalog), env.get('PYTHONPATH', '')])
    return env

def check(workspace, python, verify_assets):
    verify_snapshot()
    env = environment(workspace)
    if verify_assets:
        source = Path(read(workspace / 'g2_workspace.json')['assets'])
        for entry in read(PACKAGE / 'fixtures/strategy15_asset_manifest.json'):
            path = source / entry['path']
            if not path.is_file() or digest(path) != entry['sha256']:
                raise ValueError('Prepared asset changed: ' + str(path))
    command = ('import importlib.util; import numpy, scipy, trimesh, fcl, cv2; '
               'from pxr import Usd; import cooked_geometry; '
               'assert importlib.util.find_spec("isaacsim"); '
               'from ideal_constraints import Envelope; '
               'assert len(Envelope().k.links)==56; print("G2 dependencies and URDF OK")')
    subprocess.run([python, '-c', command], cwd=workspace, env=env, check=True)
    subprocess.run([python, 'gravity_compensation.py'], cwd=workspace, env=env, check=True)

def run(workspace, python, asset, gpu, repeat):
    verify_snapshot()
    baseline = read(PACKAGE / 'baselines.json')[asset]
    if repeat < 1 or gpu < 0:
        raise ValueError('repeat must be positive and gpu nonnegative')
    env = environment(workspace)
    stem = baseline['script'].removeprefix('grasp_trial_').removesuffix('.py')
    result_path = workspace / (stem + '_trials') / baseline['plan_dir'] / asset / f"candidate{baseline['candidate']}_repeat{repeat}" / 'result.json'
    if result_path.parent.exists():
        raise ValueError('Trial directory already exists; use a new repeat number')
    subprocess.run([python, str(workspace / baseline['script']), '--asset', asset,
                    '--plan-dir', baseline['plan_dir'], '--candidate', str(baseline['candidate']),
                    '--repeat', str(repeat), '--gpu', str(gpu)], cwd=workspace, env=env, check=True)
    # The snapshot executables report physical failure in JSON, not the process code.
    result = read(result_path)
    print(json.dumps({'result': str(result_path), 'success': result['success'],
                      'failure': result.get('failure')}, ensure_ascii=False))
    return 0 if result['success'] else 2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('prepare')
    init.add_argument('--workspace', type=Path, required=True)
    init.add_argument('--assets-from', type=Path, required=True)
    init.add_argument('--collision-deps', type=Path, required=True)
    for name in ('check', 'run'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--workspace', type=Path, required=True)
        cmd.add_argument('--python', default=sys.executable)
        if name == 'check':
            cmd.add_argument('--verify-assets', action='store_true')
        else:
            cmd.add_argument('--asset', choices=sorted(read(PACKAGE / 'baselines.json')), required=True)
            cmd.add_argument('--gpu', type=int, default=0)
            cmd.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.workspace, args.assets_from, args.collision_deps))
    elif args.command == 'check':
        check(args.workspace.resolve(strict=True), args.python, args.verify_assets)
    else:
        return run(args.workspace.resolve(strict=True), args.python, args.asset, args.gpu, args.repeat)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
