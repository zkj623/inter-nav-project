"""Run stationary lift probes with one subprocess per explicitly selected GPU."""
import argparse
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from catalog_support import PACKAGE, REPOSITORY, DEFAULT_CATALOG, input_fingerprint, verify_catalog

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=DEFAULT_CATALOG)
parser.add_argument('--runtime', type=Path, default=REPOSITORY)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--robot-usd', type=Path, default=os.environ.get('R1LITE_USD_PATH'))
parser.add_argument('--collision-deps', type=Path)
parser.add_argument('--only', default='', help='Comma-separated catalog IDs; default: all 20')
parser.add_argument('--gpus', default='0', help='Explicit GPU indices; no CUDA_VISIBLE_DEVICES changes')
parser.add_argument('--timeout', type=int, default=600)
parser.add_argument('--resume', action='store_true', help='Reuse complete results only when source/catalog fingerprints match')
args = parser.parse_args()
verify_catalog(args.root)
if args.robot_usd is None:
    parser.error('Set --robot-usd or R1LITE_USD_PATH')
assets = json.loads((args.root / 'asset_manifest.json').read_text())['assets']
ids = [a['id'] for a in assets]
if args.only:
    selected = args.only.split(',')
    if set(selected) - set(ids):
        parser.error('Unknown catalog IDs: ' + ','.join(sorted(set(selected) - set(ids))))
    ids = [aid for aid in ids if aid in selected]
gpus = list(map(int, args.gpus.split(',')))
if len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus) or args.timeout <= 0:
    parser.error('GPU indices must be distinct and nonnegative; timeout must be positive')
fingerprint = input_fingerprint(args.runtime, args.root, args.robot_usd)
args.output.mkdir(parents=True, exist_ok=True)
for aid in ids:
    out = args.output / aid
    if out.exists():
        result_path = out / 'result.json'
        if not args.resume or not result_path.exists():
            parser.error(f'Existing or partial result: {out}; choose a new output directory')
        result = json.loads(result_path.read_text())
        if result.get('input_fingerprint') != fingerprint:
            parser.error(f'Stale result from different code/catalog: {out}')

def worker(items, gpu):
    for aid in items:
        out = args.output / aid
        if (out / 'result.json').exists():
            continue
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
        if free < 8000:
            raise RuntimeError(f'GPU {gpu}: less than 8000 MiB free; no probe launched')
        cmd = [sys.executable, '-u', str(PACKAGE / 'sdk_launch_guard.py'), str(PACKAGE / 'lift_probe.py'),
               '--root', str(args.root), '--runtime', str(args.runtime), '--output', str(out),
               '--robot-usd', str(args.robot_usd), '--asset', aid, '--gpu', str(gpu)]
        if args.collision_deps:
            cmd += ['--collision-deps', str(args.collision_deps)]
        with (args.output / (aid + '.log')).open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = 124
        if not (out / 'result.json').exists():
            out.mkdir(parents=True, exist_ok=True)
            (out / 'result.json').write_text(json.dumps(dict(asset_id=aid, success=False, reason='PROCESS_ERROR_OR_TIMEOUT',
                exit_code=rc, transport_executed=False, input_fingerprint=fingerprint), indent=2))
        print(json.dumps(json.loads((out / 'result.json').read_text())), flush=True)

with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
    futures = [pool.submit(worker, ids[i::len(gpus)], gpu) for i, gpu in enumerate(gpus)]
    for future in futures:
        future.result()
results = [json.loads((args.output / aid / 'result.json').read_text()) for aid in ids]
(args.output / 'summary.json').write_text(json.dumps(results, indent=2))
