"""Reproduce two experts and action-only replays before starting batch collection."""
from pathlib import Path
import argparse, json, subprocess, sys
import h5py
import numpy as np
R = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=2)
a = p.parse_args()
rows = []
def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=R, check=True)
for i in [1, 2]:
    folder = f'placements/p{i:04d}'
    run(R / 'plan_threecam.py', '--asset', 'box_white', '--index', i)
    trial = R / 'realpose_trials' / folder / 'box_white/candidate0_repeat2'
    replay = R / 'threecam_replay_trials' / folder / 'box_white/candidate0_repeat2'
    if not (trial / 'result.json').exists():
        run(R / 'realpose_box_white_trial.py', '--asset', 'box_white', '--plan-dir', folder, '--gpu', a.gpu, '--repeat', 2)
    for path in [trial, replay]:
        if path == replay and not (path / 'result.json').exists():
            run(R / 'replay_threecam_box_white_trial.py', '--asset', 'box_white', '--plan-dir', folder, '--gpu', a.gpu, '--repeat', 2, '--replay-episode', trial / 'episode.hdf5')
        run(R / 'verify_realpose.py', '--trial-dir', path)
        run(R / 'verify_threecam.py', path)
        assert json.loads((path / 'acceptance.json').read_text())['accepted']
    with h5py.File(trial / 'episode.hdf5') as x, h5py.File(replay / 'episode.hdf5') as y:
        assert np.array_equal(x['action'][:], y['action'][:])
        error = float(np.max(abs(x['observations/qpos'][:] - y['observations/qpos'][:])))
        assert error <= .005
    rows.append(dict(index=i, trial=str(trial), replay=str(replay), actual_qpos_max_difference_rad=error, recorded_actions_equal=True, physical_and_threecam_verified=True))
(R / 'threecam_pilot_verified.json').write_text(json.dumps(rows, indent=2))
print('TWO_THREECAM_PILOTS_VERIFIED')
