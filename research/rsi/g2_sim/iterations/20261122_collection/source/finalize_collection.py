"""Freeze only a complete, quiescent 12 x 20 independently verified dataset."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, os
import h5py
import numpy as np

R = Path(__file__).resolve().parent
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def finalize():
    ds = R / 'dataset'
    assert not (ds / 'frozen_manifest.jsonl').exists(), 'Already frozen; never overwrite'
    cfg = json.loads((R / 'ground_lift_training_config.json').read_text())
    assets = list(json.loads((R / 'standby_preparation_config.json').read_text())['execution'])
    progress = json.loads((R / 'collection_progress.json').read_text())
    assert not progress['running'], 'Primary scheduler has not stopped'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try:
            if (proc / 'cwd').resolve() != R: continue
            command = (proc / 'cmdline').read_bytes().split(b'\0')
            assert not any(x.endswith(b'_trial.py') or x.endswith(b'run_collection_extra.py') for x in command), 'Collection process still running'
        except (FileNotFoundError, PermissionError): pass
    src = ds / 'manifest.jsonl'
    original = src.read_bytes()
    rows = [json.loads(x) for x in original.splitlines() if x]
    assert len(rows) == 240 and len({x['group_id'] for x in rows}) == 240
    assert collections.Counter(x['asset'] for x in rows) == {a: 20 for a in assets}
    provenance = json.loads((R / 'collection_provenance.json').read_text())
    for name, sha in provenance['configs'].items(): assert digest(R / name) == sha, name
    for entry in provenance['assets']: assert digest(R / entry['relative_path']) == entry['sha256'], entry['relative_path']
    coverage = {}
    for a in assets:
        selected = [x for x in rows if x['asset'] == a]
        assert collections.Counter(x['split'] for x in selected) == {'train':16,'validation':2,'test':2}
        poses = [(x['dx_m'], x['dy_m'], x['dyaw_deg']) for x in selected]
        assert len(set(poses)) == 20, a + ': duplicate commanded placements'
        values = np.asarray(poses)
        coverage[a] = dict(count=20, commanded_dx_m=[float(values[:,0].min()),float(values[:,0].max())], commanded_dy_m=[float(values[:,1].min()),float(values[:,1].max())], commanded_dyaw_deg=[float(values[:,2].min()),float(values[:,2].max())], frames=sum(x['frames'] for x in selected))
    total_bytes = 0
    for row in rows:
        D = Path(row['raw_trial']); episode = ds / row['episode']
        verification = json.loads((D / 'verification.json').read_text())
        metadata = json.loads((D / 'metadata.json').read_text())
        assert verification['audit_version'] >= 2 and verification['verified_success']
        assert verification['episode_sha256'] == row['sha256'] == digest(episode), str(episode)
        assert metadata['placement']['group_id'] == row['group_id']
        assert digest(R / 'source_versions' / (row['source_sha256']+'.py')) == row['source_sha256']
        with h5py.File(episode, 'r') as f:
            assert f.attrs['success'] and f.attrs['independently_verified']
            assert json.loads(f.attrs['joint_names']) == cfg['policy_action_joint_names']
            assert f['action'].shape == (row['frames'],16)
            assert f['phase'].asstr()[-1] == 'hold'
        total_bytes += episode.stat().st_size
    assert src.read_bytes() == original, 'Manifest changed during freeze'
    summary = dict(complete=True, frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), episodes=240, split_counts=dict(collections.Counter(x['split'] for x in rows)), total_frames=sum(x['frames'] for x in rows), total_bytes=total_bytes, coverage=coverage, manifest_sha256=hashlib.sha256(original).hexdigest(), immutable_input_files_checked=len(provenance['assets']), scope='Fixed low robot pose, varied commanded object SE2 placements; 1kg objects; ideal fixed torso; no ACT training', warnings=['Small successful placement neighborhoods; inspect per-object coverage before claiming generalization', 'Development attempts include harness fixes and adaptive sampling; counts are not an unbiased expert success-rate benchmark'])
    (ds / 'frozen_manifest.jsonl').write_bytes(original)
    (ds / 'frozen_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

if __name__ == '__main__': finalize()
