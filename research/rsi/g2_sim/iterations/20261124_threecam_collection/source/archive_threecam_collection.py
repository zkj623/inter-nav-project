"""Archive executable sources and small evidence; never add image datasets to Git."""
from pathlib import Path
import ast, hashlib, json, shutil
R = Path(__file__).resolve().parent
P = R.parent / 'g2-publish-20261118/research/rsi/g2_sim'
I = P / 'iterations/20261124_threecam_collection'
for name in ['source', 'fixtures', 'results']:
    (I / name).mkdir(parents=True, exist_ok=True)
files = ['threecam_workspace.py', 'threecam_recorder.py', 'collection_recorder.py',
         'plan_threecam.py', 'verify_threecam.py', 'verify_realpose.py',
         'run_threecam_collection.py', 'start_threecam_after_gate.py',
         'finalize_threecam_dataset.py', 'watch_threecam_freeze.py',
         'act_threecam_dataset.py', 'threecam_drive_profile.py',
         'prepare_threecam_grid.py', 'threecam_snapshot.py',
         'replay_threecam_box_white_trial.py', 'compare_threecam_replays.py',
         'archive_threecam_collection.py', 'validate_threecam_pilots.py', 'review_threecam_samples.py', 'check_threecam_archive.py']
assets = list(json.loads((R / 'standby_preparation_config.json').read_text())['execution'])
files += [f'realpose_{a}_trial.py' for a in assets]
manifest = {}
for name in files:
    f = R / name
    ast.parse(f.read_text())
    shutil.copy2(f, I / 'source' / name)
    manifest[name] = hashlib.sha256(f.read_bytes()).hexdigest()
for name in ['official_wrist_config.json', 'ground_lift_training_config.json', 'standby_preparation_config.json']:
    shutil.copy2(R / name, I / 'fixtures' / name)
versions = I / 'fixtures/source_versions'
versions.mkdir(exist_ok=True)
hashes = set(json.loads((R / 'collection_source_manifest.json').read_text()).values())
hashes.update(manifest.values())
for base in ['realpose_trials/placements', 'threecam_replay_trials/placements']:
    for result in (R / base).glob('*/*/candidate*/result.json'):
        hashes.add(json.loads(result.read_text())['source_sha256'])
for h in hashes:
    src = R / 'source_versions' / (h + '.py')
    if src.exists():
        assert hashlib.sha256(src.read_bytes()).hexdigest() == h
        shutil.copy2(src, versions / src.name)
for name in ['threecam_pilot_verified.json', 'collection_source_manifest.json',
             'planning_grid_report.json', 'collection_progress.json', 'threecam_archive_verification.json']:
    if (R / name).exists():
        shutil.copy2(R / name, I / 'results' / name)
for base in ['collection_state', 'dataset']:
    for f in (R / base).glob('*'):
        if f.is_file() and f.suffix in ['.json', '.jsonl']:
            dst = I / 'results' / base / f.name
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(f, dst)
gate = json.loads((R / 'threecam_pilot_verified.json').read_text())
# Preserve complete pilot acceptance, metadata and replay results, excluding media.
for base in ['realpose_trials/placements', 'threecam_replay_trials/placements']:
    for index in [1, 2]:
        trial = R / base / f'p{index:04d}/box_white/candidate0_repeat2'
        for name in ['result.json', 'metadata.json', 'verification.json', 'acceptance.json']:
            f = trial / name
            if f.exists():
                dst = I / 'results/pilots' / ('replay' if base.startswith('threecam') else 'expert') / str(index) / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
shutil.copy2(R / 'README_COLLECTION.md', I / 'README.md')
(I / 'results/current_source_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(archive=str(I), sources=len(manifest), archived_source_versions=len(list(versions.glob('*.py')))), indent=2))
