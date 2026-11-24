"""Check the fresh bootstrap, exact sources, common home and accepted drive profiles."""
from pathlib import Path
import ast, hashlib, json
from threecam_drive_profile import profile_for_trial
R = Path(__file__).resolve().parent
P = R.parent / 'g2-publish-20261118/research/rsi/g2_sim/iterations/20261124_threecam_collection'
B = R.parent / 'g2-threecam-bootstrap-check-20261124'
manifest = json.loads((P / 'results/current_source_manifest.json').read_text())
for name, sha in manifest.items():
    ast.parse((R / name).read_text())
    assert hashlib.sha256((P / 'source' / name).read_bytes()).hexdigest() == sha
cfg = json.loads((R / 'standby_preparation_config.json').read_text())
home = None
for asset, e in cfg['execution'].items():
    plan = json.loads((R / e['plan_dir'] / (asset + '.json')).read_text())['plans'][e['candidate']]
    if home is None:
        home = plan['q_start']
    assert home == plan['q_start'], asset + ' has a different common home'
original = json.loads((R / 'placements/p0001/basket_rectangular.json').read_text())
bootstrapped = json.loads((B / 'placements/p0001/basket_rectangular.json').read_text())
assert original == bootstrapped, 'Fresh bootstrap generated a different placement plan'
profiles = {}
for f in (R / 'collection_state').glob('*.json'):
    state = json.loads(f.read_text())
    for row in state['accepted']:
        profile = profile_for_trial(row['trial'])
        asset = row['asset']
        if asset in profiles:
            assert profiles[asset] == profile['profile_id']
        profiles[asset] = profile['profile_id']
report = dict(verified=True, source_files=len(manifest), common_home_assets=len(cfg['execution']),
              bootstrap_plan_exact=True, accepted_drive_profile_ids=profiles,
              scope='Source and bootstrap checks plus currently accepted samples; not a 240-episode completion report')
(R / 'threecam_archive_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
