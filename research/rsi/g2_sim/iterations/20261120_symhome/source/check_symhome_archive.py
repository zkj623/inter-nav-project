from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_symhome';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in json.loads((I/'manifest.json').read_text()).items():assert sha(I/n)==h,n
for f in (I/'source').glob('*.py'):ast.parse(f.read_text())
cfg=json.loads((root/'ground_lift_training_config.json').read_text());rows=json.loads((I/'trial_index.json').read_text());search=json.loads((I/'fixtures/symhome_search.json').read_text());assert cfg['common_initial_joint_positions_rad']==search['candidates'][0]['q']
assert cfg['default_joint_positions_rad']==search['default']
assert cfg['assumptions']['ideal_torso_drive'] and cfg['assumptions']['original_torso_effort_transition_verified'] is False
assert json.loads((I/'fixtures/symhome_original_effort_audit.json').read_text())['max_ratio']>1
assert set(cfg['validated_assets'])=={r['asset'] for r in rows if r['verified_success']}
for r in rows:
 assert r['source_sha256'] in {sha(p) for p in (I/'source').glob('symhome_v4*trial.py')}
 for key in ['samples.json','review.mp4']:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
for f in (I/'fixtures/symhome_plans').glob('*.json'):
 for p in json.loads(f.read_text())['plans']:
  assert p['q_start']==cfg['common_initial_joint_positions_rad']
  assert p['q_default']==cfg['default_joint_positions_rad']
  assert p['q_prepare']==cfg['preparation_trajectory']['waypoints']
print('SYMHOME_ARCHIVE_OK',len(rows),len(cfg['validated_assets']))
