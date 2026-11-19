from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_caseyaw';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,d in json.loads((I/'manifest.json').read_text()).items():assert sha(I/n)==d,n
for p in (I/'source').glob('*.py'):ast.parse(p.read_text())
cfg=json.loads((root/'ground_lift_training_config.json').read_text());old=json.loads((root/'iterations/20261119_commonfix/fixtures/commonfix_ground_lift_config.json').read_text());rows=json.loads((I/'trial_index.json').read_text())
assert cfg['common_initial_joint_positions_rad']==old['common_initial_joint_positions_rad']
assert set(cfg['validated_assets'])==set(old['validated_assets'])|({'case_handheld'} if any(r['verified_success'] for r in rows) else set())
for r in rows:
 assert r['source_sha256'] in {sha(p) for p in (I/'source').glob('caseyaw*trial.py')}
 for key in ['samples.json','review.mp4']:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
 d=json.loads((R/r['result']).read_text());assert d['plan']['object_yaw_deg'] in [0,90,180,270]
 assert d['plan']['q_start']==cfg['common_initial_joint_positions_rad']
for r in json.loads((I/'diagnostic_index.json').read_text()):
 assert r['diagnostic_only'] is True
 assert r['result'] not in {x['result'] for x in rows}
 assert r['source_sha256'] in {sha(p) for p in (I/'source').glob('caseyaw*trial.py')}
 for key in ['samples.json','review.mp4']:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
for a,b in cfg['validated_simulation_baselines'].items():
 arc=root/'iterations'/b['iteration'];assert (arc/'source'/b['harness']).exists()
 p=json.loads((arc/'fixtures'/b['plan_dir']/(a+'.json')).read_text())['plans'][b['candidate']];assert p['q_start']==cfg['common_initial_joint_positions_rad']
print('CASEYAW_ARCHIVE_OK',len(rows),'fresh trials',len(cfg['validated_assets']),'total validated')
