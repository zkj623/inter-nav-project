from pathlib import Path
import hashlib,json,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_commonfix';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,d in json.loads((I/'manifest.json').read_text()).items():assert sha(I/n)==d,n
sources={sha(p) for p in (I/'source').glob('*.py')}
for p in (I/'source').glob('*.py'):ast.parse(p.read_text())
rows=json.loads((I/'trial_index.json').read_text());cfg=json.loads((root/'ground_lift_training_config.json').read_text());old=json.loads((root/'iterations/20261119_commonstart/fixtures/commonstart_ground_lift_config.json').read_text())
assert cfg['common_initial_joint_positions_rad']==old['common_initial_joint_positions_rad']
assert set(cfg['validated_assets'])==set(old['validated_assets'])|{r['asset'] for r in rows if r['verified_success']}
for r in rows:
 assert r['source_sha256'] in sources
 for key in ['samples.json','review.mp4']:
  if key in r:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
for a,b in cfg['validated_simulation_baselines'].items():
 arc=root/'iterations'/b['iteration'];assert (arc/'source'/b['harness']).exists()
 p=json.loads((arc/'fixtures'/b['plan_dir']/(a+'.json')).read_text())['plans'][b['candidate']];assert p['q_start']==cfg['common_initial_joint_positions_rad']
assert len(set(cfg['validated_assets']+cfg['pending_adaptation_assets']))==12
print('ARCHIVE_OK',len(rows),'trials',len(cfg['validated_assets']),'passed')
