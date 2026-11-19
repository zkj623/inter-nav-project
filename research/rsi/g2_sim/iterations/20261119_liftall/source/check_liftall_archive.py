from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_liftall'
manifest=json.loads((I/'manifest.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((I/name).read_bytes()).hexdigest()==digest,name
sources={hashlib.sha256(p.read_bytes()).hexdigest():p for p in (I/'source').glob('*.py')}
for p in sources.values():ast.parse(p.read_text())
rows=json.loads((I/'trial_index.json').read_text());config=json.loads((root/'ground_lift_training_config.json').read_text())
for r in rows:
 assert r['source_sha256'] in sources,r['asset']
 for field in ['review.mp4','samples.json']:
  if field in r:assert hashlib.sha256(Path(r[field]['server_path']).read_bytes()).hexdigest()==r[field]['sha256']
passed=sorted({r['asset'] for r in rows if r['verified_success']})
assert config['validated_assets']==passed
assert len(set(config['validated_assets']+config['pending_adaptation_assets']+config['planning_excluded_assets']))==12
for aid,b in config['validated_simulation_baselines'].items():
 assert (I/'source'/b['harness']).exists()
 assert (I/'fixtures'/b['plan_dir']/(aid+'.json')).exists()
assert config['body_joint_positions_deg']==dict(zip(['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3','idx04_body_joint4','idx05_body_joint5'],[-40,142,-87,0,0]))
print('ARCHIVE_OK',len(manifest),'files',len(rows),'trials',len(passed),'passed')
