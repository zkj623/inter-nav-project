from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_liftfix'
manifest=json.loads((I/'manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in manifest.items():assert sha(I/name)==digest,name
sources={sha(p):p for p in (I/'source').glob('*.py')}
for p in sources.values():ast.parse(p.read_text())
rows=json.loads((I/'trial_index.json').read_text());config=json.loads((root/'ground_lift_training_config.json').read_text())
for r in rows:
 assert r['source_sha256'] in sources,r['asset']
 for key in ['samples.json','review.mp4']:
  assert sha(Path(r[key]['server_path']))==r[key]['sha256']
old=json.loads((root/'iterations/20261119_liftall/fixtures/liftall_ground_lift_config.json').read_text())
assert set(config['validated_assets'])==set(old['validated_assets'])|{r['asset'] for r in rows if r['verified_success']}
assert len(set(config['validated_assets']+config['pending_adaptation_assets']))==12
for a,b in config['validated_simulation_baselines'].items():
 arc=root/'iterations'/b['iteration'];assert (arc/'source'/b['harness']).exists()
 ps=json.loads((arc/'fixtures'/b['plan_dir']/(a+'.json')).read_text())['plans'];assert b['candidate']<len(ps)
assert config['body_joint_positions_deg']==old['body_joint_positions_deg']
print('ARCHIVE_OK',len(manifest),'files',len(rows),'new trials',len(config['validated_assets']),'total passed')
