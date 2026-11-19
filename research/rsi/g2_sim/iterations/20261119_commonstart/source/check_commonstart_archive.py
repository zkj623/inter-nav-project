from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_commonstart';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((I/'manifest.json').read_text())
for n,d in manifest.items():assert sha(I/n)==d,n
sources={sha(p):p for p in (I/'source').glob('*.py')}
for p in sources.values():ast.parse(p.read_text())
rows=json.loads((I/'trial_index.json').read_text());cfg=json.loads((R/'commonstart_ground_lift_config.json').read_text());home=cfg['common_initial_joint_positions_rad']
for r in rows:
 assert r['source_sha256'] in sources,r['asset']
 for key in ['samples.json','review.mp4']:
  if key in r:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
for a,b in cfg['validated_simulation_baselines'].items():
 f=I/'fixtures'/b['plan_dir']/(a+'.json');p=json.loads(f.read_text())['plans'][b['candidate']];assert p['q_start']==home
 assert (I/'source'/b['harness']).exists();assert p['q_approach'][-1]==p['q_pre']
 matching=[r for r in rows if r['asset']==a and r['verified_success'] and r['result'].split('/')[1]==b['plan_dir'] and r['source_sha256']==sha(I/'source'/b['harness'])];assert matching
assert len(set(cfg['validated_assets']+cfg['pending_adaptation_assets']))==12
print('ARCHIVE_OK',len(manifest),'files',len(rows),'trials;',len(cfg['validated_assets']),'common-start complete successes')
