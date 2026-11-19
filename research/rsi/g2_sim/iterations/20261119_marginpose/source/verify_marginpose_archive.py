from pathlib import Path
import ast,json,hashlib,subprocess
R=Path(__file__).resolve().parent;repo=R.parent/'g2-publish-20261118';root=repo/'research/rsi/g2_sim';I=root/'iterations/20261119_marginpose'
for name,digest in json.loads((I/'manifest.json').read_text()).items():
 p=I/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
 assert subprocess.check_output(['git','show',':'+str(p.relative_to(repo))],cwd=repo)==p.read_bytes(),name
c=json.loads((root/'ground_lift_training_config.json').read_text());assert min(c['body_joint_margin_deg'].values())>=10
assert c['policy_action_dimension']==16 and c['candidate_profile_id']==7
body={n:v for n,v in c['fixed_joint_positions_rad'].items() if 'body_joint' in n}
for p in (I/'fixtures/marginpose_selected7_validated_plans').glob('*.json'):
 for plan in json.loads(p.read_text())['plans']:
  for q in [plan['q_pre'],plan['q_grasp'],plan['q_close']]+plan['q_lift']:
   assert all(abs(q.get(n,0)-v)<1e-10 for n,v in body.items()),p
rows=json.loads((I/'trial_index.json').read_text());passed=[r for r in rows if r['independent_review_pass']]
for r in passed:
 assert r['min_actual_body_limit_margin_deg']>=9.9
 assert any(hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256'] for p in (I/'source').glob('*.py'))
assert set(c['validated_assets'])=={r['asset'] for r in passed}
print('Verified margin >=10deg, fixed-body plans, source hashes, staged bytes, physical passes:',[r['asset'] for r in passed])
