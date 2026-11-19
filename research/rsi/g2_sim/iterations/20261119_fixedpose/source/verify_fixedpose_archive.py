from pathlib import Path
import ast,json,hashlib,subprocess
R=Path(__file__).resolve().parent;repo=R.parent/'g2-publish-20261118';I=repo/'research/rsi/g2_sim/iterations/20261119_fixedpose'
manifest=json.loads((I/'manifest.json').read_text())
for name,digest in manifest.items():
 p=I/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
 staged=subprocess.check_output(['git','show',':'+str(p.relative_to(repo))],cwd=repo)
 assert staged==p.read_bytes(),name
summary=json.loads((I/'fixtures/fixedpose_selected3_audit_summary.json').read_text());assert sum(summary.values())==11
config=json.loads((I/'fixtures/fixed_ground_lift_config.json').read_text());body={n:v for n,v in config['fixed_joint_positions_rad'].items() if 'body_joint' in n}
for p in (I/'fixtures/fixedpose_selected3_validated_plans').glob('*.json'):
 for plan in json.loads(p.read_text())['plans']:
  for q in [plan['q_pre'],plan['q_grasp'],plan['q_close']]+plan['q_lift']:
   assert all(abs(q.get(n,0)-v)<1e-10 for n,v in body.items()),p
  assert all(x['ok'] for x in plan['fixedpose_envelope_checks'])
review=json.loads((I/'fixtures/fixedpose_physical_geometry_review.json').read_text());assert len(review)==3 and all(x['penetration_pass'] for x in review)
assert config['policy_action_dimension']==16
print('Verified',len(manifest),'archive files, 11 constant-torso plans, 3 physical geometry reviews, 16 action dimensions')
