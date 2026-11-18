from pathlib import Path
import json,hashlib,subprocess,ast
R=Path(__file__).resolve().parent;repo=R.parent/'g2-publish-20261118';I=repo/'research/rsi/g2_sim/iterations/20261118_push'
def sha(data):return hashlib.sha256(data).hexdigest()
m=json.loads((I/'manifest.json').read_text());assert set(m)=={str(p.relative_to(I)) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'}
for name,digest in m.items():
 p=I/name;assert sha(p.read_bytes())==digest,name
 staged=subprocess.check_output(['git','show',':'+str(p.relative_to(repo))],cwd=repo);assert sha(staged)==digest,('staged bytes changed',name)
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
rows=json.loads((I/'trial_index.json').read_text());baselines=json.loads((I/'baselines.json').read_text())
for r in rows:
 d=json.loads((I/'results'/r['result']).read_text());assert sha((I/'source'/r['harness']).read_bytes())==d['source_sha256']==r['source_sha256']
 assert sha(Path(r['samples_server_path']).read_bytes())==r['samples_sha256']
 if r['review_accepted']:
  assert d['success'] and not d['object_attachment'] and not d['robot_fixed_to_world']
  assert r['nonhand_contact_review_passed'] and r['force_review_passed'] and r['initialization_corrected'] and all(r['checks'].values())
  if d['plan']['mode']=='bimanual_push':assert r['both_hands_push_samples']>=5
for aid,b in baselines.items():
 matches=[r for r in rows if r['asset']==aid and r['variant']==b['variant'] and r['candidate']==b['candidate'] and r['review_accepted']]
 assert len({r['repeat'] for r in matches})>=3 and len({r['source_sha256'] for r in matches})==1
print(json.dumps(dict(verified_files=len(m),trial_count=len(rows),review_pass_count=sum(r['review_accepted'] for r in rows),repeated_baselines=list(baselines)),indent=2))
