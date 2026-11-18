from pathlib import Path
import ast,hashlib,json,subprocess
R=Path(__file__).resolve().parent;repo=R.parent/'g2-publish-20261118';I=repo/'research/rsi/g2_sim/iterations/20261118_collision'
manifest=json.loads((I/'manifest.json').read_text())
for name,expected in manifest.items():
 p=I/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 blob=subprocess.check_output(['git','show',':'+str(p.relative_to(repo))],cwd=repo);assert hashlib.sha256(blob).hexdigest()==expected,('staged',name)
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.json':json.loads(p.read_text())
summary=json.loads((I/'planning_summary.json').read_text());assert len(summary)==11
for r in summary:
 p=I/'plans'/f"strategy15_{r['variant']}_validated_plans"/(r['asset']+'.json');assert len(json.loads(p.read_text())['plans'])==r['accepted']
checks=json.loads((I/'diagnostics/guidance_validation.json').read_text());assert all(r['max_proxy_self_overlap_m']==0 for r in checks if r['variant']=='core')
d=json.loads((I/'diagnostics/shared_ik_validation.json').read_text());assert d['success'] and d['shared_body_identical'] and all(c['ok'] for c in d['envelope_checks'])
print('Verified',len(manifest),'staged files, 11 planning experiments and guidance/shared-IK regressions')
