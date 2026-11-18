from pathlib import Path
import ast,hashlib,json,subprocess
R=Path(__file__).resolve().parent;repo=R.parent/'g2-publish-20261118';I=repo/'research/rsi/g2_sim/iterations/20261118_chairs'
manifest=json.loads((I/'manifest.json').read_text())
for name,expected in manifest.items():
 p=I/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 blob=subprocess.check_output(['git','show',':'+str(p.relative_to(repo))],cwd=repo);assert hashlib.sha256(blob).hexdigest()==expected,('staged',name)
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.json':json.loads(p.read_text())
summary=json.loads((I/'planning_summary.json').read_text());assert len(summary)==8,len(summary)
for report in summary:
 p=I/'plans'/report['variant']/(report['asset']+'.json');assert len(json.loads(p.read_text())['plans'])==report['accepted']
print('Verified',len(manifest),'staged files,',len(summary),'completed planning experiments')
