"""Recreate fixtures in a separate directory and check geometry plus USD composition."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--fixtures',type=Path,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();code=Path(__file__).resolve().parent
def run(name,*args):subprocess.run([sys.executable,str(code/name),*map(str,args)],check=True)
run('materialize.py','--fixtures',a.fixtures,'--home',a.home,'--objects',a.objects,'--out',a.out)
run('validate.py','--out',a.out,'--objects',a.objects)
run('validate_usd.py','--out',a.out)
checks=[]
for ep in sorted(a.fixtures.glob('S*/episodes.json')):
 original=json.loads(ep.read_text());rebuilt=json.loads((a.out/ep.parent.name/'episodes.json').read_text())
 assert original['objects']==rebuilt['objects'] and original['tasks']==rebuilt['tasks']
 content=json.dumps({'objects':original['objects'],'tasks':original['tasks']},sort_keys=True).encode();checks.append({'scene':original['id'],'layout_and_tasks_sha256':hashlib.sha256(content).hexdigest(),'passed':True})
g=json.loads((a.out/'geometry_validation.json').read_text());u=json.loads((a.out/'usd_validation.json').read_text());assert len(checks)==10 and g['passed'] and u['passed']
a.report.write_text(json.dumps({'passed':True,'scenes':checks,'task_count':g['task_count'],'composed_stage_count':len(u['stages']),'scope':'same-machine reconstruction into a separate directory using explicitly supplied asset roots; no claim of a fresh-machine simulator or robot rollout test'},indent=2));print('Separate-directory reproduction passed')
