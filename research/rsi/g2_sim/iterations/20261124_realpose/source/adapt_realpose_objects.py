from pathlib import Path
import json,subprocess,sys,concurrent.futures
R=Path(__file__).resolve().parent;failed=[]
for p in (R/'realpose_plans').glob('*_planning.json'):
 d=json.loads(p.read_text())
 if not d['success']:failed.append(d['asset'])
def adapt(a):
 for dx in [-.01,-.02,-.03,-.04,-.05]:
  p=subprocess.run([sys.executable,str(R/'plan_realpose.py'),'--asset',a,'--dx',str(dx)],cwd=R,capture_output=True,text=True)
  with (R/'realpose_plans'/(a+'_adapt.log')).open('a') as f:f.write(p.stdout+p.stderr)
  print(a,dx,p.returncode,p.stdout[-300:] or p.stderr[-300:],flush=True)
  if p.returncode==0:break
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(adapt,failed))
rows=[json.loads(p.read_text()) for p in (R/'realpose_plans').glob('*_planning.json')];(R/'realpose_planning_summary.json').write_text(json.dumps(rows,indent=2));print('ADAPT_DONE',sum(r['success'] for r in rows),len(rows),flush=True)
