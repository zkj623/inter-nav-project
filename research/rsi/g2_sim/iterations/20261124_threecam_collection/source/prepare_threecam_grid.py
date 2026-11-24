from pathlib import Path
import json,subprocess,sys,concurrent.futures
R=Path(__file__).resolve().parent;assets=list(json.loads((R/'standby_preparation_config.json').read_text())['execution']);L=R/'planning_logs';L.mkdir(exist_ok=True)
def run(pair):
 a,i=pair;f=R/f'placements/p{i:04d}'/(a+'_planning.json')
 if not f.exists():
  with (L/f'{a}_{i:04d}.log').open('w') as log:subprocess.run([sys.executable,str(R/'plan_threecam.py'),'--asset',a,'--index',str(i)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 return json.loads(f.read_text()) if f.exists() else {'asset':a,'index':i,'success':False,'failure':'missing planning report'}
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for row in pool.map(run,[(a,i) for i in range(1,25) for a in assets]):
  rows.append(row)
  if len(rows)%12==0:print('PLANNED',len(rows),'PASSED',sum(x['success'] for x in rows),flush=True)
(R/'planning_grid_report.json').write_text(json.dumps(rows,indent=2))
