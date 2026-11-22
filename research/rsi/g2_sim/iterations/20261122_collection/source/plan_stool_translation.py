from pathlib import Path
import json,subprocess,sys,concurrent.futures,random
R=Path(__file__).resolve().parent;a='stool_rectangular';jobs=[]
for i in range(60,100):
 f=R/'collection_plans'/f'placement_{i:04d}';f.mkdir(exist_ok=True);p=f/(a+'_planning.json')
 if p.exists():continue
 p.write_text(json.dumps(dict(asset=a,index=i,success=False,reason='EXTERNAL_PLANNING_PENDING')));jobs.append(i)
def run(i):
 r=random.Random(20261122+i*73);dx=r.uniform(-.02,-.005);dy=r.uniform(-.01,.01);p=subprocess.run([sys.executable,str(R/'plan_collection.py'),'--asset',a,'--index',str(i),'--dx',str(dx),'--dy',str(dy),'--yaw','0'],cwd=R,capture_output=True,text=True);return p.stdout.strip()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for row in pool.map(run,jobs):print(row,flush=True)
