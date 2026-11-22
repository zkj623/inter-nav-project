from pathlib import Path
import json,subprocess,sys,concurrent.futures,random
R=Path(__file__).resolve().parent;assets=['box_white','box_reinforced','stool_rectangular','table_side'];jobs=[]
for a in assets:
 for i in range(100,120):
  folder=R/'collection_plans'/f'placement_{i:04d}';folder.mkdir(exist_ok=True);path=folder/(a+'_planning.json')
  if path.exists():continue
  path.write_text(json.dumps(dict(asset=a,index=i,success=False,reason='EXTERNAL_PLANNING_PENDING')));jobs.append((a,i))
def run(job):
 a,i=job;r=random.Random(20261122+i*71+assets.index(a))
 if a in ['box_white','box_reinforced']:dx=r.uniform(.003,.009);dy=r.uniform(-.003,.003);yaw=r.uniform(-.5,.5)
 elif a=='stool_rectangular':dx=r.uniform(-.004,-.0005);dy=r.uniform(-.002,.002);yaw=0.
 else:dx=r.uniform(-.003,.003);dy=r.uniform(-.003,.003);yaw=r.uniform(-.3,.3)
 p=subprocess.run([sys.executable,str(R/'plan_collection.py'),'--asset',a,'--index',str(i),'--dx',str(dx),'--dy',str(dy),'--yaw',str(yaw)],capture_output=True,text=True,cwd=R);return p.stdout.strip()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for row in pool.map(run,jobs):print(row,flush=True)
