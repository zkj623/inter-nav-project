from pathlib import Path
import json,subprocess,concurrent.futures,sys,time
R=Path(__file__).resolve().parent;cfg=json.loads((R/'standby_preparation_config.json').read_text());jobs=[(a,i) for i in range(1,25) for a in cfg['execution'] if not (R/'collection_plans'/f'placement_{i:04d}'/(a+'_planning.json')).exists()]
def run(job):
 a,i=job;cmd=[sys.executable,str(R/'plan_collection.py'),'--asset',a,'--index',str(i)];p=subprocess.run(cmd,cwd=R,capture_output=True,text=True);line=p.stdout.strip().splitlines();return dict(asset=a,index=i,returncode=p.returncode,output=line[-1] if line else p.stderr[-1000:])
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for row in pool.map(run,jobs):
  with (R/'planning_batch.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
