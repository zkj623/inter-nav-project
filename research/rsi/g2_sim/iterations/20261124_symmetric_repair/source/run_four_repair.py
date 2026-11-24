from pathlib import Path
import json,sys,subprocess,concurrent.futures,queue,argparse
R=Path(__file__).resolve().parent;pa=argparse.ArgumentParser();pa.add_argument('--plan-dir',default='repair_v1');pa.add_argument('--assets',nargs='+',default=['basket_rectangular','box_white','box_reinforced','table_side']);pa.add_argument('--gpus',nargs='+',type=int,default=[6,2,5]);pa.add_argument('--repeat',type=int,default=1);a=pa.parse_args();Q=queue.Queue()
for g in a.gpus:Q.put(g)
def run(asset):
 gpu=Q.get()
 try:
  with (R/f'{a.plan_dir}_{asset}_repeat{a.repeat}.log').open('w') as f:subprocess.run([sys.executable,str(R/f'realpose_{asset}_trial.py'),'--asset',asset,'--plan-dir',a.plan_dir,'--repeat',str(a.repeat),'--gpu',str(gpu)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  T=R/'realpose_trials'/a.plan_dir/asset/f'candidate0_repeat{a.repeat}'
  if not (T/'result.json').exists():return dict(asset=asset,verified_success=False,error='missing result')
  with (R/f'{a.plan_dir}_{asset}_audit.log').open('w') as f:subprocess.run([sys.executable,str(R/'verify_realpose.py'),'--trial-dir',str(T)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  result=json.loads((T/'verification.json').read_text());print(asset,result.get('verified_success'),result.get('failure'),result.get('rejections'),flush=True);return result
 finally:Q.put(gpu)
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as ex:
 for f in concurrent.futures.as_completed([ex.submit(run,x) for x in a.assets]):
  try:out.append(f.result())
  except Exception as e:out.append(dict(error=str(e)))
  (R/f'{a.plan_dir}_results.json').write_text(json.dumps(out,indent=2))
print('DONE',flush=True)
