from pathlib import Path
import argparse,json,subprocess,sys,time,fcntl
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--run',required=True);p.add_argument('--dataset',default='pilot4');p.add_argument('--gpu',type=int,default=3);p.add_argument('--port',type=int,default=19024);p.add_argument('--assets',nargs='+');p.add_argument('--duration',type=float,default=80.);p.add_argument('--execute',type=int,default=10);p.add_argument('--split',choices=['train','validation'],default='train');p.add_argument('--evaluation-label',default='training positions, closed-loop policy; no expert feedback');a=p.parse_args()
rows=json.loads((W/'datasets'/a.dataset/'manifest.json').read_text());assets=a.assets or list(dict.fromkeys(r['asset'] for r in rows));out=W/'evaluations'/a.run;out.mkdir(parents=True,exist_ok=True);results=[]
for asset in assets:
 row=next(r for r in rows if r['asset']==asset and r['split']==a.split);folder=row['plan_dir'];T=R/'policy_trials'/a.run/folder/asset/'candidate0_repeat1'
 if not (T/'result.json').exists():
  with (out/(asset+'_simulation.log')).open('w') as log:
   with (W/f'gpu{a.gpu}_simulation.lock').open('w') as gpu_lock:
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    rc=subprocess.run([sys.executable,str(R/f'policy_{asset}_trial.py'),'--asset',asset,'--plan-dir',folder,'--gpu',str(a.gpu),'--run',a.run,'--checkpoint',a.checkpoint,'--policy-url',f'http://127.0.0.1:{a.port}','--duration',str(a.duration),'--execute',str(a.execute)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
 if not (T/'result.json').exists():results.append(dict(asset=asset,success=False,failure='EVALUATION_NO_RESULT',trial=str(T)))
 else:
  with (out/(asset+'_audit.log')).open('w') as log:subprocess.run([sys.executable,str(R/'verify_realpose.py'),'--trial-dir',str(T)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
  result=json.loads((T/'result.json').read_text());v=json.loads((T/'verification.json').read_text()) if (T/'verification.json').exists() else {}
  results.append(dict(asset=asset,success=v.get('verified_success',False),harness_success=result.get('success'),failure=v.get('failure') or result.get('failure'),rejections=v.get('rejections'),lift_m=v.get('min_hold_lift_m',result.get('max_lift_m')),hold_contact_fraction=v.get('four_contact_fraction'),policy=result.get('policy_evaluation'),trial=str(T),video=str(T/'review.mp4')))
 (out/'summary.json').write_text(json.dumps(dict(checkpoint=a.checkpoint,run=a.run,split=a.evaluation_label,dataset_split=a.split,successes=sum(r['success'] for r in results),total=len(results),results=results),indent=2));print('POLICY_EVALUATION',json.dumps(results[-1]),flush=True)


