from pathlib import Path
import argparse,json,subprocess,sys,time,fcntl
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--run',required=True);p.add_argument('--dataset',default='pilot4');p.add_argument('--gpu',type=int,default=3);p.add_argument('--port',type=int,default=19024);p.add_argument('--assets',nargs='+');p.add_argument('--duration',type=float,default=80.);p.add_argument('--execute',type=int,default=10);p.add_argument('--time-conditioned',action='store_true');p.add_argument('--split',choices=['train','validation'],default='train');p.add_argument('--evaluation-label',default='training positions, closed-loop policy; no expert feedback');p.add_argument('--sampling-seed',type=int);p.add_argument('--protocol',choices=['stable','lock10'],default='lock10');a=p.parse_args()
if a.time_conditioned and a.protocol=='lock10':raise ValueError('lock10 requires the image/qpos policy adapter')
probe_dir=W/'pi05_focus_20261125'; probe_dir.mkdir(exist_ok=True)
import hashlib,urllib.request
sampling_seed=a.sampling_seed if a.sampling_seed is not None else int(hashlib.sha256(a.run.encode()).hexdigest()[:8],16)%(2**31)
seed_request=urllib.request.Request(f'http://127.0.0.1:{a.port}/set_seed',data=json.dumps(dict(seed=sampling_seed)).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(seed_request,timeout=30) as response:assert json.load(response)['sampling_seed']==sampling_seed
(probe_dir/(a.run+'_sampling_seed.json')).write_text(json.dumps(dict(seed=sampling_seed,run=a.run)))
with (probe_dir/(a.run+'_offline_probe.log')).open('w') as probe_log:
 subprocess.run([sys.executable,str(W/'probe_pi05_focus.py'),'--checkpoint',a.checkpoint,'--port',str(a.port),'--dataset',a.dataset,'--split',a.split,'--output',str(probe_dir/(a.run+'_offline_probe.json'))],cwd=W,stdout=probe_log,stderr=subprocess.STDOUT,timeout=1800,check=True)
rows=json.loads((W/'datasets'/a.dataset/'manifest.json').read_text());assets=a.assets or list(dict.fromkeys(r['asset'] for r in rows if r['split']==a.split));out=W/'evaluations'/a.run;out.mkdir(parents=True,exist_ok=True);results=[]
for asset in assets:
 row=next(r for r in rows if r['asset']==asset and r['split']==a.split);folder=row['plan_dir'];T=R/'policy_trials'/a.run/folder/asset/'candidate0_repeat1'
 if not (T/'result.json').exists():
  with (out/(asset+'_simulation.log')).open('w') as log:
   with (W/f'gpu{a.gpu}_simulation.lock').open('w') as gpu_lock:
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    rc=subprocess.run([sys.executable,str(R/((f'policy_time_{asset}_trial.py') if a.time_conditioned else f'{a.protocol}_policy_{asset}_trial.py')),'--asset',asset,'--plan-dir',folder,'--gpu',str(a.gpu),'--run',a.run,'--checkpoint',a.checkpoint,'--policy-url',f'http://127.0.0.1:{a.port}','--duration',str(a.duration),'--execute',str(a.execute)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
 if not (T/'result.json').exists():results.append(dict(asset=asset,success=False,failure='EVALUATION_NO_RESULT',trial=str(T)))
 else:
  existing=json.loads((T/'result.json').read_text())
  if not existing.get('acceptance_revision','').startswith('lock10' if a.protocol=='lock10' else 'stable'):raise RuntimeError('Existing trial has a different protocol; use a new run name')
  with (out/(asset+'_audit.log')).open('w') as log:subprocess.run([sys.executable,str(R/f'verify_{a.protocol}_lift.py'),'--trial-dir',str(T)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
  result=json.loads((T/'result.json').read_text());v=json.loads((T/'stable_verification.json').read_text()) if (T/'stable_verification.json').exists() else {}
  if v.get('verified_success',False):
   with (out/(asset+'_threecam_audit.log')).open('w') as camera_log:subprocess.run([sys.executable,str(R/'verify_pi05_threecam.py'),str(T)],cwd=R,stdout=camera_log,stderr=subprocess.STDOUT,timeout=1800)
   camera=json.loads((T/'stable_threecam_acceptance.json').read_text()) if (T/'stable_threecam_acceptance.json').exists() else {}
   if not camera.get('accepted',False):
    v['verified_success']=False;v['failure']='THREECAM_VALIDATION';v['rejections']=camera.get('rejections',['THREECAM_AUDIT_NO_RESULT'])
  results.append(dict(asset=asset,success=v.get('verified_success',False),harness_success=result.get('success'),failure=v.get('failure') or result.get('failure'),rejections=v.get('rejections'),lift_m=v.get('min_hold_lift_m',result.get('max_lift_m')),max_lift_m=result.get('max_lift_m'),min_hold_lift_m=result.get('min_hold_lift_m'),max_hold_tilt_deg=result.get('max_hold_tilt_deg'),hold_contact_fraction=v.get('four_contact_fraction'),policy=result.get('policy_evaluation'),trial=str(T),video=str(T/'review.mp4')))
 (out/'summary.json').write_text(json.dumps(dict(protocol=a.protocol,checkpoint=a.checkpoint,run=a.run,sampling_seed=sampling_seed,split=a.evaluation_label,dataset_split=a.split,successes=sum(r['success'] for r in results),total=len(results),results=results),indent=2));print('POLICY_EVALUATION',json.dumps({k:v for k,v in results[-1].items() if k!='policy'}),flush=True)




subprocess.run([sys.executable,str(W/'export_pi05_focus_review.py'),'--run',a.run],cwd=W,check=True,timeout=3600)
