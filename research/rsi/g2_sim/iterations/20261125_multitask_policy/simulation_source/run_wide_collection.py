"""Resumable 12-object collection. Only independently accepted unique placements count."""
from pathlib import Path
import hashlib
import argparse,subprocess,sys,json,concurrent.futures,queue,time,threading,numpy as np
R=Path(__file__).resolve().parent;pa=argparse.ArgumentParser();pa.add_argument('--assets',nargs='+');pa.add_argument('--target',type=int,default=10);pa.add_argument('--max-index',type=int,default=60);pa.add_argument('--gpus',nargs='+',type=int,default=[5]);args=pa.parse_args();assets=args.assets or list(json.loads((R/'standby_preparation_config.json').read_text())['execution']);gpuq=queue.Queue()
import fcntl
run_lock=(R/'wide_collection.lock').open('w');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (R/'threecam_pilot_verified.json').exists(),'Two synchronized replay pilots must pass before collection'
for g in args.gpus:gpuq.put(g)
S=R/'wide_collection_state';S.mkdir(exist_ok=True);L=R/'wide_collection_logs';L.mkdir(exist_ok=True);lock=threading.Lock();started=time.time()
def write(a,report):
 f=S/(a+'.json');tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(f)
def invoke(command,log,timeout=1800):
 with log.open('w') as f:
  try:return subprocess.run([sys.executable,*map(str,command)],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
  except subprocess.TimeoutExpired:return 124
def collect(a):
 accepted=[];attempts=[];report=dict(asset=a,target=args.target,status='running',accepted=accepted,attempts=attempts)
 for i in range(1,args.max_index+1):
  if len(accepted)>=args.target:break
  folder=f'wideplacements/p{i:04d}';repeat=1;T=R/'realpose_trials'/folder/a/f'candidate0_repeat{repeat}';planfile=R/folder/(a+'_planning.json');report.update(active_index=i,stage='planning');write(a,report)
  if not planfile.exists():invoke([R/'plan_wide_threecam.py','--asset',a,'--index',i],L/f'{a}_{i:04d}_plan.log',300)
  if not planfile.exists():attempts.append(dict(index=i,accepted=False,failure='PLANNER_ERROR'));continue
  p=json.loads(planfile.read_text())
  if not p['success']:attempts.append({**p,'accepted':False});continue
  xy=np.array([p['dx_m'],p['dy_m']])
  if any(np.linalg.norm(xy-[r['dx_m'],r['dy_m']])<.002 for r in accepted):attempts.append({**p,'accepted':False,'failure':'PLACEMENT_NEAR_DUPLICATE_UNDER_2MM'});continue
  if not (T/'result.json').exists():
   report.update(stage='waiting_for_gpu');write(a,report)
   gpu=gpuq.get()
   try:
    report.update(stage='simulation',gpu=gpu);write(a,report);rc=invoke([R/f'realpose_{a}_trial.py','--asset',a,'--plan-dir',folder,'--gpu',gpu,'--repeat',repeat],L/f'{a}_{i:04d}_run.log')
   finally:gpuq.put(gpu)
   if not (T/'result.json').exists():attempts.append({**p,'accepted':False,'failure':'HARNESS_NO_RESULT','exit_code':rc});continue
  report.update(stage='verification');write(a,report)
  if not (T/'verification.json').exists():invoke([R/'verify_realpose.py','--trial-dir',T],L/f'{a}_{i:04d}_physics.log',900)
  if not (T/'verification.json').exists():attempts.append({**p,'accepted':False,'failure':'PHYSICS_AUDITOR_NO_RESULT'});continue
  if not (T/'acceptance.json').exists() or json.loads((T/'acceptance.json').read_text()).get('auditor_sha256')!=hashlib.sha256((R/'verify_threecam.py').read_bytes()).hexdigest():invoke([R/'verify_threecam.py',T],L/f'{a}_{i:04d}_threecam.log',900)
  if not (T/'acceptance.json').exists():attempts.append({**p,'accepted':False,'failure':'THREECAM_AUDITOR_NO_RESULT'});continue
  v=json.loads((T/'acceptance.json').read_text());r={**p,'trial':str(T),'accepted':v['accepted'],'failure':v.get('rejections',[])};attempts.append(r)
  if v['accepted']:accepted.append(r)
  write(a,report);print(a,len(accepted),'/',args.target,'attempt',i,'accepted',v['accepted'],'reasons',v.get('rejections'),flush=True)
 report.update(status='complete' if len(accepted)==args.target else 'needs_iteration',stage='idle',active_index=None);write(a,report);return report
with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets)) as pool:
 futures={pool.submit(collect,a):a for a in assets}
 while futures:
  done,_=concurrent.futures.wait(futures,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
  for f in done:
   a=futures.pop(f)
   try:f.result()
   except Exception as ex:print('WORKER_ERROR',a,repr(ex),flush=True)
  reports=[json.loads(f.read_text()) for f in S.glob('*.json')];summary=dict(elapsed_s=time.time()-started,counts={r['asset']:len(r['accepted']) for r in reports},active={r['asset']:dict(stage=r['stage'],index=r.get('active_index')) for r in reports},complete=all(len(r['accepted'])==args.target for r in reports) and len(reports)==12)
  t=R/'wide_collection_progress.tmp.json';t.write_text(json.dumps(summary,indent=2));t.replace(R/'wide_collection_progress.json')
print('COLLECTION_RUN_FINISHED',flush=True)
