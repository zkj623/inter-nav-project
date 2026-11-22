from pathlib import Path
from collections import Counter
import json,subprocess,sys,os,time,threading,concurrent.futures
R=Path(__file__).resolve().parent;PY=sys.executable;assets=list(json.loads((R/'standby_preparation_config.json').read_text())['execution']);lock=threading.Lock()
def stats():
 good={a:set() for a in assets};pending=Counter();attempts=Counter();active=Counter()
 for D in R.glob('collection_trials/collection_plans/placement_*/*/candidate*'):
  a=D.parent.name
  if a not in good:continue
  f=D/'result.json'
  if not f.exists():active[a]+=1;continue
  try:d=json.loads(f.read_text())
  except json.JSONDecodeError:continue
  attempts[a]+=1;v=D/'verification.json'
  if v.exists():
   try:r=json.loads(v.read_text())
   except json.JSONDecodeError:continue
   if r['verified_success']:good[a].add(d['plan']['collection_placement']['group_id'])
  elif d['success']:pending[a]+=1
 return good,pending,attempts,active

def pick(gpu):
 good,pending,attempts,active=stats()
 for a in sorted(assets,key=lambda x:(len(good[x])+pending[x]+active[x],attempts[x],assets.index(x))):
  if len(good[a])+pending[a]+active[a]>=20 or attempts[a]>=60:continue
  for p in sorted(R.glob(f'collection_plans/placement_*/{a}_planning.json')):
   try:d=json.loads(p.read_text())
   except json.JSONDecodeError:continue
   if not d['success']:continue
   i=d['index'];D=R/'collection_trials'/'collection_plans'/f'placement_{i:04d}'/a/'candidate0_repeat1'
   try:D.mkdir(parents=True,exist_ok=False)
   except FileExistsError:continue
   (D/'reservation.json').write_text(json.dumps(dict(owner_pid=os.getpid(),gpu=gpu,asset=a,index=i,wall_time=time.time(),scope='exclusive OUT-directory reservation; one physical process only')))
   return a,i,D
 return None

def gpu_ready(gpu,cap):
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(gpu)],text=True).strip())
 if free<6800:return False
 ps=subprocess.check_output(['ps','-u','zkj','-o','args='],text=True);count=sum('/python ' in l and '_trial.py' in l and ('collect_' in l or 'replay_' in l) and '--gpu '+str(gpu)+' ' in l+' ' and 'bash -c' not in l for l in ps.splitlines())
 return count<cap

def worker(gpu,cap):
 while True:
  good,pending,attempts,active=stats()
  if all(len(good[a])>=20 or attempts[a]>=60 for a in assets):return
  if not gpu_ready(gpu,cap):time.sleep(10);continue
  with lock:job=pick(gpu)
  if job is None:time.sleep(10);continue
  a,i,D=job;log=R/'batch_logs'/f'{a}_{i:04d}_parallel.log';env=dict(os.environ,G2_RESERVED_TRIAL=str(D.resolve()))
  print('EXTRA_LAUNCH',gpu,a,i,flush=True)
  with log.open('w') as out:
   try:p=subprocess.run([PY,str(R/f'collect_{a}_parallel_trial.py'),'--asset',a,'--plan-dir',f'collection_plans/placement_{i:04d}','--gpu',str(gpu)],cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=1800);print('EXTRA_FINISH',gpu,a,i,p.returncode,flush=True)
   except Exception as e:print('EXTRA_ERROR',gpu,a,i,str(e),flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 list(pool.map(lambda x:worker(*x),[(6,3),(6,3),(6,3),(4,2),(4,2)]))
