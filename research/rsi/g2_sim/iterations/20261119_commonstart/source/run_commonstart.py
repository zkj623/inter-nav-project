from pathlib import Path
import json,sys,subprocess,queue,threading,time
R=Path(__file__).resolve().parent
deadline=time.monotonic()+1800
while not (R/'commonstart_planning.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('common start planner')
 time.sleep(5)
doc=json.loads((R/'commonstart_planning.json').read_text());jobs=queue.Queue();events=[];lock=threading.Lock()
for a in doc['accepted']:jobs.put(a)
def save(e):
 with lock:
  events.append(e);(R/'commonstart_events.json').write_text(json.dumps(events,indent=2));print(e,flush=True)
def worker(gpu):
 while not jobs.empty():
  try:a=jobs.get_nowait()
  except queue.Empty:return
  with open(R/f'commonstart_trial_{a}.log','w') as log:
   p=subprocess.Popen([sys.executable,str(R/f'commonstart_{a}_trial.py'),'--asset',a,'--plan-dir','commonstart_plans','--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);save(dict(asset=a,status='running',pid=p.pid,gpu=gpu));code=p.wait()
  f=R/f'commonstart_trials/commonstart_plans/{a}/candidate0_repeat1/result.json'
  if f.exists():
   d=json.loads(f.read_text());save(dict(asset=a,status='complete',success=d['success'],failure=d.get('failure')))
  else:save(dict(asset=a,status='missing_result',exit_code=code))
ts=[threading.Thread(target=worker,args=(g,)) for g in [0,2,3,6]]
for t in ts:t.start()
for t in ts:t.join()
save(dict(status='batch_complete'))
