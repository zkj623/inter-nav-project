from pathlib import Path
import json,subprocess,sys,os,threading
R=Path(__file__).resolve().parent
source=(R/'liftall_general_trial.py').read_text().replace("ROOT/'liftall_general_trials'","ROOT/'liftfix_trials'")
(R/'liftfix_trial.py').write_text(source)
events=[];lock=threading.Lock()
def save(e):
 with lock:
  events.append(e);(R/'liftfix_events.json').write_text(json.dumps(events,indent=2));print(e,flush=True)
def worker(aid,gpu):
 doc=json.loads((R/'liftfix_plans'/(aid+'.json')).read_text())
 for ci,p in enumerate(doc['plans']):
  with open(R/f'liftfix_trial_{aid}_{ci}.log','w') as log:
   proc=subprocess.Popen([sys.executable,str(R/'liftfix_trial.py'),'--asset',aid,'--candidate',str(ci),'--plan-dir','liftfix_plans','--gpu',str(gpu)],cwd=R,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
   save(dict(asset=aid,candidate=ci,status='running',pid=proc.pid,gpu=gpu));code=proc.wait()
  out=R/f'liftfix_trials/liftfix_plans/{aid}/candidate{ci}_repeat1/result.json'
  if out.exists():
   d=json.loads(out.read_text());save(dict(asset=aid,candidate=ci,status='complete',success=d['success'],failure=d.get('failure')))
   if d['success']:break
  else:save(dict(asset=aid,candidate=ci,status='missing_result',exit_code=code))
ts=[threading.Thread(target=worker,args=(a,g)) for a,g in [('bin_round_pedal',2),('basket_rectangular',4),('basket_woven',5),('case_handheld',6)]]
for t in ts:t.start()
for t in ts:t.join()
save(dict(status='batch_complete'))
