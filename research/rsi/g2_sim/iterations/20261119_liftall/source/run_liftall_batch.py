from pathlib import Path
import json,subprocess,sys,time,os,queue,threading
R=Path(__file__).resolve().parent;deadline=time.monotonic()+900
while not (R/'liftall_audit_summary.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Audits not complete')
 time.sleep(5)
audit=json.loads((R/'liftall_audit_summary.json').read_text())
assets=['bin_lidded','bin_round_lidded','box_white','basket_woven','basket_rectangular','stool_wood_four_leg','table_side','bin_round_pedal','case_handheld','box_reinforced','stool_rectangular']
jobs=queue.Queue();events=[];lock=threading.Lock()
def save(event):
 with lock:
  events.append(dict(time=time.time(),**event));(R/'liftall_batch_events.json').write_text(json.dumps(events,indent=2));print(event,flush=True)
for aid in assets:
 if audit.get(aid)==1:jobs.put(aid)
 else:save(dict(asset=aid,status='planning_rejected'))
def worker(gpu):
 while True:
  try:aid=jobs.get_nowait()
  except queue.Empty:return
  label='under' if aid=='stool_rectangular' else 'general';out=R/f'liftall_{label}_trials/liftall_validated_plans/{aid}/candidate0_repeat1'
  if out.exists():save(dict(asset=aid,status='existing_output_not_overwritten'));continue
  with open(R/f'liftall_trial_{aid}.log','w') as log:
   p=subprocess.Popen([sys.executable,str(R/f'liftall_{label}_trial.py'),'--asset',aid,'--plan-dir','liftall_validated_plans','--gpu',str(gpu),'--repeat','1'],cwd=R,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
   save(dict(asset=aid,status='running',gpu=gpu,pid=p.pid));code=p.wait()
  if (out/'result.json').exists():
   d=json.loads((out/'result.json').read_text());save(dict(asset=aid,status='complete',success=d['success'],failure=d.get('failure'),exit_code=code))
  else:save(dict(asset=aid,status='missing_result',exit_code=code))
threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in [2,4,5,6]]
for t in threads:t.start()
for t in threads:t.join()
save(dict(status='batch_complete'))
