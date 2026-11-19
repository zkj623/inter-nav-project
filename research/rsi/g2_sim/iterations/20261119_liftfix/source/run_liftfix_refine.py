from pathlib import Path
import json,subprocess,sys,time,threading
R=Path(__file__).resolve().parent
def run(aid,rawfolder,folder,indices,gpu):
 count=len(json.loads((R/rawfolder/(aid+'.json')).read_text())['plans']);deadline=time.monotonic()+2400
 while True:
  completed=[]
  for p in (R/'liftfix_trials'/rawfolder/aid).glob('*/result.json'):completed.append(json.loads(p.read_text()))
  if any(d['success'] for d in completed):print(aid,'RAW_PASSED_SKIP_REFINE',flush=True);return
  if len(completed)>=count:break
  if time.monotonic()>deadline:raise TimeoutError(aid)
  time.sleep(10)
 for ci in indices:
  with open(R/f'liftfix_refine_{aid}_{ci}.log','w') as log:
   proc=subprocess.Popen([sys.executable,str(R/'liftfix_balance_trial.py'),'--asset',aid,'--plan-dir',folder,'--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
   print(aid,ci,'RUNNING',proc.pid,flush=True);code=proc.wait()
  p=R/f'liftfix_balance_trials/{folder}/{aid}/candidate{ci}_repeat1/result.json'
  if not p.exists():print(aid,'MISSING',code,flush=True);continue
  d=json.loads(p.read_text());print(aid,ci,d['success'],d.get('failure'),flush=True)
  if d['success']:break
ts=[threading.Thread(target=run,args=x) for x in [('basket_woven','liftfix_plans','liftfix_forward_plans',[0,1],5),('case_handheld','liftfix_plans','liftfix_plans',[1],6),('box_reinforced','liftfix_box_plans','liftfix_box_plans',[1],0),('bin_pedal','liftfix_pedal_plans','liftfix_pedal_plans',[1],3)]]
for t in ts:t.start()
for t in ts:t.join()
print('REFINE_DONE',flush=True)
