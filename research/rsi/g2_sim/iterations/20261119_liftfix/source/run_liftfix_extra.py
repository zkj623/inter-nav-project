from pathlib import Path
import json,subprocess,sys,threading
R=Path(__file__).resolve().parent
def run(aid,folder,gpu):
 for ci,p in enumerate(json.loads((R/folder/(aid+'.json')).read_text())['plans']):
  with open(R/f'liftfix_extra_{aid}_{ci}.log','w') as log:
   proc=subprocess.Popen([sys.executable,str(R/'liftfix_trial.py'),'--asset',aid,'--candidate',str(ci),'--plan-dir',folder,'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
   print(aid,ci,'RUNNING',proc.pid,flush=True);code=proc.wait()
  out=R/f'liftfix_trials/{folder}/{aid}/candidate{ci}_repeat1/result.json'
  if not out.exists():print(aid,ci,'MISSING_RESULT',code,flush=True);continue
  d=json.loads(out.read_text());print(aid,ci,d['success'],d.get('failure'),flush=True)
  if d['success']:break
ts=[threading.Thread(target=run,args=args) for args in [('box_reinforced','liftfix_box_plans',0),('bin_pedal','liftfix_pedal_plans',3)]]
for t in ts:t.start()
for t in ts:t.join()
print('EXTRA_DONE',flush=True)
