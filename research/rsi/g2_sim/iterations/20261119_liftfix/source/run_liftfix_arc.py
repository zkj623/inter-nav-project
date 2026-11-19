from pathlib import Path
import json,time,sys,subprocess,threading
R=Path(__file__).resolve().parent
def run(aid,previous,gpu):
 p=R/previous;deadline=time.monotonic()+1200
 while not p.exists():
  if time.monotonic()>deadline:raise TimeoutError(aid)
  time.sleep(5)
 if json.loads(p.read_text())['success']:print(aid,'PREVIOUS_PASSED_SKIP',flush=True);return
 with open(R/f'liftfix_arc_trial_{aid}.log','w') as log:
  proc=subprocess.Popen([sys.executable,str(R/'liftfix_preload_trial.py'),'--asset',aid,'--plan-dir','liftfix_arc_plans','--candidate','1','--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  print(aid,'RUNNING',proc.pid,flush=True);code=proc.wait()
 p=R/f'liftfix_preload_trials/liftfix_arc_plans/{aid}/candidate1_repeat1/result.json'
 if p.exists():
  d=json.loads(p.read_text());print(aid,d['success'],d.get('failure'),flush=True)
 else:print(aid,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [('case_handheld','liftfix_balance_trials/liftfix_pedal_plans/case_handheld/candidate1_repeat1/result.json',6),('bin_round_pedal','liftfix_preload_trials/liftfix_plans/bin_round_pedal/candidate1_repeat1/result.json',5)]]
for t in ts:t.start()
for t in ts:t.join()
print('ARC_DONE',flush=True)
