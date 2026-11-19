from pathlib import Path
import json,subprocess,sys,time
R=Path(__file__).resolve().parent;p=R/'liftfix_pedal_plans/case_handheld.json';deadline=time.monotonic()+1000
while not p.exists():
 if time.monotonic()>deadline:raise TimeoutError('case geometry planner')
 time.sleep(5)
plans=json.loads(p.read_text())['plans']
for ci in range(min(2,len(plans))):
 with open(R/f'liftfix_casegeometry_{ci}.log','w') as log:
  proc=subprocess.Popen([sys.executable,str(R/'liftfix_balance_trial.py'),'--asset','case_handheld','--plan-dir','liftfix_pedal_plans','--candidate',str(ci),'--gpu','6'],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  print(ci,'RUNNING',proc.pid,flush=True);code=proc.wait()
 f=R/f'liftfix_balance_trials/liftfix_pedal_plans/case_handheld/candidate{ci}_repeat1/result.json'
 if not f.exists():print('MISSING',code,flush=True);continue
 d=json.loads(f.read_text());print(ci,d['success'],d.get('failure'),flush=True)
 if d['success']:break
print('CASE_GEOMETRY_DONE',flush=True)
