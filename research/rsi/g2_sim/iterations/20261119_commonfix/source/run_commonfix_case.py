from pathlib import Path
import subprocess,sys,json,threading
R=Path(__file__).resolve().parent
def run(indices,gpu):
 for ci in indices:
  with open(R/f'commonfix_case_{ci}.log','w') as log:
   p=subprocess.Popen([sys.executable,str(R/'commonfix_case_handheld_trial.py'),'--asset','case_handheld','--plan-dir','commonfix_plans','--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(ci,'RUNNING',p.pid,flush=True);code=p.wait()
  f=R/f'commonfix_trials/commonfix_plans/case_handheld/candidate{ci}_repeat1/result.json'
  if f.exists():
   d=json.loads(f.read_text());print(ci,d['success'],d.get('failure'),flush=True)
   if d['success']:break
  else:print(ci,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [([0,1],0),([2,3],3)]]
for t in ts:t.start()
for t in ts:t.join()
print('CASE_DONE',flush=True)
