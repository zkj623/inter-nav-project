from pathlib import Path
import json,subprocess,sys,threading
R=Path(__file__).resolve().parent
def run(ci,gpu):
 with open(R/f'caseyaw_trial_{ci}.log','w') as log:
  p=subprocess.Popen([sys.executable,str(R/'caseyaw_trial.py'),'--asset','case_handheld','--plan-dir','caseyaw_plans','--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(ci,'RUNNING',p.pid,flush=True);code=p.wait()
 f=R/f'caseyaw_trials/caseyaw_plans/case_handheld/candidate{ci}_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(ci,d['success'],d.get('failure'),flush=True)
 else:print(ci,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [(0,0),(1,6)]]
for t in ts:t.start()
for t in ts:t.join()
print('CASEYAW_DONE',flush=True)
