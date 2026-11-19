from pathlib import Path
import json,sys,subprocess,threading
R=Path(__file__).resolve().parent
for label in ['narrow','wide']:
 for i,p in enumerate(json.loads((R/f'commonfix_{label}_plans/case_handheld.json').read_text())['plans']):print('PLAN',label,i,p['candidate']['tilt_deg'],p['candidate']['height_fraction'],p['object_position'],flush=True)
def run(label,ci,gpu):
 folder=f'commonfix_{label}_plans'
 with open(R/f'commonfix_width_trial_{label}_{ci}.log','w') as log:
  proc=subprocess.Popen([sys.executable,str(R/'commonfix_solver_case_handheld_trial.py'),'--asset','case_handheld','--plan-dir',folder,'--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(label,ci,'RUNNING',proc.pid,flush=True);code=proc.wait()
 f=R/f'commonfix_solver_trials/{folder}/case_handheld/candidate{ci}_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(label,ci,d['success'],d.get('failure'),flush=True)
 else:print(label,ci,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [('narrow',0,0),('wide',2,3)]]
for t in ts:t.start()
for t in ts:t.join()
print('WIDTH_DONE',flush=True)
