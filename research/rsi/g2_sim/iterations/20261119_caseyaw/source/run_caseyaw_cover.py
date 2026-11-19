from pathlib import Path
import subprocess,sys,json,threading
R=Path(__file__).resolve().parent
def run(label,gpu):
 for script,args in [(f'plan_caseyaw_{label}.py',['--asset','case_handheld']),(f'connect_caseyaw_{label}.py',['case_handheld'])]:
  with open(R/f'caseyaw_{label}_{script.split("_")[0]}.log','w') as log:subprocess.run([sys.executable,str(R/script)]+args,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 folder=f'caseyaw_{label}_plans';doc=json.loads((R/folder/'case_handheld.json').read_text());print(label,'PLANS',len(doc['plans']),flush=True)
 if not doc['plans']:return
 with open(R/f'caseyaw_{label}_trial.log','w') as log:
  p=subprocess.Popen([sys.executable,str(R/'caseyaw_live_trial.py'),'--asset','case_handheld','--plan-dir',folder,'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(label,'RUNNING',p.pid,flush=True);code=p.wait()
 f=R/f'caseyaw_trials/{folder}/case_handheld/candidate0_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(label,d['success'],d.get('failure'),flush=True)
 else:print(label,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [('rear',0),('other',6),('narrow90',2)]]
for t in ts:t.start()
for t in ts:t.join()
print('COVER_DONE',flush=True)
