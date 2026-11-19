from pathlib import Path
import json,subprocess,sys,threading
R=Path(__file__).resolve().parent
def run(ci,gpu):
 with open(R/f'commonfix_pinch_basket_{ci}.log','w') as log:
  p=subprocess.Popen([sys.executable,str(R/'commonfix_basket_rectangular_trial.py'),'--asset','basket_rectangular','--plan-dir','commonfix_pinch_plans','--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(ci,'RUNNING',p.pid,flush=True);code=p.wait()
 f=R/f'commonfix_trials/commonfix_pinch_plans/basket_rectangular/candidate{ci}_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(ci,d['success'],d.get('failure'),flush=True)
 else:print(ci,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [(0,2),(1,5)]]
for t in ts:t.start()
for t in ts:t.join()
print('BASKET_DONE',flush=True)
