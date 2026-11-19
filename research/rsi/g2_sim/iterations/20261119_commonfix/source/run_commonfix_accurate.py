from pathlib import Path
import json,sys,time,subprocess,threading
R=Path(__file__).resolve().parent
def run(a,folder,gpu):
 previous=R/f'commonfix_precise_trials/{folder}/{a}/candidate0_repeat1/result.json'
 while not previous.exists():time.sleep(5)
 with open(R/f'commonfix_accurate_{a}.log','w') as log:
  p=subprocess.Popen([sys.executable,str(R/f'commonfix_accurate_{a}_trial.py'),'--asset',a,'--plan-dir',folder,'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(a,'RUNNING',p.pid,flush=True);code=p.wait()
 f=R/f'commonfix_accurate_trials/{folder}/{a}/candidate0_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(a,d['success'],d.get('failure'),flush=True)
 else:print(a,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [('basket_rectangular','commonfix_pinch_plans',2),('case_handheld','commonstart_plans',3)]]
for t in ts:t.start()
for t in ts:t.join()
print('ACCURATE_DONE',flush=True)
