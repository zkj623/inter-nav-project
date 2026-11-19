from pathlib import Path
import subprocess,sys,json,threading
R=Path(__file__).resolve().parent
def run(aid,folder,ci,mode,gpu):
 with open(R/f'liftpair_{mode}_{aid}.log','w') as log:
  p=subprocess.Popen([sys.executable,str(R/f'liftpair_{mode}_trial.py'),'--asset',aid,'--plan-dir',folder,'--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  print(mode,aid,'RUNNING',p.pid,flush=True);code=p.wait()
 f=R/f'liftpair_{mode}_trials/{folder}/{aid}/candidate{ci}_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(mode,aid,d['success'],d.get('failure'),flush=True)
 else:print(mode,aid,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [('case_handheld','liftfix_arc_plans',1,'roll',0),('case_handheld','liftfix_arc_plans',1,'soft',2),('box_reinforced','liftfix_box_plans',0,'roll',3),('box_reinforced','liftfix_box_plans',0,'soft',4)]]
for t in ts:t.start()
for t in ts:t.join()
print('PAIR_DONE',flush=True)
