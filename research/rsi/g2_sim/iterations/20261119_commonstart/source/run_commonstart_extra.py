from pathlib import Path
import json,time,subprocess,sys
R=Path(__file__).resolve().parent
while not (R/'commonstart_extra_planning.json').exists():time.sleep(5)
while True:
 f=R/'commonstart_events.json'
 if f.exists() and any(e.get('status')=='batch_complete' for e in json.loads(f.read_text())):break
 time.sleep(5)
for r in json.loads((R/'commonstart_extra_planning.json').read_text()):
 if r['path'] is None:continue
 a=r['asset']
 with open(R/f'commonstart_extra_trial_{a}.log','w') as log:
  p=subprocess.Popen([sys.executable,str(R/f'commonstart_{a}_trial.py'),'--asset',a,'--plan-dir','commonstart_extra_plans','--gpu','0'],cwd=R,stdout=log,stderr=subprocess.STDOUT);print('RUNNING',a,p.pid,flush=True);p.wait()
 print('DONE',a,flush=True)
