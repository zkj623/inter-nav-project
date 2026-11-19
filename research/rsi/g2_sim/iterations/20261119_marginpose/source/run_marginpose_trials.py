from pathlib import Path
import time,json,subprocess,os,sys
R=Path(__file__).resolve().parent;p=R/'marginpose_selected7_audit_summary.json';deadline=time.monotonic()+600
while not p.exists():
 if time.monotonic()>deadline:raise TimeoutError('Planning audits not complete')
 time.sleep(5)
summary=json.loads(p.read_text());jobs=[]
for aid,label,gpu in [('box_reinforced','deadband',2),('case_handheld','deadband',5),('stool_rectangular','support',0)]:
 assert summary[aid]==1,aid
 out=R/f'marginpose_{label}_trials/marginpose_selected7_validated_plans/{aid}/candidate0_repeat1'
 assert not out.exists(),out
 log=open(R/f'margin_trial_{aid}.log','w')
 proc=subprocess.Popen([sys.executable,str(R/f'marginpose_{label}_trial.py'),'--asset',aid,'--plan-dir','marginpose_selected7_validated_plans','--gpu',str(gpu),'--repeat','1'],cwd=R,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append(dict(asset=aid,pid=proc.pid,gpu=gpu));log.close()
(R/'marginpose_jobs.json').write_text(json.dumps(jobs,indent=2));print(jobs,flush=True)
