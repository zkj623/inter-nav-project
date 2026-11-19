from pathlib import Path
import json,subprocess,sys,threading,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
doc=json.loads((R/'caseyaw_reverse_plans/case_handheld.json').read_text())
for i,p in enumerate(doc['plans']):
 fk=k.fk(p['q_grasp'],np.array(p['base_pose']));print('REVERSE_PLAN',i,p['candidate'],p['object_position'],'wrist_x_offsets',[fk[f'arm_{side}_end_link'][0,3]-p['object_position'][0] for side in ['l','r']],flush=True)
def run(ci,gpu):
 with open(R/f'caseyaw_reverse_{ci}.log','w') as log:
  proc=subprocess.Popen([sys.executable,str(R/'caseyaw_live_trial.py'),'--asset','case_handheld','--plan-dir','caseyaw_reverse_plans','--candidate',str(ci),'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT);print(ci,'RUNNING',proc.pid,flush=True);code=proc.wait()
 f=R/f'caseyaw_trials/caseyaw_reverse_plans/case_handheld/candidate{ci}_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(ci,d['success'],d.get('failure'),flush=True)
 else:print(ci,'MISSING',code,flush=True)
ts=[threading.Thread(target=run,args=x) for x in [(0,0),(1,6)]]
for t in ts:t.start()
for t in ts:t.join()
print('REVERSE_DONE',flush=True)
