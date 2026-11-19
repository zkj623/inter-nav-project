from pathlib import Path
import json,ast,numpy as np
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
line=next(s for s in (R/'search_marginpose.py').read_text().splitlines() if s.startswith('profiles='));profiles=ast.literal_eval(line.split('=',1)[1])
for i,angles in enumerate(profiles):
 q=dict(zip([n for n in k.by_name if 'body_joint' in n],np.radians(angles)));fk=k.fk(q,transform((0,0,.04)))
 p=R/f'marginpose_search_{i}.json';rows=json.loads(p.read_text()) if p.exists() else []
 print('PROFILE',i,angles,'shoulder',np.round(fk['arm_l_link1'][:3,3],3).tolist(),'tested',len(rows),'passed',sum(x['passed'] for x in rows),'fail',[x['asset'] for x in rows if not x['passed']],flush=True)
for p in R.glob('marginpose_*_trials/**/result.json'):
 d=json.loads(p.read_text());print('TRIAL',p.relative_to(R),{n:d.get(n) for n in ['success','failure','error','min_hold_lift_m','max_hold_tilt_deg','max_frozen_joint_error_rad']},flush=True)
