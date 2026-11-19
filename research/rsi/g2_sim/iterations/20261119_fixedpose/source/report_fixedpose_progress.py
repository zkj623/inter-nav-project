from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics,transform
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for f in sorted(R.glob('fixedpose_*_trials/**/result.json')):
 d=json.loads(f.read_text());print('TRIAL',f.relative_to(R),{n:d.get(n) for n in ['success','failure','error','min_hold_lift_m','max_hold_tilt_deg','max_frozen_joint_error_rad','simulated_seconds']},flush=True)
for f in sorted(R.glob('fixedpose_search_*.json')):
 d=json.loads(f.read_text());print('SEARCH',f.name,len(d),sum(x['passed'] for x in d),[x['asset'] for x in d if not x['passed']],flush=True)
body=dict(zip([n for n in k.by_name if 'body_joint' in n],np.radians([-55,150,-80,0,0])))
fk=k.fk(body,transform((0,0,.04)))
print('POSTURE',body,'shoulder',fk['arm_l_link1'][:3,3].tolist(),'torso_tilt',Envelope().check(body)['torso_tilt_deg'])
