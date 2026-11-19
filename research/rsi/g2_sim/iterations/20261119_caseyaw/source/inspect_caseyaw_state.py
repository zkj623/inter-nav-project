from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
files=list(R.glob('caseyaw_trials/**/result.json'))+[R.parent/'g2-commonfix-20261119/commonfix_solver_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json']
for f in files:
 d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());p=d['plan'];q=ss[0]['object_quaternion'];rot=Rotation.from_quat(np.array(q)[[1,2,3,0]]).as_euler('xyz',degrees=True);fk=k.fk(p['q_grasp'],np.array(p['base_pose']))
 print(str(f), 'planned_yaw',p.get('object_yaw_deg'),'actual_initial_euler',rot.tolist(),'grasp_wrist',[fk[f'arm_{s}_end_link'][:3,3].tolist() for s in ('l','r')],flush=True)
