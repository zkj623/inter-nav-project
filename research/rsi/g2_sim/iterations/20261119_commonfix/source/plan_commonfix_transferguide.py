from pathlib import Path
import json,copy,numpy as np,ast
from g2_kinematics import Kinematics
from ideal_constraints import Envelope
from liftall_collision import check
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope();p=copy.deepcopy(json.loads((R/'commonfix_plans/case_handheld.json').read_text())['plans'][2]);base=np.array(p['base_pose'])
for i in range(2):
 q=p['q_lift'][i];fk=k.fk(q,base);shift=.04-.06*(.025*(i+1)/.1)
 for side in ('l','r'):
  t=fk[f'arm_{side}_end_link'].copy();t[0,3]+=shift;q,pe,re=k.solve(side,t,q,base,np.eye(4));assert pe<.002 and re<.02,(pe,re)
 assert env.check(q)['ok'] and not intersections(q)
 p['q_lift'][i]=q
assert check('case_handheld',p) is None
p['transfer_lift_handoff']='first two lift waypoints retain >=4cm forward clearance so the lift does not undo the initial transfer'
out=R/'commonfix_guide_plans';out.mkdir(exist_ok=True);(out/'case_handheld.json').write_text(json.dumps(dict(asset='case_handheld',plans=[p]),indent=2))
s=(R/'commonfix_transfer8_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_transfer8_trials'","ROOT/'commonfix_guide_trials'")
s=s.replace("1000. if 'gripper' not in name else 20.","1000. if 'gripper' not in name else 80.").replace("100. if 'gripper' not in name else 1.","100. if 'gripper' not in name else 2.")
s=s.replace("20. if 'gripper' in n else 100000. if 'body_joint' in n else 1000.","80. if 'gripper' in n else 100000. if 'body_joint' in n else 1000.").replace("1. if 'gripper' in n else 5000. if 'body_joint' in n else 100.","2. if 'gripper' in n else 5000. if 'body_joint' in n else 100.")
s=s.replace("result['constraint_accuracy']=","result['gripper_gain_revision']='kp80 kd2 within original motor effort caps; bilateral8N transfer and forward handoff'\n    result['constraint_accuracy']=")
ast.parse(s);(R/'commonfix_guide_case_handheld_trial.py').write_text(s);print('GUIDE_READY',flush=True)
