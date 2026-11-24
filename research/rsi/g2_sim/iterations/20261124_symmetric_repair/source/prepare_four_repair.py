from pathlib import Path
import json,hashlib,ast,copy,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v1';O.mkdir(exist_ok=True)
assets=['basket_rectangular','box_white','box_reinforced','table_side']
for a in assets:
 d=json.loads((R/'realpose_plans'/(a+'.json')).read_text());p=d['plans'][0];p['repair_revision']='v1 softer damped grippers; reinforced box lower contact band'
 if a=='box_reinforced':
  k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');base=np.array(p['base_pose']);tcp=np.array(p['tcp'])
  def shift(q):
   fk=k.fk(q,base);out=q.copy()
   for side in ['l','r']:
    t=fk[f'arm_{side}_end_link']@tcp;t[0,3]-=.02;t[2,3]-=.01;out,pe,re=k.solve(side,t,out,base,tcp);assert pe<.0015 and re<.015,(pe,re)
   return out
  for key in ['q_pre','q_grasp','q_close']:p[key]=shift(p[key])
  for key in ['q_approach','q_lift']:p[key]=[shift(q) for q in p[key]]
 (O/(a+'.json')).write_text(json.dumps(d,indent=2))
 src=R/f'realpose_{a}_trial.py';s=(R.parent/'g2-wrist-symmetric-20261124'/src.name).read_text();s=s.replace("else 20.)","else 10.)").replace("else 1.)","else 4.)")
 s=s.replace("else 20. if 'gripper' in n","else 10. if 'gripper' in n").replace("else 1. if 'gripper' in n","else 4. if 'gripper' in n")
 s=s.replace("result['drive_assumption']=", "result['repair_gripper_gains']={'kp':10.,'kd':4.,'effort_limits':'unchanged'}\n    result['drive_assumption']=")
 ast.parse(s);src.write_text(s);b=src.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b)
print('PREPARED_REPAIR_V1')
