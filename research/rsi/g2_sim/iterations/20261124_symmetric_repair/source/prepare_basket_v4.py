from pathlib import Path
import json,hashlib,ast,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v4';O.mkdir(exist_ok=True);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');a='basket_rectangular'
d=json.loads((R/'realpose_plans'/(a+'.json')).read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);oldyaw=p.get('object_yaw_deg',0);rot=Rotation.from_euler('z',-oldyaw,degrees=True).as_matrix();pos=np.array(p['object_position']);print('OLD_YAW',oldyaw,flush=True)
def solve(q):
 fk=k.fk(q,base);out=q.copy()
 for side in ['l','r']:
  t=fk[f'arm_{side}_end_link']@tcp;t[:3,3]=pos+rot@(t[:3,3]-pos);t[:3,:3]=rot@t[:3,:3]
  out,pe,re=k.solve(side,t,out,base,tcp);assert pe<.0015 and re<.015,(side,pe,re)
 return out
for key in ['q_pre','q_grasp','q_close']:p[key]=solve(p[key])
for key in ['q_lift']:p[key]=[solve(q) for q in p[key]]
p['q_approach']=[];p['object_yaw_deg']=0.;p['repair_revision']='v4 rotate basket yaw to0 at original center for left/right rim grasp instead of front/back; symmetric robot home unchanged'
(O/(a+'.json')).write_text(json.dumps(d,indent=2))
src=R/f'realpose_{a}_trial.py';s=(R.parent/'g2-wrist-symmetric-20261124'/src.name).read_text();kp=10.;kd=2.
s=s.replace("'gripper' not in name else 20.)",f"'gripper' not in name else {kp})").replace("'gripper' not in name else 1.)",f"'gripper' not in name else {kd})")
s=s.replace("else 20. if 'gripper' in n",f"else {kp} if 'gripper' in n").replace("else 1. if 'gripper' in n",f"else {kd} if 'gripper' in n")
s=s.replace("result['drive_assumption']=",f"result['repair_gripper_gains']={{'kp':{kp},'kd':{kd},'effort_limits':'unchanged'}}\n    result['drive_assumption']=")
ast.parse(s);src.write_text(s);b=src.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('BASKET_V4_READY')
