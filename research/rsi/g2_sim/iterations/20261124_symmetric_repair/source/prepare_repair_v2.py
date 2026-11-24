from pathlib import Path
import json,copy,hashlib,ast,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v2';O.mkdir(exist_ok=True);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for a in ['basket_rectangular','box_white']:
 d=json.loads((R/'realpose_plans'/(a+'.json')).read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);rot=Rotation.from_euler('z',p.get('object_yaw_deg',0),degrees=True).as_matrix()
 def shift(q,stage,i=0):
  fk=k.fk(q,base);out=q.copy()
  for side in ['l','r']:
   t=fk[f'arm_{side}_end_link']@tcp
   if a=='basket_rectangular':
    t[:3,3]+=rot@np.array([-.5*p['candidate']['left' if side=='l' else 'right']['pose'][0][3],0,.015])
   else:
    t[0,3]-=.02;t[2,3]-=.008
    if stage=='q_lift':t[0,3]+=.04*min(1.,(i+1)/4)
   out,pe,re=k.solve(side,t,out,base,tcp);assert pe<.0015 and re<.015,(a,stage,side,pe,re)
  return out
 for key in ['q_pre','q_grasp','q_close']:p[key]=shift(p[key],key)
 for key in ['q_approach','q_lift']:p[key]=[shift(q,key,i) for i,q in enumerate(p[key])]
 p['repair_revision']='v2: basket inward-along-rim contact; white lower rearward contact with forward lift clearance'
 (O/(a+'.json')).write_text(json.dumps(d,indent=2))
 src=R/f'realpose_{a}_trial.py';s=(R.parent/'g2-wrist-symmetric-20261124'/src.name).read_text();kp,kd=(5.,.5) if a=='basket_rectangular' else (20.,4.)
 s=s.replace("'gripper' not in name else 20.)",f"'gripper' not in name else {kp})").replace("'gripper' not in name else 1.)",f"'gripper' not in name else {kd})")
 s=s.replace("else 20. if 'gripper' in n",f"else {kp} if 'gripper' in n").replace("else 1. if 'gripper' in n",f"else {kd} if 'gripper' in n")
 s=s.replace("result['drive_assumption']=",f"result['repair_gripper_gains']={{'kp':{kp},'kd':{kd},'effort_limits':'unchanged'}}\n    result['drive_assumption']=")
 ast.parse(s);src.write_text(s);b=src.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b)
print('PREPARED_V2')
