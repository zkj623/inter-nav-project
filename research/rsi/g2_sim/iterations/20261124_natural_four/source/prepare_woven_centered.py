from pathlib import Path
import json,numpy as np,copy,hashlib
from g2_kinematics import Kinematics
from natural_pose_ik import solve_natural
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for deg in [45,60]:
 d=json.loads((R/f'woven_handle{deg}/basket_woven.json').read_text());p=d['plans'][0];B=np.array(p['base_pose']);tcp=np.array(p['tcp']);old=k.fk(p['q_grasp'],B)['arm_r_end_link']@tcp;delta=np.array(p['object_position'])+[0,-.197,.4203]-old[:3,3];v=copy.deepcopy(p);seed=p['q_start'].copy()
 def solve(q,key):
  global seed
  t=k.fk(q,B)['arm_r_end_link']@tcp;t[:3,3]+=delta;x,pe,re=solve_natural(k,'r',t,seed,B,tcp,key=='lift');assert pe<.002 and re<.02,(deg,key,pe,re)
  for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{21+i}_arm_l_joint{i+1}']=sg*x[f'idx{61+i}_arm_r_joint{i+1}']
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key],key)
 v['q_lift']=[solve(q,'lift') for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()];v['posture_revision']=f'woven_center{deg}: handle contact line through object center, tilted wrists, bounded pinch force and anchored posture'
 O=R/f'woven_center{deg}';O.mkdir(exist_ok=True);(O/'basket_woven.json').write_text(json.dumps(dict(asset='basket_woven',plans=[v]),indent=2));print('READY',deg,flush=True)
f=R/'realpose_basket_woven_trial.py';f.write_bytes((R/'realpose_basket_rectangular_trial.py').read_bytes());data=f.read_bytes();(R/'source_versions'/(hashlib.sha256(data).hexdigest()+'.py')).write_bytes(data)
