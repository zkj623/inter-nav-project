from pathlib import Path
import json,numpy as np,copy
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
from natural_pose_ik import solve_natural
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'natural_v1/basket_woven.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp'])
for deg in [30,45,60]:
 v=copy.deepcopy(p);seed=p['q_start'].copy();turn=Rotation.from_euler('x',deg,degrees=True).as_matrix()
 def solve(q,key):
  global seed
  t=k.fk(q,base)['arm_r_end_link']@tcp;t[:3,:3]=turn@t[:3,:3]
  if key=='q_pre':
   g=k.fk(p['q_grasp'],base)['arm_r_end_link']@tcp;t[:3,3]=g[:3,3]-.09*t[:3,2]
  x,pe,re=solve_natural(k,'r',t,seed,base,tcp,key=='lift');assert pe<.002 and re<.02,(deg,key,pe,re)
  for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{21+i}_arm_l_joint{i+1}']=sg*x[f'idx{61+i}_arm_r_joint{i+1}']
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 try:
  for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key],key)
  v['q_lift']=[solve(q,'lift') for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()];v['posture_revision']=f'handle_tilt{deg}: mirror upper-handle pinch, outward approach to lower wrist/elbow, compact continuous IK with anchored feedback'
  O=R/f'woven_handle{deg}';O.mkdir(exist_ok=True);(O/'basket_woven.json').write_text(json.dumps(dict(asset='basket_woven',plans=[v]),indent=2));fk=k.fk(v['q_lift'][-1],base);print('READY',deg,'span',abs(fk['arm_l_link4'][1,3]-fk['arm_r_link4'][1,3]),'height',fk['arm_l_link4'][2,3]-fk['arm_l_link1'][2,3],flush=True)
 except AssertionError as e:print('FAIL',e,flush=True)
