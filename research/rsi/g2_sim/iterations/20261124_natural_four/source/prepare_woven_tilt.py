from pathlib import Path
import json,numpy as np,copy
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'natural_v1/basket_woven.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);original=k.fk(p['q_grasp'],base)['arm_r_end_link']@tcp
for deg in [15,30,45]:
 v=copy.deepcopy(p);seed=p['q_start'].copy();turn=Rotation.from_euler('x',deg,degrees=True).as_matrix();delta=np.array(p['object_position'])+[0,-.1825,.34]-original[:3,3]
 def solve(q):
  global seed
  t=k.fk(q,base)['arm_r_end_link']@tcp;t[:3,3]+=delta;t[:3,:3]=turn@t[:3,:3];x,pe,re=k.solve('r',t,seed,base,tcp);assert pe<.002 and re<.02,(deg,pe,re)
  for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{21+i}_arm_l_joint{i+1}']=sg*x[f'idx{61+i}_arm_r_joint{i+1}']
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 try:
  for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key])
  v['q_lift']=[solve(q) for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()];v['posture_revision']=f'woven lower rim, outward wrist approach tilt{deg}, mirror path and anchored IK'
  O=R/f'woven_tilt{deg}';O.mkdir(exist_ok=True);(O/'basket_woven.json').write_text(json.dumps(dict(asset='basket_woven',plans=[v]),indent=2));print('READY',deg,flush=True)
 except AssertionError as e:print('UNREACHABLE',str(e),flush=True)
