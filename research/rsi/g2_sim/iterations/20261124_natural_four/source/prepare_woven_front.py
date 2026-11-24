from pathlib import Path
import json,numpy as np,copy
from g2_kinematics import Kinematics
from posture_anchor import solve_posture
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for shift in [.012,.0237]:
 d=json.loads((R/'woven_center45/basket_woven.json').read_text());p=d['plans'][0];B=np.array(p['base_pose']);tcp=np.array(p['tcp']);seed=p['q_start'].copy()
 def solve(q):
  global seed
  t=k.fk(q,B)['arm_r_end_link']@tcp;t[0,3]+=shift
  from natural_pose_ik import solve_natural
  x,pe,re=solve_natural(k,'r',t,seed,B,tcp);assert pe<.002 and re<.02,(shift,pe,re)
  for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{21+i}_arm_l_joint{i+1}']=sg*x[f'idx{61+i}_arm_r_joint{i+1}']
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 for key in ['q_pre','q_grasp','q_close']:p[key]=solve(p[key])
 p['q_lift']=[solve(q) for q in p['q_lift']];p['q_approach']=[p['q_pre'].copy()];p['posture_revision']=f'woven45 forward handle contact{shift}m to avoid convex braid gap; mirror compact posture and anchored feedback'
 O=R/f'woven_front{int(shift*1000)}';O.mkdir(exist_ok=True);(O/'basket_woven.json').write_text(json.dumps(d,indent=2));print('READY',O.name,flush=True)
