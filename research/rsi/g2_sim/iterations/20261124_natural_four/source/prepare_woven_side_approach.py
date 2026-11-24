from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for angle in [60,75,90]:
 d=json.loads((R/f'woven_tilt{angle}/basket_woven.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);t=k.fk(p['q_grasp'],base)['arm_r_end_link']@tcp;t[:3,3]-=.09*t[:3,2];x,pe,re=k.solve('r',t,p['q_grasp'],base,tcp)
 if pe>.002 or re>.02:print('FAIL',angle,pe,re);continue
 for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{21+i}_arm_l_joint{i+1}']=sg*x[f'idx{61+i}_arm_r_joint{i+1}']
 for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=p['q_grasp'][n]=-.785
 p['q_pre']=x;p['q_approach']=[x.copy()];p['posture_revision']+= '; approach along outward normal, fully open grippers'
 O=R/f'woven_side{angle}';O.mkdir(exist_ok=True);(O/'basket_woven.json').write_text(json.dumps(d,indent=2));print('READY',angle,flush=True)
