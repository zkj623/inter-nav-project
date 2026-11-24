from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
from posture_anchor import solve_posture
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'natural_v1/basket_rectangular.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);seed=p['q_close'].copy();out=[]
for i,q in enumerate(p['q_lift']):
 target=k.fk(q,base)['arm_r_end_link']@tcp;x,pe,re=solve_posture(k,'r',target,seed,base,tcp);assert pe<.002 and re<.02,(i,pe,re)
 for j,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{21+j}_arm_l_joint{j+1}']=sg*x[f'idx{61+j}_arm_r_joint{j+1}']
 for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
 out.append(x);seed=x;print(i,pe,re,flush=True)
p['q_lift']=out;p['posture_revision']='natural_v2 continuous lift IK seeded from preceding waypoint, mirrored joints, bounded anchored runtime feedback'
O=R/'natural_v2';O.mkdir(exist_ok=True);(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
