from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
from posture_anchor import solve_posture
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');O=R/'compact_continuous';O.mkdir(exist_ok=True)
for a,folder in [('stool_wood_four_leg','compact_soft'),('table_side','compact_paths')]:
 d=json.loads((R/folder/(a+'.json')).read_text());p=d['plans'][0];B=np.array(p['base_pose']);tcp=np.array(p['tcp']);seed=p['q_close'].copy();qs=[]
 try:
  for q in p['q_lift']:
   t=k.fk(q,B)['arm_l_end_link']@tcp;x,pe,re=solve_posture(k,'l',t,seed,B,tcp);assert pe<.002 and re<.02,(a,pe,re)
   for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{61+i}_arm_r_joint{i+1}']=sg*x[f'idx{21+i}_arm_l_joint{i+1}']
   for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
   qs.append(x);seed=x
  p['q_lift']=qs;p['posture_revision']+='; re-solve lift continuously from final Cartesian approach posture';(O/(a+'.json')).write_text(json.dumps(d,indent=2));print('READY',a,flush=True)
 except AssertionError as e:print('FAIL',e,flush=True)
