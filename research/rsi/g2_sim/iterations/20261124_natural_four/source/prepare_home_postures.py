from pathlib import Path
import json,copy,numpy as np
from g2_kinematics import Kinematics
from posture_anchor import solve_posture
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');O=R/'natural_home';O.mkdir(exist_ok=True)
for a,side in [('basket_woven','r'),('stool_wood_four_leg','l'),('table_side','l')]:
 d=json.loads((R/'natural_v1'/(a+'.json')).read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);seed=p['q_start'].copy();v=copy.deepcopy(p)
 def solve(q,anchor=False):
  global seed
  t=k.fk(q,base)[f'arm_{side}_end_link']@tcp
  x,pe,re=(solve_posture(k,side,t,seed,base,tcp) if anchor else k.solve(side,t,seed,base,tcp));assert pe<.002 and re<.02,(a,pe,re)
  for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):
   ln=f'idx{21+i}_arm_l_joint{i+1}';rn=f'idx{61+i}_arm_r_joint{i+1}'
   if side=='l':x[rn]=sg*x[ln]
   else:x[ln]=sg*x[rn]
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 try:
  for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key])
  v['q_lift']=[solve(q,True) for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()];v['posture_revision']='natural_home mirrored contact targets with home-seeded continuous IK and bounded anchored feedback';(O/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[v]),indent=2))
  fk=k.fk(v['q_lift'][-1],base);print('READY',a,'elbow_span',abs(fk['arm_l_link4'][1,3]-fk['arm_r_link4'][1,3]),'elbow_above_shoulder',fk['arm_l_link4'][2,3]-fk['arm_l_link1'][2,3],flush=True)
 except AssertionError as e:print('FAIL',e,flush=True)
