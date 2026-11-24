from pathlib import Path
import json,numpy as np,copy
from g2_kinematics import Kinematics
from natural_pose_ik import solve_natural
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');O=R/'natural_compact';O.mkdir(exist_ok=True)
for a,side in [('basket_woven','r'),('stool_wood_four_leg','l'),('table_side','l'),('basket_rectangular','r')]:
 d=json.loads((R/'natural_v1'/(a+'.json')).read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);v=copy.deepcopy(p);seed=p['q_start'].copy()
 def solve(q,local=False):
  global seed
  t=k.fk(q,base)[f'arm_{side}_end_link']@tcp;x,pe,re=solve_natural(k,side,t,seed,base,tcp,local);assert pe<.002 and re<.02,(a,pe,re)
  for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):
   ln=f'idx{21+i}_arm_l_joint{i+1}';rn=f'idx{61+i}_arm_r_joint{i+1}'
   if side=='l':x[rn]=sg*x[ln]
   else:x[ln]=sg*x[rn]
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 try:
  for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key])
  v['q_lift']=[solve(q,True) for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()];v['posture_revision']='natural_compact: symmetric contact targets with low/flared-elbow penalty, continuous IK and bounded anchored feedback';(O/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[v]),indent=2))
  fk=k.fk(v['q_lift'][-1],base);print('READY',a,'elbow_span',abs(fk['arm_l_link4'][1,3]-fk['arm_r_link4'][1,3]),'height',fk['arm_l_link4'][2,3]-fk['arm_l_link1'][2,3],flush=True)
 except AssertionError as e:print('FAIL',e,flush=True)
