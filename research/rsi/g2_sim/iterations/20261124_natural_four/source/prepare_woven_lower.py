from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'natural_v1/basket_woven.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);orig=k.fk(p['q_grasp'],base)['arm_r_end_link']@tcp
target=orig.copy();target[:3,3]=np.array(p['object_position'])+[0,-.1825,.34];delta=target[:3,3]-orig[:3,3];sign=[-1,1,-1,1,-1,1,-1]
for variant,seed_source in [('woven_lower_old','old'),('woven_lower_home','home')]:
 import copy
 out=copy.deepcopy(d);v=out['plans'][0];seed=p['q_start'].copy()
 def solve(q):
  global seed
  t=k.fk(q,base)['arm_r_end_link']@tcp;t[:3,3]+=delta;x,pe,re=k.solve('r',t,q if seed_source=='old' else seed,base,tcp);assert pe<.002 and re<.02,(variant,pe,re)
  for i,sg in enumerate(sign):x[f'idx{21+i}_arm_l_joint{i+1}']=sg*x[f'idx{61+i}_arm_r_joint{i+1}']
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key])
 v['q_lift']=[solve(q) for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()];v['posture_revision']=variant+' lower basket rim rather than upper handle, symmetric elbows and posture-anchored feedback'
 O=R/variant;O.mkdir(exist_ok=True);(O/'basket_woven.json').write_text(json.dumps(out,indent=2));print('READY',variant,delta.tolist(),flush=True)
