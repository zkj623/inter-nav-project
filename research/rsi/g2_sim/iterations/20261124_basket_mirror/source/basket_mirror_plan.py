from pathlib import Path
import json,copy,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'repair_final_plans/basket_rectangular.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);M=np.diag([1.,-1.,1.]);sign=[-1,1,-1,1,-1,1,-1]
names={s:[n for n in k.by_name if f'arm_{s}_joint' in n and k.by_name[n]['kind']=='revolute'] for s in ['l','r']}
for variant in ['mirror_right','mirror_home_seed']:
 out=copy.deepcopy(d);v=out['plans'][0];seed=p['q_start'].copy();previous=None
 def solve(q,key):
  global seed
  fk=k.fk(q,base);target=fk['arm_r_end_link']@tcp
  # Preserve the intended right rim contact pose, removing accumulated IK drift.
  target[:3,:3]=np.array(p['candidate']['right']['pose'])[:3,:3]
  # Keep the verified executed TCP position; legacy candidate metadata differs from it.
  seed0=q.copy() if variant=='mirror_right' else seed.copy()
  x,pe,re=k.solve('r',target,seed0,base,tcp);assert pe<.002 and re<.02,(variant,key,pe,re)
  for ln,rn,sg in zip(names['l'],names['r'],sign):x[ln]=sg*x[rn]
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:x[n]=q[n]
  seed=x.copy();return x
 for key in ['q_pre','q_grasp','q_close']:v[key]=solve(p[key],key)
 v['q_lift']=[solve(q,'q_lift') for q in p['q_lift']];v['q_approach']=[v['q_pre'].copy()]
 pose=np.array(v['candidate']['right']['pose']);pose[:3,3]=M@pose[:3,3];pose[:3,:3]=M@pose[:3,:3]@M;v['candidate']['left']['pose']=pose.tolist()
 v['repair_revision']=variant+' symmetric joint mirror and opposite gripper yaw; unchanged low home/body/object center and force controller'
 O=R/variant;O.mkdir(exist_ok=True);(O/'basket_rectangular.json').write_text(json.dumps(out,indent=2));print('READY',variant,flush=True)
