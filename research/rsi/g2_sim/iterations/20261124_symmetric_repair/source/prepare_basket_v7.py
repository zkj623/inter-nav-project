from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v7';O.mkdir(exist_ok=True);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
d=json.loads((R/'repair_v6/basket_rectangular.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp'])
def convert(q,stage):
 fk=k.fk(q,base);out=q.copy()
 for side in ['l','r']:
  t=fk[f'arm_{side}_end_link']@tcp;t[0,3]-=p['candidate']['left' if side=='l' else 'right']['pose'][0][3]
  if stage=='q_lift':t[0,3]+=.04
  out,pe,re=k.solve(side,t,out,base,tcp);assert pe<.0015 and re<.015,(stage,side,pe,re)
 return out
for key in ['q_pre','q_grasp','q_close']:p[key]=convert(p[key],key)
p['q_lift']=[convert(q,'q_lift') for q in p['q_lift']];p['q_approach']=[p['q_pre'].copy()]
for side in ['left','right']:p['candidate'][side]['pose'][0][3]=0.
p['repair_revision']='v7 yaw0 symmetric side-wall midpoints at x0; original pinch depth;40mm forward first lift;four-branch preload and centering'
(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2));print('BASKET_V7_READY')
