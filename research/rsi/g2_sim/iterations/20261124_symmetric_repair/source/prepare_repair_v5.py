from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v5';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v4/basket_rectangular.json').read_text());p=d['plans'][0];p['q_approach']=[p['q_pre'].copy()];p['repair_revision']='v5 yaw0 rim grasp with explicit continuous low-home to pregrasp segment';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'realpose_plans/box_white.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);rr=Rotation.from_euler('y',-8,degrees=True).as_matrix()
def convert(q,stage,i=0):
 fk=k.fk(q,base);out=q.copy()
 for side in ['l','r']:
  end=fk[f'arm_{side}_end_link'].copy();center=end[:3,3]+end[:3,:3]@np.array(p['candidate']['contact_centers_local'][side]);end[:3,3]=center+rr@(end[:3,3]-center);end[:3,:3]=rr@end[:3,:3];end[2,3]-=.012
  if stage=='q_lift':end[0,3]+=.04
  out,pe,re=k.solve(side,end@tcp,out,base,tcp);assert pe<.0015 and re<.015,(stage,i,side,pe,re)
 return out
for key in ['q_pre','q_grasp','q_close']:p[key]=convert(p[key],key)
for key in ['q_approach','q_lift']:p[key]=[convert(q,key,i) for i,q in enumerate(p[key])]
p['repair_revision']='v5 support centered at original x; -8deg wrist pitch about contact center; 12mm lower contact; 40mm early forward lift'
(O/'box_white.json').write_text(json.dumps(d,indent=2));print('V5_READY')
