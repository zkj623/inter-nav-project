from pathlib import Path
import json,hashlib,ast,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v3';O.mkdir(exist_ok=True);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
d=json.loads((R/'repair_v2/box_white.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp'])
for i,q in enumerate(p['q_lift']):
 fk=k.fk(q,base);out=q.copy()
 for side in ['l','r']:
  t=fk[f'arm_{side}_end_link']@tcp;t[0,3]+=.04-.04*min(1.,(i+1)/4)
  out,pe,re=k.solve(side,t,out,base,tcp);assert pe<.0015 and re<.015,(i,side,pe,re)
 p['q_lift'][i]=out
p['repair_revision']='v3 40mm forward wrist displacement in first25mm lift; no object pose writes';(O/'box_white.json').write_text(json.dumps(d,indent=2));print('WHITE_V3_READY')
