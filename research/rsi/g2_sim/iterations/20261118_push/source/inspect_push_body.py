from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for folder,aid in [('strategy15_pushreach_validated_plans','chair_dining'),('strategy15_pushcenter_validated_plans','chair_wheeled')]:
 p=json.loads((R/folder/(aid+'.json')).read_text())['plans'][0];q=p['q_pre'];fk=k.fk(q,transform((0,0,.04)))
 print(aid,'body_deg',{n:float(np.degrees(v)) for n,v in q.items() if 'body_joint' in n})
 for n in ['body_link3','body_link4','body_link5','arm_l_link1']:
  d=np.load(R/'robot_assets'/(n+'.npz'));v=d['vertices'];vw=v@fk[n][:3,:3].T+fk[n][:3,3];print(n,'worldbounds',vw.min(0).tolist(),vw.max(0).tolist())
