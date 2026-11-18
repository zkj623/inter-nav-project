from pathlib import Path
import json,numpy as np
from pushnear_candidates import candidates
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for n in ['base_link','body_link1','body_link2','arm_l_link2']:
 d=np.load(R/'robot_assets'/(n+'.npz'));v=d['vertices'];print('BOUNDS',n,v.min(0).tolist(),v.max(0).tolist())
for aid in ['chair_dining','chair_wheeled']:
 cs,_=candidates(aid)
 cs.sort(key=lambda c:abs(c['contact_height']-.27)+(.02 if c['wrist_roll_variant']!='original' else 0))
 for c in cs[:8]:
  print('CONTACT',aid,c['contact_height'],c['yaw'],c['wrist_roll_variant'],{s:(np.array(t)[:3,3]+[.36+c['extent_x']/2+.18,0,.002]).tolist() for s,t in c['targets'].items()})
