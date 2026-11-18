import numpy as np
from scipy.spatial.transform import Rotation
from supportfork_candidates import candidates as supports
from strategy15_frame_candidates import make_candidates

def candidates(aid):
 ss,counts=supports(aid);features=make_candidates(aid);out=[]
 for support in ss:
  yaw=support['yaw'];rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix();right=[]
  for c in features:
   t=np.array(c['pose']);t[:3,3]=rz@t[:3,3];t[:3,:3]=rz@t[:3,:3]
   if t[1,3]>-.04 or t[0,3]>.08:continue
   if aid=='chair_dining':
    if c['axis']!=(1 if yaw in (90,270) else 0):continue
    t[:3,:3]=np.column_stack([[1,0,0],[0,0,-1],[0,1,0]])
   right.append({**c,'pose':t.tolist()})
  right.sort(key=lambda c:abs(np.array(c['pose'])[0,3])+.4*abs(np.array(c['pose'])[1,3]))
  for grasp in right[:3]:out.append({**support,'right':grasp,'feature':'left_under_support_right_structural_pinch'})
 return out,counts
