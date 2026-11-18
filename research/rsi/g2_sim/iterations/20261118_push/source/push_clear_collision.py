import numpy as np
from scipy.spatial.transform import Rotation
from support_collision import Checker as BaseChecker
class Checker(BaseChecker):
 def check(self,p):
  a,b=p['q_pre'],p['q_grasp'];ns=set(a)|set(b)
  rows=[({n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in ns},0.,True) for u in np.linspace(0,1,7)]
  rows += [(q,dx,False) for q,dx in zip(p['q_push'],p['push_offsets_m'])]
  rows += [(p['q_release'],p['push_offsets_m'][-1],True)]
  rz=Rotation.from_euler('z',p['object_yaw_deg'],degrees=True).as_matrix()
  for index,(q,dx,fingers) in enumerate(rows):
   fk=self.k.fk(q,np.array(p['base_pose']));pos=np.array(p['object_position'])+[dx,0,0]
   for name,g in self.robot.items():
    if not fingers and any(name.startswith(f'gripper_{side}_') for side in p['candidate']['targets']):continue
    margin=-.001 if name.startswith('gripper_') else .015
    t=fk[name].copy();t[:3,3]=rz.T@(t[:3,3]-pos);t[:3,:3]=rz.T@t[:3,:3]
    c=t[:3,:3]@g.bounds.mean(0)+t[:3,3];e=np.abs(t[:3,:3])@(np.ptp(g.bounds,axis=0)/2)
    if np.any(c+e+margin<self.obj.bounds[0]) or np.any(c-e-margin>self.obj.bounds[1]):continue
    gap=self.obj.compound_gap(g,t)
    if gap<margin:return dict(waypoint=index,link=name,gap_m=gap)
  return None
