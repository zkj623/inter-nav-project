from pathlib import Path
import json,numpy as np,sys
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,cache,check
R=Path(__file__).resolve().parent;a=sys.argv[1];p=json.loads((R/'realpose_plans'/(a+'.json')).read_text())['plans'][0];check(a,p);obj=cache[a];base=np.array(p['base_pose']);pos=np.array(p['object_position']);rot=Rotation.from_euler('z',p.get('object_yaw_deg',0),degrees=True).as_matrix()
trajectory=[(p['q_start'],0,False)]+[(q,0,False) for q in p.get('q_approach',[])]+[(p['q_pre'],0,False),(p['q_grasp'],0,False),(p['q_close'],0,True)]+[(q,.025*(i+1),True) for i,q in enumerate(p['q_lift'])];tested=0;hit=None
for (q0,z0,f0),(q1,z1,f1) in zip(trajectory,trajectory[1:]):
 steps=max(3,int(max(abs(q0.get(n,0)-q1.get(n,0)) for n in k.by_name if 'arm_' in n)/.06)+2)
 for u in np.linspace(0,1,steps):
  q={n:(1-u)*q0.get(n,0)+u*q1.get(n,0) for n in set(q0)|set(q1)};fk=k.fk(q,base);dz=(1-u)*z0+u*z1;tested+=1
  for name,g in robot.items():
   if (f0 or f1) and name.startswith('gripper_'):continue
   t=fk[name].copy();t[:3,3]=rot.T@(t[:3,3]-pos-[0,0,dz]);t[:3,:3]=rot.T@t[:3,:3];c=t[:3,:3]@g.bounds.mean(0)+t[:3,3];e=abs(t[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
   if np.any(c+e<obj.bounds[0]) or np.any(c-e>obj.bounds[1]):continue
   gap=obj.compound_gap(g,t)
   if gap<-.001:hit=dict(link=name,gap_m=float(gap),sample=tested);break
  if hit:break
 if hit:break
report=dict(asset=a,success=hit is None,samples=tested,hit=hit,scope='all approach/grasp/lift interpolations, <=0.06rad joint interval; finger contact allowed only after closure; model lift heights as planned')
(R/'realpose_plans'/(a+'_path_screen.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
