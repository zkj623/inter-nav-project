from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from plan_grasps import mesh,bvh
from g2_kinematics import Kinematics
import fcl,trimesh
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for aid,indices in [('chair_dining',[0,1,12]),('chair_wheeled',[0,2,14])]:
 d=json.loads((R/'strategy15_lowpose_plans'/(aid+'.json')).read_text())
 for ix in indices:
  p=d['plans'][ix];a,b=p['q_pre'],p['q_grasp'];step=3 if aid=='chair_dining' and ix<12 else 0
  u=step/4;q={n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in set(a)|set(b)};fk=k.fk(q,np.array(p['base_pose']))
  ob=fcl.CollisionObject(bvh(mesh(aid)),fcl.Transform(Rotation.from_euler('z',p.get('object_yaw_deg',0),degrees=True).as_matrix(),np.array(p['object_position'])))
  out={}
  for name in ['arm_r_link7','body_link2','body_link4','body_link5']:
   a=np.load(R/'robot_assets'/(name+'.npz'));m=trimesh.Trimesh(a['vertices'],a['faces'],process=False)
   t=fk[name];fb=fcl.CollisionObject(bvh(m),fcl.Transform(t[:3,:3],t[:3,3]));res=fcl.CollisionResult();fcl.collide(fb,ob,fcl.CollisionRequest(),res)
   gap=fcl.distance(fb,ob,fcl.DistanceRequest(),fcl.DistanceResult());out[name]={'visible_collision':res.is_collision,'gap_mm':round(gap*1000,2)}
  print(aid,ix,out,flush=True)
