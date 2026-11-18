from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
rows=[]
for q in (-.785,-.6,-.4,-.3,-.2,-.15,-.1,0.):
 fk=k.fk({'idx31_gripper_l_inner_joint1':q,'idx71_gripper_r_inner_joint1':q})
 for side in ('l','r'):
  inv=np.linalg.inv(fk[f'arm_{side}_end_link']);centers=[]
  for finger in ('inner','outer'):
   name=f'gripper_{side}_{finger}_link4';v=np.load(R/'robot_assets'/(name+'.npz'))['vertices']
   centers.append((inv@fk[name]@np.r_[v.mean(0),1.])[:3])
  row=dict(q=q,side=side,distal_mesh_midpoint_local_m=np.mean(centers,axis=0).tolist(),distal_mesh_center_distance_m=float(np.linalg.norm(centers[0]-centers[1])))
  rows.append(row)
  print(q,side,np.round(row['distal_mesh_midpoint_local_m'],4).tolist(),round(row['distal_mesh_center_distance_m'],4))
(R/'gripper_center_check.json').write_text(json.dumps(rows,indent=2))
