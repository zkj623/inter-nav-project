from pathlib import Path
import numpy as np
from g2_kinematics import Kinematics
from plan_grasps import mesh
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
print('LIMIT',k.by_name['idx31_gripper_l_inner_joint1']['lower'],k.by_name['idx31_gripper_l_inner_joint1']['upper'])
for q in [0.,-.3,-.6]:
 fk=k.fk({'idx31_gripper_l_inner_joint1':q});inv=np.linalg.inv(fk['arm_l_end_link']);vv=[]
 for n in ['gripper_l_inner_link4','gripper_l_outer_link4']:
  t=inv@fk[n];v=mesh(n).vertices@t[:3,:3].T+t[:3,3];tip=v[v[:,2]>v[:,2].max()-.0004];print(q,n,'tip minmax',tip.min(0).tolist(),tip.max(0).tolist(),'mean',tip.mean(0).tolist());vv.append(tip)
