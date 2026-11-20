from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');p=json.loads((R/'directprep_plans/bin_round_lidded.json').read_text())['plans'][0];q=p['q_close'];fk=k.fk(q,np.array(p['base_pose']));d=np.load(R/'object_assets/bin_round_lidded.npz');v=d['vertices'];print('object',v.min(0),v.max(0))
for z in [.12,.18,.22,.26,.30,.32,.36]:
 vv=v[abs(v[:,2]-z)<.012];print('slice',z,len(vv),vv.min(0).tolist() if len(vv) else None,vv.max(0).tolist() if len(vv) else None)
for side in ('l','r'):
 normal=np.array([0.,-1 if side=='l' else 1,0]);patches=[];w=fk[f'arm_{side}_end_link'];inv=np.linalg.inv(w)
 for finger in ('inner','outer'):
  n=f'gripper_{side}_{finger}_link4';verts=np.load(R/'robot_assets'/(n+'.npz'))['vertices'];world=verts@fk[n][:3,:3].T+fk[n][:3,3];dep=world@normal;patch=world[dep>dep.max()-.0002].mean(0);patches.append(patch);print(side,finger,'contact',patch,'local',(inv@np.r_[patch,1])[:3])
 print(side,'separation',patches[0]-patches[1],'center',np.mean(patches,axis=0),'rot',w[:3,:3])
