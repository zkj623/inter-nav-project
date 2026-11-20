from pathlib import Path
import json,numpy as np,itertools
from g2_kinematics import Kinematics
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'narrowrest_candidate_config.json').read_text());q=d['standby_joint_positions_rad'];home=d['work_joint_positions_rad'];left=[next(n for n in q if f'arm_l_joint{i}' in n) for i in range(1,8)];f0=k.fk(q)
print([(n,k.by_name[n]['axis'].tolist(),k.by_name[n]['lower'],k.by_name[n]['upper']) for n in left])
for s1,s3 in itertools.product([-np.pi,np.pi],repeat=2):
 t={**q,left[0]:q[left[0]]+s1,left[1]:-q[left[1]],left[2]:q[left[2]]+s3};f=k.fk(t);pe=np.linalg.norm(f['arm_l_end_link'][:3,3]-f0['arm_l_end_link'][:3,3]);re=np.linalg.norm(Rotation.from_matrix(f0['arm_l_end_link'][:3,:3].T@f['arm_l_end_link'][:3,:3]).as_rotvec());print('EQUIV',s1,s3,pe,re,[t[n] for n in left],all(k.by_name[n]['lower']+.05<t[n]<k.by_name[n]['upper']-.05 for n in left))
