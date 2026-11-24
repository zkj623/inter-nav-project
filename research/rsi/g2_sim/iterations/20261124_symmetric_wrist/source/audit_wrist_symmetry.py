from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');cfg=json.loads((R/'ground_lift_training_config.json').read_text());q=cfg['common_initial_joint_positions_rad'];S=np.diag([1,-1,1]);base=np.array(transform((0,0,.04)));zero=k.fk({},base);report={}
for label,qq in [('current',q),('right6_same_as_left',{**q,'idx66_arm_r_joint6':q['idx26_arm_l_joint6']})]:
 fk=k.fk(qq,base);rows=[]
 for suffix in ['link1','link2','link3','link4','link5','link6','link7','end_link']:
  l='arm_l_'+suffix;r='arm_r_'+suffix
  if l not in fk:continue
  C=zero[l][:3,:3].T@S@zero[r][:3,:3];expected=S@fk[l][:3,:3]@C;err=Rotation.from_matrix(expected.T@fk[r][:3,:3]).magnitude()*180/np.pi
  rows.append(dict(link=suffix,left_xyz=fk[l][:3,3].tolist(),right_xyz=fk[r][:3,3].tolist(),mirror_position_error_m=float(np.linalg.norm(S@fk[l][:3,3]-fk[r][:3,3])),mirror_rotation_error_deg=float(err)))
 report[label]=rows
report['angles_deg']={n:float(np.degrees(v)) for n,v in q.items() if 'arm_' in n};(R/'wrist_symmetry_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
