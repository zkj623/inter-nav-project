from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for variant,pd in [('pushsafestart','pushsafe'),('pushshortstart','pushshort')]:
 root=R/f'strategy15_{variant}_trials'/f'strategy15_{pd}_validated_plans/chair_dining/candidate0_repeat1';ss=json.loads((root/'samples.json').read_text());p=json.loads((R/f'strategy15_{pd}_validated_plans/chair_dining.json').read_text())['plans'][0]
 print(variant)
 for s in [ss[0]]+ss[70:180:20]:
  # Base remains nearly upright; compare actual FK with the recorded base translation.
  base=transform(s['base_position']);fk=k.fk(s['joint_positions'],base);cmd=k.fk(s['joint_targets'],base)
  print(round(s['t'],2),s['phase'],'object',np.round(s['object_position'],3).tolist(),'base',np.round(s['base_position'],3).tolist(),'gripper',{n:round(v,3) for n,v in s['joint_positions'].items() if n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']},'wrist',{side:np.round(fk[f'arm_{side}_end_link'][:3,3],3).tolist() for side in ['l','r']},'command',{side:np.round(cmd[f'arm_{side}_end_link'][:3,3],3).tolist() for side in ['l','r']})
