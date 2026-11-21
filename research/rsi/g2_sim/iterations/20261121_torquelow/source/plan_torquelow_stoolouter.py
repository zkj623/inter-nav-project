from pathlib import Path
import json,copy,numpy as np
from g2_kinematics import Kinematics
from liftall_collision import check
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');a='stool_wood_four_leg';doc=json.loads((R/'torquelow_stoolvertical_plans'/f'{a}.json').read_text());p=doc['plans'][0];f=k.fk(p['q_grasp'],np.array(p['base_pose']));cp=(f['arm_l_end_link']@np.array(p['tcp']))[:3,3];pos=np.array(p['object_position']);offset=cp[:2]-pos[:2];r=float(np.linalg.norm(offset));pos[:2]=cp[:2]-offset/r*.167;p['object_position']=pos.tolist();p['candidate']['experimental_grip_distance_from_object_center_m']=.167;p['wrist_clearance_object_offset_m']=.167-r;p['scope']+='; object moved radially away to keep wrist shell clear; unchanged floor height and robot path';hit=check(a,p);print('OUTER_GRIP_GEOMETRY',hit,'offset_m',.167-r,flush=True);assert hit is None
out=R/'torquelow_stoolouter_plans';out.mkdir(exist_ok=True);(out/f'{a}.json').write_text(json.dumps(doc,indent=2))
