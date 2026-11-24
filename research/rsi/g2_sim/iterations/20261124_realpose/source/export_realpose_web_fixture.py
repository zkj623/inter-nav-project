from pathlib import Path
import json
from g2_kinematics import Kinematics,transform
from fast_gravity_compensation import GravityCompensation
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');g=GravityCompensation();q=json.loads((R/'ground_lift_training_config.json').read_text())['common_initial_joint_positions_rad'];fk=k.fk(q,transform((0,0,.04)));f=dict(name='real-low',q=q,poses={n:fk[n].tolist() for n in ['body_link1','body_link2','body_link3','head_link3','arm_l_end_link','arm_r_end_link','gripper_l_inner_link1','gripper_r_inner_link1']},torques=g.torques(q,transform((0,0,.04))));(R/'realpose_web_fixture.json').write_text(json.dumps(f))
