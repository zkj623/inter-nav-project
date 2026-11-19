from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics,transform
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
body=dict(zip([n for n in k.by_name if 'body_joint' in n],np.radians([-50,150,-85,0,0])))
frozen={n:float(body.get(n,0)) for n,j in k.by_name.items() if j['kind'] in ('revolute','continuous') and 'arm_' not in n and 'gripper' not in n}
action=[n for n,j in k.by_name.items() if j['kind'] in ('revolute','continuous') and not j['mimic'] and ('arm_' in n or 'gripper' in n)]
fk=k.fk(body,transform((0,0,.04)))
out=dict(name='g2_ground_lift_fixed_torso_v1',task='in_place_lift_only',candidate_profile_id=3,
 fixed_joint_positions_rad=frozen,body_joint_positions_deg={n:float(np.degrees(v)) for n,v in body.items()},
 fixed_base_pose=dict(position_m=[0,0,.04],quaternion_wxyz=[1,0,0,0]),
 policy_action_joint_names=action,policy_action_dimension=len(action),policy_action_type='absolute_joint_position_targets_rad',
 shoulder_height_m=float(fk['arm_l_link1'][2,3]),torso_tilt_deg=Envelope().check(body)['torso_tilt_deg'],
 joint_lock_method='non-arm/non-gripper angular limits set to initial position; world fixed joint on root; no object attachment',
 initialization_scope='common torso/head/chassis; arms initialize from each object collision-checked pregrasp; a common arm reset/approach is not yet validated',
 excluded_from_initial_lift_set=dict(chair_dining='no verified lift; push only',chair_wheeled='no verified lift; push only',stool_small='rigid substrate unconfirmed',table_side='existing grasp/lift path not reachable in this common pose within stance search'),
 assumptions=dict(mass_kg=1.,ideal_torso_drive=True,training_not_started=True,geometry_screen_not_physics_success=True),
 plan_directory='fixedpose_selected3_validated_plans')
(R/'fixed_ground_lift_config.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
