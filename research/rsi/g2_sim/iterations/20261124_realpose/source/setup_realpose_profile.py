from pathlib import Path
import json,shutil,numpy as np
from g2_kinematics import Kinematics,transform
from fast_gravity_compensation import GravityCompensation
R=Path(__file__).resolve().parent;OLD=R.parent/'g2-act-collect-20261122';B=R/'baseline_configs';B.mkdir(exist_ok=False)
for n in ['ground_lift_training_config.json','standby_preparation_config.json']:shutil.copy2(R/n,B/n)
if not (R/'home_floor_assets').exists():(R/'home_floor_assets').symlink_to(OLD/'home_floor_assets',target_is_directory=True)
home=json.loads((R/'ground_lift_training_config.json').read_text());prep=json.loads((R/'standby_preparation_config.json').read_text())
body=dict(zip(['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'],map(float,np.radians([-59,149,-75]))));head={'idx13_head_joint3':float(np.radians(25))};oldq=home['common_initial_joint_positions_rad'];newq={**oldq,**body,**head}
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');base=transform((0,0,.04));fk=k.fk(newq,base)
home['name']='g2_ground_lift_realpose_v4';home['profile_id']='realpose_user_reported_20261124';home.pop('candidate_profile_id',None)
home['common_initial_joint_positions_rad']=newq;home['fixed_joint_positions_rad'].update(body|head);home['body_joint_positions_deg']={n:float(np.degrees(newq[n])) for n in newq if 'body_joint' in n};home['head_joint_positions_deg']={n:float(np.degrees(newq[n])) for n in newq if 'head_joint' in n}
home['body_joint_margin_deg']={n:float(np.degrees(min(newq[n]-j['lower'],j['upper']-newq[n]))) for n,j in k.by_name.items() if 'body_joint' in n}
home['minimum_body_margin_acceptance_deg']=2.9
home['margin_policy']='Real-machine-reported body posture has about3deg J1/J2 model margin; replaces the old10deg margin only for this explicitly selected profile. Original angular hard limits and grasp/hold gates retained.'
home['shoulder_height_m']=float(fk['arm_l_link1'][2,3]);home['torso_tilt_deg']=15.
home['status']='new default posture selected; geometry and physical revalidation pending; old240episode dataset is a separate frozen profile'
home['hardware_evidence']={'source':'user-reported real-machine feasible body posture','ankle_deg':-59,'knee_deg':149,'waist_deg':-75,'zero_reference_and_signs_assumed_equal_to_model':True,'head25_and_arm_pose_not_hardware_validated':True,'hardware_torque_limits_not_confirmed':True}
home['previous_config_archive']='baseline_configs/ground_lift_training_config.json'
for key in ['validated_assets','geometry_screen_pass_assets','geometry_screen_fail_assets','pending_adaptation_assets','validated_simulation_baselines','latest_baseline_verified_repeat_counts']:home[key]=[] if isinstance(home.get(key),list) else {}
home['validated_old_profile_dataset']=str(OLD/'dataset/frozen_manifest.jsonl')
home['camera_profile']={'name':'head_front','reference_frame':'head_link3','head_pitch_deg':25,'source':'local official Genie Sim scene_flat_g2_op.yaml','resolution':[640,400],'validation_scope':'pending geometry projection and rendered semantic visibility checks','home_floor':'wood_floor visual material only'}
home['preparation_trajectory']={'method':'joint-space quintic reference, newly generated; transition not physically revalidated','source':'standby_preparation_config.json/preparation_waypoints'}
prep['work_joint_positions_rad']=newq;prep['work_pose_changed']=True;prep['source_iteration']='20261124_realpose';prep['previous_work_config']='baseline_configs/ground_lift_training_config.json';prep['preparation_verified']=False
prep['end_to_end_verified_assets']=[];prep['end_to_end_tested_assets']=[];prep['other_assets_not_end_to_end_retested']=list(prep['execution']);prep['geometrically_audited_assets']=[];prep['status']='Real-machine body posture selected; new preparation and per-object revalidation pending.'
standby=prep['standby_joint_positions_rad'];waypoints=[]
for u in [.25,.5,.75,1.]:
    w=u*u*u*(10-15*u+6*u*u);waypoints.append({n:float((1-w)*standby.get(n,0)+w*newq.get(n,0)) for n in set(standby)|set(newq)})
prep['preparation_waypoints']=waypoints
for a,e in prep['execution'].items():e['previous_plan_dir']=e['plan_dir'];e['plan_dir']='realpose_plans'
for name,value in [('ground_lift_training_config.json',home),('standby_preparation_config.json',prep)]: (R/name).write_text(json.dumps(value,indent=2))
g=GravityCompensation();oldfk=k.fk(oldq,base)
report=dict(profile_id=home['profile_id'],body_deg=home['body_joint_positions_deg'],head_deg=home['head_joint_positions_deg'],min_body_margin_deg=min(home['body_joint_margin_deg'].values()),head_upper_margin_deg=float(np.degrees(k.by_name['idx13_head_joint3']['upper']-newq['idx13_head_joint3'])),upper_body_translation_m=(fk['body_link3'][:3,3]-oldfk['body_link3'][:3,3]).tolist(),robot_self_weight_torques_nm=g.torques(newq,base),old_dataset_unchanged=True,preparation_verified=False)
(R/'realpose_profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
