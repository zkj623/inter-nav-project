from pathlib import Path
import ast,json,numpy as np
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent
for label,old in [('deadband','fixedpose_deadbandv2_trial.py'),('support','fixedpose_supportstable_trial.py')]:
 s=(R/old).read_text().replace("'fixedpose_deadbandv2_trials'","'marginpose_deadband_trials'").replace("'fixedpose_supportstable_trials'","'marginpose_support_trials'")
 ast.parse(s);(R/f'marginpose_{label}_trial.py').write_text(s)
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');c=json.loads((R/'fixed_ground_lift_config.json').read_text())
body=dict(zip([n for n in k.by_name if 'body_joint' in n],np.radians([-40,142,-87,0,0])))
c['name']='g2_ground_lift_margin_pose_v2';c['candidate_profile_id']=7;c['fixed_joint_positions_rad'].update(body)
c['body_joint_positions_deg']={n:float(np.degrees(v)) for n,v in body.items()};c['shoulder_height_m']=float(k.fk(body,transform((0,0,.04)))['arm_l_link1'][2,3])
c['body_joint_margin_deg']={n:float(np.degrees(min(v-k.by_name[n]['lower'],k.by_name[n]['upper']-v))) for n,v in body.items()}
assert min(c['body_joint_margin_deg'].values())>=10
c['plan_directory']='marginpose_selected7_validated_plans';c['status']='candidate pending physical verification'
rows=json.loads((R/'marginpose_search_7.json').read_text());c['geometry_screen_pass_assets']=[r['asset'] for r in rows if r['passed']]
c['geometry_screen_fail_assets']=[r['asset'] for r in rows if not r['passed']]
c['excluded_from_initial_lift_set'].pop('table_side',None)
for aid in c['geometry_screen_fail_assets']:c['excluded_from_initial_lift_set'][aid]='no valid retargeted path in current bounded search'
(R/'margin_ground_lift_config.json').write_text(json.dumps(c,indent=2))
print(json.dumps({key:c[key] for key in ['body_joint_positions_deg','body_joint_margin_deg','shoulder_height_m','geometry_screen_pass_assets','geometry_screen_fail_assets']},indent=2))
