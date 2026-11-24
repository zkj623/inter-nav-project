from pathlib import Path
import json,shutil,numpy as np
R=Path(__file__).resolve().parent;A=R/'head25_archive';A.mkdir(exist_ok=False)
for n in ['ground_lift_training_config.json','standby_preparation_config.json','realpose_pilot_validation.json','realpose_profile.json']:shutil.copy2(R/n,A/n)
shutil.copytree(R/'realpose_plans',A/'realpose_plans')
value=float(np.radians(27));home=json.loads((R/'ground_lift_training_config.json').read_text());home['common_initial_joint_positions_rad']['idx13_head_joint3']=value;home['fixed_joint_positions_rad']['idx13_head_joint3']=value;home['head_joint_positions_deg']['idx13_head_joint3']=27.;home['camera_profile']['head_pitch_deg']=27;home['camera_profile']['head_selection_reason']='25deg clipped rectangular stool model by3px;27deg chosen for more image margin while retaining about2.97deg neck margin.';(R/'ground_lift_training_config.json').write_text(json.dumps(home,indent=2))
prep=json.loads((R/'standby_preparation_config.json').read_text());prep['work_joint_positions_rad']['idx13_head_joint3']=value
for i,w in enumerate(prep['preparation_waypoints']):
 u=(i+1)/len(prep['preparation_waypoints']);s=u**3*(10-15*u+6*u*u);w['idx13_head_joint3']=float(s*value)
(R/'standby_preparation_config.json').write_text(json.dumps(prep,indent=2))
for p in (R/'realpose_plans').glob('*.json'):
 if p.name.endswith('_planning.json'):continue
 d=json.loads(p.read_text())
 for plan in d['plans']:
  for key in ['q_start','q_default','q_pre','q_grasp','q_close']:plan[key]['idx13_head_joint3']=value
  for key in ['q_prepare','q_approach','q_lift']:
   for q in plan.get(key,[]):q['idx13_head_joint3']=value
  plan['fixed_body']['idx13_head_joint3']=value
 p.write_text(json.dumps(d,indent=2))
p=R/'head_pilot_recorder.py';s=p.read_text();s=s.replace('head_joint_deg=25.',"head_recorder_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),head_joint_deg=float(np.degrees(result['plan']['q_start']['idx13_head_joint3']))")
p.write_text(s);b=p.read_bytes();(R/'dependency_versions'/(__import__('hashlib').sha256(b).hexdigest()+'.py')).write_bytes(b)
p=R/'verify_realpose_pilots.py';s=p.read_text();s=s.replace("assets=['case_handheld'", "import argparse\nparser=argparse.ArgumentParser();parser.add_argument('--repeat',type=int,default=1);args=parser.parse_args()\nassets=['case_handheld'");s=s.replace("/'candidate0_repeat1'", "/f'candidate0_repeat{args.repeat}'");p.write_text(s)
p=R/'render_realpose_visibility_fast.py';s=p.read_text();s=s.replace("O=R/'head_visibility_final'","O=R/'head_visibility_head27'");s=s.replace("([0,25,29] if aid=='case_handheld' else [25])","([25,27,29] if aid=='case_handheld' else [27])");(R/'render_realpose_visibility_head27.py').write_text(s)
print('HEAD27_PROFILE_UPDATED; previous head25 results archived separately')
