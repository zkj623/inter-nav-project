from pathlib import Path
import json
R=Path(__file__).resolve().parent
for folder in sorted(R.glob('caseyaw*plans')):
 f=folder/'case_handheld.json'
 if not f.exists():continue
 for i,p in enumerate(json.loads(f.read_text())['plans']):print(folder.name,i,p.get('object_yaw_deg'),p.get('candidate',{}).get('object_yaw_deg'),p['object_position'],p.get('hold_gripper_q'),p.get('candidate',{}).get('tilt_deg'),flush=True)
for f in R.glob('caseyaw_trials/**/result.json'):
 d=json.loads(f.read_text());print('ACTUAL',f.parts[-4],d.get('plan',{}).get('object_yaw_deg'),d.get('plan',{}).get('object_position'),flush=True)
