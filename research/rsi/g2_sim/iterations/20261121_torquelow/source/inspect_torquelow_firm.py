from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
for f in R.glob('torquelow_firm_trials/**/result.json'):
 d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());s=ss[-1];print(f.relative_to(R),d['failure'],'tilt',s['tilt_deg']);print('BODY',[(n,round(s['joint_positions'][n],6),round(s['joint_targets'][n],6),round(s['joint_positions'][n]-s['joint_targets'][n],6),round(s['gravity_torques_nm'].get(n,0.),3)) for n in s['joint_positions'] if 'body_joint' in n]);print('FINGERS',{n:round(v,2) for n,v in s['finger_forces'].items() if v>.1});print('SELF',json.loads(f.with_name('self_contacts.json').read_text())[-2:])
