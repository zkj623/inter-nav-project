from pathlib import Path
import json
R=Path(__file__).resolve().parent
for f in sorted(R.glob('torquelow_single_trials/**/result.json')):
 d=json.loads(f.read_text());s=json.loads(f.with_name('samples.json').read_text());print(f.relative_to(R),d['success'],d.get('failure'),'last',{n:s[-1].get(n) for n in ['phase','t','lift_m','tilt_deg','finger_forces']});print('nonhand',json.loads(f.with_name('nonhand_contacts.json').read_text())[-8:]);print('env',json.loads(f.with_name('environment_contacts.json').read_text())[-4:]);print('self',json.loads(f.with_name('self_contacts.json').read_text())[-4:])
