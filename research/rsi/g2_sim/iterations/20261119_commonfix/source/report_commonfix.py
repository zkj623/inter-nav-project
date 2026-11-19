from pathlib import Path
import json
R=Path(__file__).resolve().parent
for f in sorted(R.glob('commonfix*_trials/**/result.json')):
 d=json.loads(f.read_text());print(str(f.relative_to(R)),{k:d.get(k) for k in ['success','failure','max_lift_m','min_hold_lift_m','hold_four_finger_contact_fraction']},flush=True)
f=R/'commonstart_extra_plans/stool_rectangular.json'
if f.exists():print('stool approach waypoints',len(json.loads(f.read_text())['plans'][0]['q_approach']))
