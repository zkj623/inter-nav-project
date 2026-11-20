from pathlib import Path
import json
R=Path(__file__).resolve().parent
for f in sorted(R.glob('restpose_v2_trials/**/result.json')):
 d=json.loads(f.read_text());print(str(f.relative_to(R)),{k:d.get(k) for k in ['test_type','success','failure','standby_max_tracking_error_rad','prepared_max_tracking_error_rad','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction']})
for f in sorted(R.glob('restpose_v2_trials/**/live.json')):
 if not f.with_name('result.json').exists():print('ACTIVE',str(f.relative_to(R)),json.loads(f.read_text()))
