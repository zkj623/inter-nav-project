from pathlib import Path
import json
R=Path(__file__).resolve().parent
for f in sorted(R.glob('symhome_v4*_trials/**/result.json')):
 d=json.loads(f.read_text());print(f.parts[-3],{k:d.get(k) for k in ['success','failure','prepared_max_tracking_error_rad','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction']})
for f in sorted(R.glob('symhome_v4*_trials/**/progress.json')):
 if not f.with_name('result.json').exists():print('ACTIVE',f.parts[-3],json.loads(f.read_text()))
