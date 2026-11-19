from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in sorted(R.glob('liftall*_trials/**/result.json')):
 d=json.loads(p.read_text());print('RESULT',p.relative_to(R),{k:d.get(k) for k in ['success','failure','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction','simulated_seconds']},flush=True)
p=R/'liftall_batch_events.json'
if p.exists():
 events=json.loads(p.read_text());latest={e['asset']:e for e in events if 'asset' in e};print('BATCH',latest,flush=True)
