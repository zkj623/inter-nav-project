from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in sorted(R.glob('liftfix*_trials/**/result.json')):
 d=json.loads(p.read_text())
 if d['success']:continue
 keys=['asset','failure','hold_relative_drift_m','hold_bilateral_contact_fraction','hold_four_finger_contact_fraction','min_hold_lift_m','max_hold_tilt_deg','max_loaded_visual_gap_m','max_contact_point_surface_error_m','squeeze_m','balance_deg','settled_contact_updates']
 print(str(p.relative_to(R)),{n:d.get(n) for n in keys},flush=True)
