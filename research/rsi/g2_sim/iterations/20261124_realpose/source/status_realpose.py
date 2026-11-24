from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in R.glob('realpose_trials/**/live.json'):
 d=json.loads(p.read_text());print(p.parts[-2],{k:d[k] for k in ['t','phase','lift_m','max_tracking_error_rad'] if k in d})
for p in R.glob('realpose_trials/**/result.json'):
 d=json.loads(p.read_text());print('RESULT',d['asset'],d['success'],d.get('failure'),d.get('min_hold_lift_m'))
for p in (R/'realpose_plans').glob('*_planning.json'):
 d=json.loads(p.read_text());print('PLAN',d['asset'],d['success'],d.get('object_shift_x_m',0))
