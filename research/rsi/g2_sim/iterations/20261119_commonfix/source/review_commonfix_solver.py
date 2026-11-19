from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;rows=[]
for f in R.glob('commonfix_solver_trials/**/samples.json'):
 ss=json.loads(f.read_text());r=json.loads(f.with_name('result.json').read_text()) if f.with_name('result.json').exists() else {}
 for phase in ['common_approach','contact_refine','lift','hold']:
  vals=[s['max_mimic_error_rad'] for s in ss if s['phase']==phase and 'max_mimic_error_rad' in s]
  if vals:rows.append(dict(trial=str(f.relative_to(R)),phase=phase,count=len(vals),median_error_rad=float(np.median(vals)),max_error_rad=max(vals)))
(R/'commonfix_solver_review.json').write_text(json.dumps(rows,indent=2))
for row in rows:print(row,flush=True)
