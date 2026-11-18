from pathlib import Path
import json,collections
R=Path(__file__).resolve().parent
s=json.loads((R/'strategy15_status.json').read_text())
print(json.dumps({'counts':dict(collections.Counter(r['status'] for r in s)),'states':[{k:r.get(k) for k in ('asset','status','last_planner','candidates_so_far','validated_candidates','error') if r.get(k) is not None} for r in s]},indent=2))
for p in sorted((R/'strategy15_trials').glob('*/*/*/result.json')):
 d=json.loads(p.read_text());print(json.dumps({'path':str(p.relative_to(R)),**{k:d.get(k) for k in ('success','failure','error','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction','max_contact_surface_error_m')},'strategy':(d.get('plan') or {}).get('strategy')}))
