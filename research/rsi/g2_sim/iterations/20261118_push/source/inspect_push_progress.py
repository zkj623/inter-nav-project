from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in sorted(R.glob('strategy15_push*_validated_plans/*.json')):
 d=json.loads(p.read_text());print('PLANS',p.parent.name,d['asset'],len(d['plans']),d.get('rejections'))
for p in sorted(R.glob('strategy15_push*_trials/*/*/*/result.json')):
 d=json.loads(p.read_text());print('TRIAL',str(p.relative_to(R)),{k:d.get(k) for k in ['success','failure','error','push_forward_m','max_push_tilt_deg','max_hand_force_n','simulated_seconds']})
 if not d['success']:
  s=json.loads(p.with_name('samples.json').read_text());print('LAST', {k:s[-1].get(k) for k in ['phase','t','base_position','base_tilt_deg','max_tracking_error_rad','finger_forces']} if s else 'none')
