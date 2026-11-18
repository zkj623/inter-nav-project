from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in sorted(R.glob('strategy15_*_trials/*/*/*/result.json')):
 if not any(v in str(p) for v in ('loadclock','levelclock','forcefeature3','forcefeature4','forcefeature5','diverse_trials')):continue
 d=json.loads(p.read_text());print(str(p.relative_to(R)),{k:d.get(k) for k in ('success','failure','min_hold_lift_m','max_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction','force_closure_q')})
 if d.get('success'):continue
 ss=json.loads(p.with_name('samples.json').read_text())
 for phase in ('force_close','lift','hold','contact_settle'):
  rows=[s for s in ss if s['phase']==phase]
  if not rows:continue
  for s in rows[-3:]:print(phase,round(s['t'],2),'lift',round(s['lift_m'],4),'tilt',round(s['tilt_deg'],1),'forces',{n:round(v,1) for n,v in s['finger_forces'].items() if n.endswith('link4')})
