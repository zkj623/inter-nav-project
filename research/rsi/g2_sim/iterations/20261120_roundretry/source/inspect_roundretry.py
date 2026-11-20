from pathlib import Path
import json
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-symhome-20261120')
for f in sorted(R.glob('*trials/**/bin_round_lidded/**/result.json')):
 d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text()) if f.with_name('samples.json').exists() else [];print(str(f.relative_to(R)),{n:d.get(n) for n in ['success','failure','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction']},'last',[(x['phase'],round(x['t'],2),x.get('finger_forces')) for x in ss[-1:]])
