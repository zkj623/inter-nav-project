from pathlib import Path
import json
from collections import Counter
R=Path(__file__).resolve().parent
for p in sorted(R.glob('strategy15_*_trials/*/*/*/result.json')):
 d=json.loads(p.read_text());print(str(p.relative_to(R)),{k:d.get(k) for k in ('success','failure','includes_descent','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction')})
 if d.get('success'):
  c=d['plan']['candidate'];print('candidate', {k:c.get(k) for k in ('height_fraction','tilt_deg','object_yaw_deg')});continue
 ss=json.loads(p.with_name('samples.json').read_text())
 for sample in ss[-3:]:
  f=sample['finger_forces'];print(sample['phase'],round(sample['lift_m'],4),round(sample['tilt_deg'],1),{side+finger:round(sum(v for n,v in f.items() if n.startswith(f'gripper_{side}_{finger}_')),2) for side in ('l','r') for finger in ('inner','outer')})
for p in list((R/'strategy15_supportstaged_validated_plans').glob('*.json'))+list((R/'strategy15_supportcartesian_validated_plans').glob('*.json')):
 d=json.loads(p.read_text());print('DESCENT',p.parent.name,p.stem,Counter((r['reason'],r.get('phase'),r.get('link',str(r.get('pair')))) for r in d['rejections']))
