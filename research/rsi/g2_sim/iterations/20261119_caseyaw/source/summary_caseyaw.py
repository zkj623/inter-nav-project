from pathlib import Path
import json
R=Path(__file__).resolve().parent
for r in json.loads((R/'caseyaw_verified_results.json').read_text()):
 if 'hold_samples' in r:
  print(json.dumps({k:v for k,v in r.items() if k not in ['samples.json','review.mp4','geometry_hits']}))
  print('GEOMETRY_HITS',len(r.get('geometry_hits',[])),r.get('geometry_hits',[])[:2])
for f in sorted(R.glob('caseyaw*_trials/**/progress.json')):
 if f.with_name('result.json').exists():continue
 d=json.loads(f.read_text());print('ACTIVE',str(f.relative_to(R)),d['t'],d['phase'],round(d['lift_m'],4),round(d['tilt_deg'],2))
