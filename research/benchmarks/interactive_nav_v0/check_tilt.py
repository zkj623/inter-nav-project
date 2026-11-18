"""Evaluate final uprightness from already recorded quaternions; no new simulation claims."""
import argparse,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
for p in sorted(a.out.glob('S*/physics_smoke.json')):
 d=json.loads(p.read_text())
 for x in d['objects']:
  q=x['final_orientation_wxyz'];tilt=math.degrees(math.acos(max(-1,min(1,1-2*(q[1]**2+q[2]**2)))));x['tilt_degrees']=tilt;x['passed']=x['passed'] and tilt<15
 d['passed']=all(x['passed'] for x in d['objects']);p.write_text(json.dumps(d,indent=2));print(d['scene'],max(x['tilt_degrees'] for x in d['objects']),d['passed'])
