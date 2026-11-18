from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
rows=[]
for p in sorted(R.glob('strategy15_*_trials/*/*/*/result.json')):
 d=json.loads(p.read_text())
 if not d.get('success'):continue
 vertices=np.load(R/'object_assets'/(d['asset']+'.npz'))['vertices']
 hold=[s for s in json.loads(p.with_name('samples.json').read_text()) if s['phase']=='hold']
 clearance=[]
 for s in hold:
  w,x,y,z=s['object_quaternion'];rot=Rotation.from_quat([x,y,z,w]).as_matrix()
  clearance.append(float(np.min(vertices@rot[2,:]+s['object_position'][2])-d.get('support_height_m',0.)))
 row=dict(result=str(p.relative_to(R)),asset=d['asset'],minimum_visible_ground_clearance_m=min(clearance),passed_10mm=min(clearance)>.01)
 rows.append(row);print(row,flush=True)
(R/'ground_clearance_check.json').write_text(json.dumps(rows,indent=2))
assert all(r['passed_10mm'] for r in rows)
