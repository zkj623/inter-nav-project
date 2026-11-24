from pathlib import Path
import json,numpy as np
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;p=json.loads((R/'standby_preparation_config.json').read_text());a=p['standby_joint_positions_rad'];b=p['work_joint_positions_rad'];hits=[]
for u in np.linspace(0,1,61):
 w=u**3*(10-15*u+6*u*u);q={n:(1-w)*a.get(n,0)+w*b.get(n,0) for n in set(a)|set(b)};h=intersections(q)
 if h:hits.append(dict(progress=float(u),pairs=h))
report=dict(samples=61,visible_self_collision_free=not hits,hits=hits,scope='geometry only, not a physical transition verification');(R/'realpose_preparation_geometry.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
