from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
from gravity_compensation import GravityCompensation
R=Path(__file__).resolve().parent;d=json.loads((R/'symhome_search.json').read_text());k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');g=GravityCompensation();a=d['default'];b=d['candidates'][0]['q'];worst={}
for u in np.linspace(0,1,121):
 q={n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in a};tt=g.torques(q)
 for n,t in tt.items():
  ratio=abs(t)/k.by_name[n]['effort']
  if n not in worst or ratio>worst[n]['ratio']:worst[n]=dict(torque_nm=t,original_effort_nm=k.by_name[n]['effort'],ratio=ratio,fraction=float(u))
out=dict(worst=worst,max_ratio=max(x['ratio'] for x in worst.values()));(R/'symhome_original_effort_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
