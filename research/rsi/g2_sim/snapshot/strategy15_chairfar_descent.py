from pathlib import Path
import json,numpy as np,sys
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;p=R/'strategy15_chairfar_validated_plans'/((sys.argv[1] if len(sys.argv)>1 else 'basket_fabric')+'.json');d=json.loads(p.read_text())
for plan in d['plans']:
 pre=plan['q_pre'];start={n:(0. if 'body_joint' in n else v) for n,v in pre.items()};path=[{n:(1-u)*start[n]+u*pre[n] for n in pre} for u in np.linspace(0,1,25)]
 hits=[dict(step=i,pairs=intersections(q)) for i,q in enumerate(path) if intersections(q)]
 plan['descent_self_audit']=hits
 if not hits:plan['q_start']=start
print(json.dumps([dict(mode=x['mode'],candidate=x['candidate'],descent_collisions=x['descent_self_audit']) for x in d['plans']],indent=2));p.write_text(json.dumps(d,indent=2))

