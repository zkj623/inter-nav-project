from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
from audit_self_collision import intersections
out=ROOT/'strategy15_clear_validated_plans';out.mkdir(exist_ok=True)
aid=sys.argv[1]
doc=json.loads((ROOT/'strategy15_clear_plans'/(aid+'.json')).read_text());accepted=[];rejects=[]
for index,plan in enumerate(doc['plans']):
    a=plan['q_pre'];b=plan['q_grasp'];names=set(a)|set(b)
    qs=[{n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in names} for u in np.linspace(0,1,7)]
    qs+=[plan.get('q_close',b)]+plan['q_lift']
    bad=None
    for step,q in enumerate(qs):
        hits=intersections(q)
        if hits:bad={'original_candidate':index,'waypoint':step,'pairs':hits};break
    if bad:rejects.append(bad)
    else:accepted.append({**plan,'original_candidate':index,'self_collision_audit_waypoints':len(qs)})
doc['plans']=accepted;doc['self_collision_rejections']=rejects
(out/(aid+'.json')).write_text(json.dumps(doc,indent=2))
print(aid,'accepted',len(accepted),'rejected',len(rejects),flush=True)
