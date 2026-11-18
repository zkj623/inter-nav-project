from pathlib import Path
import sys,json,numpy as np
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;aid=sys.argv[1];source=sys.argv[2]
doc=json.loads((R/source/(aid+'.json')).read_text());accepted=[];rejected=[];env=Envelope()
for i,p in enumerate(doc['plans']):
 start=p['q_pre'].copy()
 for n in start:
  if 'body_joint' in n:start[n]=0.
 checks=[env.check({n:(1-u)*start[n]+u*p['q_pre'][n] for n in start}) for u in np.linspace(0,1,25)]
 if not all(c['ok'] for c in checks):rejected.append(dict(candidate=i,reason='descent_envelope'));continue
 accepted.append({**p,'q_start':start,'descent_envelope_checks':checks,'source_plan_dir':source,'source_candidate':i,'initial_pose_scope':'standing descent and under-support'})
doc['plans']=accepted;doc['descent_envelope_rejections']=rejected
out=R/'strategy15_supportdesc_validated_plans';out.mkdir(exist_ok=True);(out/(aid+'.json')).write_text(json.dumps(doc,indent=2))
print(aid,'descent_envelope',len(accepted),flush=True)
