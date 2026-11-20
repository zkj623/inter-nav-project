from pathlib import Path
import json,itertools,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');d=json.loads((R/'restpose_search.json').read_text());q=d['candidates'][0]['q'];left=[next(n for n in q if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in q if f'arm_r_joint{i}' in n) for i in range(1,8)];best=None
reflection=np.diag([1.,-1.,1.]);zero=k.fk({n:0. for n in q});local_reflection=zero['arm_l_end_link'][:3,:3].T@reflection@zero['arm_r_end_link'][:3,:3]
for signs in itertools.product([-1,1],repeat=7):
 err=0.
 for values in [[.2,-.7,.3,-1.2,.4,-.8,.5],[-.5,.5,-.4,-.7,-.3,.2,-.6]]:
  test={**{n:0. for n in q},**dict(zip(left,values)),**{r:v*s for r,v,s in zip(right,values,signs)}};fk=k.fk(test)
  err+=sum(np.linalg.norm(fk[f'arm_r_{name}'][:3,3]-fk[f'arm_l_{name}'][:3,3]*[1,-1,1]) for name in ['link2','link3','link4','link5','link6','link7','end_link'])
  err+=.05*np.linalg.norm(fk['arm_r_end_link'][:3,:3]-reflection@fk['arm_l_end_link'][:3,:3]@local_reflection)
 if best is None or err<best[0]:best=(err,signs)
assert best[0]<1e-4,best
if not (R/'restpose_search_v1.json').exists():(R/'restpose_search_v1.json').write_text(json.dumps(d,indent=2))
for c in d['candidates']:
 c['q'].update({r:c['q'][l]*s for l,r,s in zip(left,right,best[1])})
 c['minimum_arm_margin_deg']=float(min(np.degrees(min(c['q'][n]-k.by_name[n]['lower'],k.by_name[n]['upper']-c['q'][n])) for n in left+right))
 assert c['minimum_arm_margin_deg']>9.
d['mirror_signs']=best[1];d['full_end_link_mirror_audit_error_m']=best[0]
(R/'restpose_search.json').write_text(json.dumps(d,indent=2));print('FULL_MIRROR',best,flush=True)
