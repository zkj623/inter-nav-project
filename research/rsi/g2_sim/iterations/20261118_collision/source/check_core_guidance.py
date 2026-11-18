from pathlib import Path
import json,numpy as np
from chair_core_guidance import Guidance
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;g=Guidance('chair_dining');k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
source=R.parent/'g2-support-20261118/strategy15_supportpath_validated_plans/stool_rectangular.json'
q=json.loads(source.read_text())['plans'][0]['q_grasp']
for label,pose in [('zero',{}),('verified_stool_grasp',q)]:
 fk=k.fk(pose,transform((0,0,.04)));pts=np.array([fk[n][:3,:3]@c+fk[n][:3,3] for n,c in zip(g.links,g.local)])
 a,b=g.pairs.T;pen=g.radii[a]+g.radii[b]-.015-np.linalg.norm(pts[a]-pts[b],axis=1)
 top=sorted([(float(pen[t]),g.links[a[t]],g.links[b[t]]) for t in np.flatnonzero(pen>0)],reverse=True)[:12]
 print(label,'spheres',len(g.links),'proxy_self_overlap',top,flush=True)
