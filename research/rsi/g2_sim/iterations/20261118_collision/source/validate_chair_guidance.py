from pathlib import Path
import json,numpy as np
from chair_collision_guidance import Guidance as Broad
from chair_core_guidance import Guidance as Core
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
p=R.parent/'g2-support-20261118/strategy15_supportpath_validated_plans/stool_rectangular.json';q=json.loads(p.read_text())['plans'][0]['q_grasp'];rows=[]
for variant,cls in [('broad',Broad),('core',Core)]:
 g=cls('chair_dining')
 for label,pose in [('zero',{}),('verified_stool_grasp',q)]:
  fk=k.fk(pose,transform((0,0,.04)));pts=np.array([fk[n][:3,:3]@c+fk[n][:3,3] for n,c in zip(g.links,g.local)])
  a,b=g.pairs.T;pen=g.radii[a]+g.radii[b]-.015-np.linalg.norm(pts[a]-pts[b],axis=1)
  row=dict(variant=variant,pose=label,spheres=len(g.links),max_proxy_self_overlap_m=float(max(0.,pen.max())),overlapping_proxy_pairs=int(sum(pen>0)));rows.append(row)
  if variant=='core':assert not np.any(pen>0),row
(R/'guidance_validation.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
