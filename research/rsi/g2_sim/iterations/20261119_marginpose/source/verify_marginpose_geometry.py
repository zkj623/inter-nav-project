from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from fixedpose_collision import k,robot,CookedGeometry,ConvexPart
R=Path(__file__).resolve().parent;records=[]
for file in R.glob('marginpose_*_trials/**/result.json'):
 d=json.loads(file.read_text())
 if not d.get('success'):continue
 aid=d['asset'];pr=R/'object_assets'/(aid+'_cache_provenance.json');path=json.loads(pr.read_text())['path'] if pr.exists() else R/'object_assets'/(aid+'_cooked.json')
 if aid=='stool_rectangular':path=R/'strategy15_surface_assets'/(aid+'_cooked.json')
 dd=json.loads(Path(path).read_text());obj=CookedGeometry([ConvexPart(h['vertices_object_m'],str(i)) for i,h in enumerate(dd['hulls'])]);hits=[];worst=0.
 samples=json.loads(file.with_name('samples.json').read_text())
 for s in samples:
  # Root is physically anchored at the configured orientation; verify position separately.
  base=np.eye(4);base[:3,3]=s['base_position'];fk=k.fk(s['joint_positions'],base)
  rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix();pos=np.array(s['object_position'])
  for name,g in robot.items():
   if name.startswith('gripper_'):continue
   pose=fk[name].copy();pose[:3,3]=rot.T@(pose[:3,3]-pos);pose[:3,:3]=rot.T@pose[:3,:3]
   c=pose[:3,:3]@g.bounds.mean(0)+pose[:3,3];e=np.abs(pose[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
   if np.any(c+e<obj.bounds[0]) or np.any(c-e>obj.bounds[1]):continue
   gap=obj.compound_gap(g,pose);worst=min(worst,gap)
   if gap<-.001:hits.append(dict(t=s['t'],phase=s['phase'],link=name,gap_m=gap))
 row=dict(result=str(file.relative_to(R)),samples=len(samples),min_nonhand_gap_m=worst,penetration_pass=not hits,hits=hits,scope='posthoc sampled geometry, not nonhand impulse measurement')
 records.append(row);print(json.dumps(row),flush=True)
(R/'marginpose_physical_geometry_review.json').write_text(json.dumps(records,indent=2))
