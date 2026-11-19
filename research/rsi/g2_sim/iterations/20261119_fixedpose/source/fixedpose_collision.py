from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
space={'__file__':str(R/'strategy15_support_audit_native_self.py')}
source=(R/'strategy15_support_audit_native_self.py').read_text().split('for aid in sys.argv[1:]:')[0]
exec(compile(source,'native_collision_library','exec'),space)
from cooked_geometry import CookedGeometry,ConvexPart
k=space['k'];robot=space['robot'];cache={}
def check(aid,p):
 if aid not in cache:
  provenance=R/'object_assets'/(aid+'_cache_provenance.json')
  path=json.loads(provenance.read_text())['path'] if provenance.exists() else R/'object_assets'/(aid+'_cooked.json')
  if aid=='stool_rectangular':path=R/'strategy15_surface_assets'/(aid+'_cooked.json')
  d=json.loads(Path(path).read_text());cache[aid]=CookedGeometry([ConvexPart(h['vertices_object_m'],str(i)) for i,h in enumerate(d['hulls'])])
 obj=cache[aid];a,b=p['q_pre'],p['q_grasp'];names=set(a)|set(b)
 traj=[({n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in names},0.,True) for u in np.linspace(0,1,7)]
 traj += [(p['q_close'],0.,False)]+[(q,.025*(i+1),False) for i,q in enumerate(p['q_lift'])]
 for step,(q,dz,fingers) in enumerate(traj):
  hit=space['collision'](q)
  if hit:return dict(kind='native_self',step=step,hit=hit)
  fk=k.fk(q,np.array(p['base_pose']));pos=np.array(p['object_position'])+[0,0,dz];rot=Rotation.from_euler('z',p.get('object_yaw_deg',0),degrees=True).as_matrix()
  for name,g in robot.items():
   if not fingers and name.startswith('gripper_'):continue
   pose=fk[name].copy();pose[:3,3]=rot.T@(pose[:3,3]-pos);pose[:3,:3]=rot.T@pose[:3,:3]
   center=pose[:3,:3]@g.bounds.mean(0)+pose[:3,3];extent=np.abs(pose[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
   if np.any(center+extent<obj.bounds[0]) or np.any(center-extent>obj.bounds[1]):continue
   gap=obj.compound_gap(g,pose)
   if gap<-.001:return dict(kind='object',step=step,link=name,gap_m=gap)
 return None
