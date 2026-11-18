"""Exact cooked-union robot/object checks for an already solved support path."""
from pathlib import Path
import json,sys,numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R.parent/'r1lite-collision-deps'),str(R.parent/'r1lite-contact-geometry-final-20261110/research/rsi/household_catalog')]
from cooked_geometry import CookedGeometry,ConvexPart
from g2_kinematics import Kinematics

class Checker:
 def __init__(self,aid):
  def load(path):
   d=json.loads(Path(path).read_text());return CookedGeometry([ConvexPart(h['vertices_object_m'],str(i)) for i,h in enumerate(d['hulls'])])
  self.k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
  self.robot={r['name']:load(R/'robot_assets'/(r['name']+'_cooked.json')) for r in json.loads((R/'robot_assets/all_link_manifest.json').read_text())}
  prov=R/'object_assets'/(aid+'_cache_provenance.json')
  path=json.loads(prov.read_text())['path'] if prov.exists() else R/'object_assets'/(aid+'_cooked.json')
  self.obj=load(R/'strategy15_surface_assets'/(aid+'_cooked.json') if aid=='stool_rectangular' else path)
 def check(self,plan):
  a,b=plan['q_pre'],plan['q_grasp'];names=set(a)|set(b)
  rows=[({n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in names},0.,True) for u in np.linspace(0,1,7)]
  rows += [(q,.025*(i+1),False) for i,q in enumerate(plan['q_lift'])]
  rz=Rotation.from_euler('z',plan.get('object_yaw_deg',0.),degrees=True).as_matrix()
  for index,(q,dz,check_fingers) in enumerate(rows):
   fk=self.k.fk(q,np.array(plan['base_pose']));position=np.array(plan['object_position'])+[0,0,dz]
   for name,g in self.robot.items():
    if not check_fingers and name.startswith('gripper_'):continue
    pose=fk[name].copy();pose[:3,3]=rz.T@(pose[:3,3]-position);pose[:3,:3]=rz.T@pose[:3,:3]
    center=g.bounds.mean(0);ext=np.ptp(g.bounds,axis=0)/2;center=pose[:3,:3]@center+pose[:3,3];ext=np.abs(pose[:3,:3])@ext
    if np.any(center+ext<self.obj.bounds[0]) or np.any(center-ext>self.obj.bounds[1]):continue
    gap=self.obj.compound_gap(g,pose)
    if gap<-.001:return dict(waypoint=index,link=name,gap_m=gap)
  return None
