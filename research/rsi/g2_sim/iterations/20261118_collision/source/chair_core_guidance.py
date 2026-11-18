"""Smooth approximate avoidance for IK; exact meshes still decide acceptance."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from plan_grasps import mesh
from audit_self_collision import near,samehead
R=Path(__file__).resolve().parent
class Guidance:
 def __init__(self,aid):
  m=mesh(aid);rng=np.random.default_rng(6018)
  faces=rng.choice(len(m.faces),size=24000,p=m.area_faces/m.area_faces.sum())
  uv=rng.random((len(faces),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
  tri=m.vertices[m.faces[faces]];samples=tri[:,0]+uv[:,0,None]*(tri[:,1]-tri[:,0])+uv[:,1,None]*(tri[:,2]-tri[:,0])
  self.tree=cKDTree(np.vstack([samples,m.vertices]));self.local=[];self.radii=[];self.links=[]
  for p in sorted((R/'robot_assets').glob('*.npz')):
   n=p.stem
   if n.startswith('gripper_') or n.endswith('end_link'):continue
   v=np.load(p)['vertices'];lo=v.min(0);hi=v.max(0);center=(lo+hi)/2;half=(hi-lo)/2;axis=int(np.argmax(half))
   rad=max(.012,min(.065,.70*float(np.min(np.delete(half,axis)))))
   span=max(0.,half[axis]-rad*.5);count=max(1,min(5,int(np.ceil(2*span/max(rad,.02)))+1))
   for a in np.linspace(-span,span,count):
    c=center.copy();c[axis]+=a;self.links.append(n);self.local.append(c);self.radii.append(rad)
  self.local=np.array(self.local);self.radii=np.array(self.radii)
  pairs=[(i,j) for i,a in enumerate(self.links) for j,b in enumerate(self.links[:i]) if a!=b and not near(a,b) and not samehead(a,b) and (a.startswith('arm_') or b.startswith('arm_')) and a!='base_link' and b!='base_link']
  self.pairs=np.array(pairs);self.set_pose([0,0,0],0)
 def set_pose(self,position,yaw):
  self.position=np.array(position);self.rotation=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
 def residual(self,poses):
  pts=np.array([poses[n][:3,:3]@c+poses[n][:3,3] for n,c in zip(self.links,self.local)])
  distance=self.tree.query((pts-self.position)@self.rotation,workers=1)[0]
  obj=4*np.maximum(0.,self.radii+.008-distance)
  a,b=self.pairs.T;dist=np.linalg.norm(pts[a]-pts[b],axis=1)
  own=1.5*np.maximum(0.,self.radii[a]+self.radii[b]-.015-dist)
  return np.r_[obj,own]
