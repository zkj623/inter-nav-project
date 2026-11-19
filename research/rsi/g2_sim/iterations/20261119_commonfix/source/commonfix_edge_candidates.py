"""Geometric narrow-section candidates inside object-specific structural regions."""
import numpy as np
from plan_grasps import mesh
BANDS={'basket_woven':(.72,.99),'basket_rectangular':(.65,.97),'stool_wood_four_leg':(.88,.99),'stool_rectangular':(.65,.98),'table_side':(.87,.99),'chair_dining':(.32,.49),'chair_wheeled':(.08,.30),'stool_small':(.02,.18),'box_white':(.45,.90),'case_handheld':(.80,1.)}
def make_candidates(aid):
 m=mesh(aid);lo,hi=m.bounds;low,high=BANDS.get(aid,(.35,.88));out=[]
 # A tabletop/wooden seat is pinched across its VERTICAL thickness, entering
 # horizontally from outside. Horizontal-only sections cannot represent this.
 if aid in ('table_side','stool_wood_four_leg','stool_rectangular','case_handheld'):
  mid=(lo+hi)/2
  for angle in np.linspace(0,2*np.pi,32,endpoint=False):
   radial=np.array([np.cos(angle),np.sin(angle),0.])
   for radius in (.94,.99,.85):
    xy=mid[:2]+radius*.5*m.extents[:2]*radial[:2]
    hits,_,faces=m.ray.intersects_location([[xy[0],xy[1],lo[2]-.1]],[[0,0,1]],multiple_hits=True)
    order=np.argsort(hits[:,2]) if len(hits) else [];hits=hits[order];faces=faces[order]
    if len(hits)<2:continue
    a,b=hits[-2:];width=b[2]-a[2];center=(a+b)/2
    if not .004<=width<=.075 or not low<=(center[2]-lo[2])/m.extents[2]<=high:continue
    if min(abs(m.face_normals[faces[-2],2]),abs(m.face_normals[faces[-1],2]))<.45:continue
    closing=np.array([0.,0.,1.]);approach=-radial;t=np.eye(4);t[:3,:3]=np.column_stack([closing,np.cross(approach,closing),approach]);t[:3,3]=center
    out.append(dict(score=float(-10+abs(center[0])*3+abs(width-.035)),width=float(width),pose=t.tolist(),axis=2,contact_pair_object_m=[a.tolist(),b.tolist()],region_height_fraction=float((center[2]-lo[2])/m.extents[2]),feature='vertical_edge_thickness',approach_object=approach.tolist()))
 for axis in (0,1):
  other=1-axis
  for z in np.linspace(lo[2]+high*m.extents[2],lo[2]+low*m.extents[2],16):
   for v in np.linspace(lo[other]+.005,hi[other]-.005,23):
    origin=np.zeros(3);origin[axis]=lo[axis]-.1;origin[other]=v;origin[2]=z
    direction=np.eye(3)[axis];hits,_,faces=m.ray.intersects_location([origin],[direction],multiple_hits=True)
    order=np.argsort(hits[:,axis]) if len(hits) else [];hits=hits[order];faces=faces[order]
    for i in range(len(hits)-1):
     a,b=hits[i:i+2];width=b[axis]-a[axis]
     if not .004<=width<=.075:continue
     if min(abs(m.face_normals[faces[i],axis]),abs(m.face_normals[faces[i+1],axis]))<.45:continue
     center=(a+b)/2;x=direction;zz=np.array([0.,0.,-1]);t=np.eye(4);t[:3,:3]=np.column_stack([x,np.cross(zz,x),zz]);t[:3,3]=center
     if aid=='chair_wheeled' and np.linalg.norm(center[:2])<.12:continue
     score=(hi[2]-z)*3+np.linalg.norm(center[:2])*.5+abs(width-.035)
     out.append(dict(score=float(score),width=float(width),pose=t.tolist(),axis=axis,contact_pair_object_m=[a.tolist(),b.tolist()],region_height_fraction=float((z-lo[2])/m.extents[2])))
 out.sort(key=lambda c:c['score']);unique=[]
 for c in out:
  if not any(c['axis']==u['axis'] and np.linalg.norm(np.array(c['pose'])[:3,3]-np.array(u['pose'])[:3,3])<.015 for u in unique):unique.append(c)
 return unique
