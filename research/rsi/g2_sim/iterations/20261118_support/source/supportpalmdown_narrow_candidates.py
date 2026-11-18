"""Four-tip support candidates from actual distal meshes and downward-facing surfaces."""
from pathlib import Path
import json,itertools,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
from plan_grasps import mesh
R=Path(__file__).resolve().parent

def candidates(aid):
 k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');m=mesh(aid)
 fk=k.fk({'idx31_gripper_l_inner_joint1':-.2,'idx71_gripper_r_inner_joint1':-.2})
 band={'chair_dining':(.32,.55),'chair_wheeled':(.28,.58),'stool_rectangular':(.25,.42),'stool_small':(.005,.10)}[aid]
 shape={}
 for side in ('l','r'):
  inv=np.linalg.inv(fk[f'arm_{side}_end_link']);shape[side]=[]
  for finger in ('inner','outer'):
   parts=[]
   for link in (1,2,3,4):
    name=f'gripper_{side}_{finger}_link{link}';v=mesh(name).vertices;t=inv@fk[name]
    parts.append(v@t[:3,:3].T+t[:3,3])
   shape[side].append(np.concatenate(parts))
 out=[];counts={}
 for yaw in (0,90,180,270):
  rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix();mm=m.copy();mm.vertices=mm.vertices@rz.T
  choices={}
  for side,sign in [('l',1),('r',-1)]:
   opts=[]
   for x in (-.06,0.,.06):
    for frac in (.45,.65,.80,.85,.92,.98):
     y=sign*mm.extents[1]*.5*frac
     # Local X opens the fingers fore/aft, local Z approaches inward.
     xx=np.array([-1. if side=='l' else 1.,0,0]);zz=np.array([0.,-sign,0.]);base=np.column_stack([xx,np.cross(zz,xx),zz])
     for tilt,roll in itertools.product((0,10,20,35,50),np.linspace(-.24,.24,17)):
      zz=np.cos(np.radians(tilt))*np.array([0.,-sign,0.])+np.array([0.,0.,-np.sin(np.radians(tilt))]);base=np.column_stack([xx,np.cross(zz,xx),zz])
      rr=base@Rotation.from_rotvec([0,0,roll]).as_matrix();points=[]
      for verts in shape[side]:
       vv=verts@rr.T;zmax=vv[:,2].max();points.append(vv[vv[:,2]>zmax-.0002].mean(0))
      points=np.array(points);offset=points[:,:2]-points[:,:2].mean(0);hits=[];valid=True
      for off in offset:
       xy=np.array([x,y])+off
       loc,_,faces=mm.ray.intersects_location([[*xy,-.1]],[[0,0,1]],multiple_hits=True)
       good=[(p,mm.face_normals[f]) for p,f in zip(loc,faces) if band[0]<=p[2]<=band[1] and mm.face_normals[f,2]<-.60]
       if not good:valid=False;break
       hits.append(max(good,key=lambda pn:pn[0][2])[0])
      if not valid:continue
      hits=np.array(hits);wrist_heights=hits[:,2]-points[:,2]
      error=float(np.ptp(wrist_heights))
      if error>.0015:continue
      t=np.eye(4);t[:3,:3]=rr;t[:2,3]=[x,y]-points[:,:2].mean(0);t[2,3]=min(wrist_heights)-.001
      if min((v@rr.T+t[:3,3])[:,2].min() for v in shape[side])<.006:continue
      opts.append(dict(pose=t.tolist(),contacts=hits.tolist(),roll=float(roll),tilt_deg=tilt,height_error_m=error,center=[x,y],score=error+abs(x)*.03))
   opts.sort(key=lambda x:x['score']);choices[side]=opts
  counts[str(yaw)]={s:len(v) for s,v in choices.items()}
  for l in choices['l']:
   for r in sorted(choices['r'],key=lambda r:abs(r['center'][0]+l['center'][0])+abs(np.mean(np.array(r['contacts'])[:,2])-np.mean(np.array(l['contacts'])[:,2])))[:3]:
    pts=np.array(l['contacts']+r['contacts'])
    if np.ptp(pts[:,2])>.04:continue
    # Require the support polygon to cover the horizontal object origin.
    from scipy.spatial import ConvexHull
    hull=ConvexHull(pts[:,:2])
    if max(hull.equations[:,-1])>-.003:continue
    out.append(dict(yaw=yaw,left=l,right=r,extent_x=float(mm.extents[0]),score=l['score']+r['score']))
 out.sort(key=lambda c:c['score'])
 # Do not spend all IK attempts on nearly identical wrist rolls.
 selected=[]
 for c in out:
  if any(c['yaw']==u['yaw'] and c['left']['tilt_deg']==u['left']['tilt_deg'] and np.linalg.norm(np.array(c['left']['center'])-u['left']['center'])<.02 and np.linalg.norm(np.array(c['right']['center'])-u['right']['center'])<.02 for u in selected):continue
  selected.append(c)
 from support_hand_clearance import filter_candidates
 selected,rejected=filter_candidates(aid,selected,opening=-.2)
 counts["hand_clearance_rejections"]=rejected
 return selected,counts

if __name__=='__main__':
 import sys
 for aid in sys.argv[1:]:
  cs,counts=candidates(aid);(R/f'support_geometry_{aid}.json').write_text(json.dumps(dict(asset=aid,candidates=cs,counts=counts),indent=2));print(aid,'support pairs',len(cs),counts,flush=True)
