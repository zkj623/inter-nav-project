from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation
from plan_grasps import mesh
from support_collision import Checker
R=Path(__file__).resolve().parent
def candidates(aid):
 checker=Checker(aid);k=checker.k;m=mesh(aid);fk=k.fk({'idx31_gripper_l_inner_joint1':0.,'idx71_gripper_r_inner_joint1':0.})
 local={};front={}
 for side in ('l','r'):
  inv=np.linalg.inv(fk[f'arm_{side}_end_link']);vv=[]
  for name in checker.robot:
   if name.startswith(f'gripper_{side}_') or name==f'arm_{side}_link7':local[name]=(side,inv@fk[name])
   if name.startswith(f'gripper_{side}_'):
    t=inv@fk[name];v=mesh(name).vertices;vv.append(v@t[:3,:3].T+t[:3,3])
  vv=np.concatenate(vv);z=vv[:,2].max();front[side]=vv[vv[:,2]>z-.0004].mean(0)
 out=[];rejected={}
 for yaw in (0,180,90,270):
  rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix();mm=m.copy();mm.vertices=mm.vertices@rz.T
  for z in ((.22,.20,.24) if aid=='chair_dining' else (.22,.28,.16,.10)):
   points=[]
   for y in np.linspace(-mm.extents[1]*.46,mm.extents[1]*.46,35):
    hits,_,faces=mm.ray.intersects_location([[mm.bounds[0,0]-.1,y,z]],[[1,0,0]],multiple_hits=True)
    if not len(hits):continue
    i=int(np.argmin(hits[:,0]));normal=mm.face_normals[faces[i]]
    if normal[0]>-.5:continue
    p=hits[i]
    if any(np.linalg.norm(p-u)<.018 for u in points):continue
    points.append(p)
   left=sorted([p for p in points if p[1]>.04],key=lambda p:p[0]);right=sorted([p for p in points if p[1]<-.04],key=lambda p:p[0])
   groups=[]
   for l in left[:4]:
    for rr in sorted(right,key=lambda rr:abs(rr[1]+l[1])+abs(rr[0]-l[0]))[:2]:
     if abs(l[0]-rr[0])<.08:groups.append({'l':l,'r':rr})
   if aid=='chair_wheeled':
    groups=[{'l':p} for p in sorted(points,key=lambda p:abs(p[1]))[:4]]
   for group in groups:
    targets={};valid=True
    for side,point in group.items():
     t=np.eye(4);t[:3,:3]=np.column_stack([[0,1,0],[0,0,1],[1,0,0]])
     t[:3,3]=point-t[:3,:3]@front[side]-np.array([.003,0,0]);targets[side]=t
     for dx in (-.04,0.):
      pose=t.copy();pose[0,3]+=dx
      for name,(s,lp) in local.items():
       if s!=side:continue
       tr=pose@lp;tr[:3,3]=rz.T@tr[:3,3];tr[:3,:3]=rz.T@tr[:3,:3];g=checker.robot[name]
       center=tr[:3,:3]@g.bounds.mean(0)+tr[:3,3];ext=np.abs(tr[:3,:3])@(np.ptp(g.bounds,axis=0)/2)
       if np.any(center+ext<checker.obj.bounds[0]) or np.any(center-ext>checker.obj.bounds[1]):continue
       if checker.obj.compound_gap(g,tr)<-.001:
        rejected[name]=rejected.get(name,0)+1;valid=False;break
      if not valid:break
     if not valid:break
    if valid:out.append(dict(yaw=yaw,targets={s:t.tolist() for s,t in targets.items()},contact_points={s:p.tolist() for s,p in group.items()},extent_x=float(mm.extents[0]),contact_height=z,mode='bimanual_push' if len(group)==2 else 'single_push'))
 (R/f'pushsafe_geometry_{aid}.json').write_text(json.dumps(dict(candidates=out,hand_rejections=rejected),indent=2))
 return out,rejected
