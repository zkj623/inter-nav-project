from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from support_collision import Checker

def filter_candidates(aid,candidates,opening=-.785):
 checker=Checker(aid);fk=checker.k.fk({'idx31_gripper_l_inner_joint1':opening,'idx71_gripper_r_inner_joint1':opening})
 local={};rejected=Counter();out=[]
 for side in ('l','r'):
  inv=np.linalg.inv(fk[f'arm_{side}_end_link'])
  for name in checker.robot:
   if name.startswith(f'gripper_{side}_') or name==f'arm_{side}_link7':local[name]=(side,inv@fk[name])
 cache={}
 for c in candidates:
  rz=Rotation.from_euler('z',c['yaw'],degrees=True).as_matrix();valid=True
  for side,key in [('l','left'),('r','right')]:
   target=np.array(c[key]['pose']);cachekey=(c['yaw'],side,tuple(target.ravel()))
   if cachekey in cache:
    if cache[cachekey]:rejected[cache[cachekey]]+=1;valid=False;break
    continue
   pre=target.copy();horizontal=pre[:2,2]/np.linalg.norm(pre[:2,2]);pre[:2,3]-=.09*horizontal;pre[2,3]-=.025
   bad=None
   for u in (0.,.5,1.):
    t=target.copy();t[:3,3]=(1-u)*pre[:3,3]+u*target[:3,3]
    for name,(s,lp) in local.items():
     if s!=side:continue
     pose=t@lp;pose[:3,3]=rz.T@pose[:3,3];pose[:3,:3]=rz.T@pose[:3,:3];g=checker.robot[name]
     center=pose[:3,:3]@g.bounds.mean(0)+pose[:3,3];ext=np.abs(pose[:3,:3])@(np.ptp(g.bounds,axis=0)/2)
     if np.any(center+ext<checker.obj.bounds[0]) or np.any(center-ext>checker.obj.bounds[1]):continue
     if checker.obj.compound_gap(g,pose)<-.001:bad=name;break
    if bad:break
   cache[cachekey]=bad
   if bad:rejected[bad]+=1;valid=False;break
  if valid:out.append(c)
 return out,dict(rejected)
