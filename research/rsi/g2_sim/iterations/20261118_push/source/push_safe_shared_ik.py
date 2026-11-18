"""Choose one torso pose jointly for pregrasp and the end of a horizontal push."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
def solve_shared(k,env,base,guide,pretargets,hightargets,seed,tcp,pos,yaw):
 body=[n for n,j in k.by_name.items() if 'body_joint' in n and j['kind']=='revolute']
 arms=[n for n,j in k.by_name.items() if 'arm_' in n and '_joint' in n and j['kind']=='revolute']
 names=body+arms+arms;nb=len(body);na=len(arms)
 x0=np.array([seed.get(n,0.) for n in names]);lo=np.array([k.by_name[n]['lower'] for n in names]);hi=np.array([k.by_name[n]['upper'] for n in names]);lo[3:5]=[-.30,-1.4];hi[3:5]=[.30,1.4];lo[:nb]=np.maximum(lo[:nb],x0[:nb]-np.radians(3.));hi[:nb]=np.minimum(hi[:nb],x0[:nb]+np.radians(3.));x0=np.clip(x0,lo+1e-6,hi-1e-6)
 def unpack(x,i):return {**seed,**dict(zip(body,x[:nb])),**dict(zip(arms,x[nb+i*na:nb+(i+1)*na]))}
 def residual(x):
  out=[]
  for i,(targets,dz) in enumerate([(pretargets,0.),(hightargets,.15)]):
   q=unpack(x,i);fk=k.fk(q,base)
   for side,t in targets.items():
    actual=fk[f'arm_{side}_end_link']@tcp
    out.extend(4*(actual[:3,3]-t[:3,3]));out.extend(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec())
   guide.set_pose(np.array(pos)+[dz,0,0],yaw);out.extend(guide.residual(fk))
  tilt=np.arccos(np.clip((fk['body_link5'][:3,:3]@env.upright.T)[2,2],-1,1))
  return np.r_[out,.001*(x-x0),10*max(0.,tilt-np.radians(18.))]
 fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=180,ftol=1e-6,xtol=1e-6,gtol=1e-6)
 qs=[unpack(fit.x,i) for i in range(2)];errors=[]
 for q,targets in zip(qs,[pretargets,hightargets]):
  fk=k.fk(q,base)
  for side,t in targets.items():
   actual=fk[f'arm_{side}_end_link']@tcp;pe=np.linalg.norm(actual[:3,3]-t[:3,3]);re=np.linalg.norm(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec());errors.append([float(pe),float(re)])
 if any(pe>.004 or re>.04 for pe,re in errors):return None,dict(errors=errors,nfev=fit.nfev)
 return qs,dict(errors=errors,nfev=fit.nfev)
