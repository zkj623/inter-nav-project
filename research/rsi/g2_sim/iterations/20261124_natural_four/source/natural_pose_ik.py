import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
def solve_natural(k,side,target,seed,base,tcp,local=False):
 names=[j['name'] for j in k.joints if f'arm_{side}_joint' in j['name'] and j['kind']!='fixed'];ref=np.array([seed[n] for n in names]);lo=np.array([k.by_name[n]['lower'] for n in names])+.005;hi=np.array([k.by_name[n]['upper'] for n in names])-.005
 if local:lo=np.maximum(lo,ref-.5);hi=np.minimum(hi,ref+.5)
 def residual(x):
  fk=k.fk({**seed,**dict(zip(names,x))},base);p=fk[f'arm_{side}_end_link']@tcp;el=fk[f'arm_{side}_link4'][:3,3];sh=fk[f'arm_{side}_link1'][:3,3];want_y=(1 if side=='l' else -1)*.28
  return np.r_[20*(p[:3,3]-target[:3,3]),2*Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).as_rotvec(),.08*(el[1]-want_y),.08*max(0.,el[2]-sh[2]+.1),.001*(x-ref)]
 fit=least_squares(residual,np.clip(ref,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=240,ftol=1e-9,xtol=1e-9,gtol=1e-9);q={**seed,**dict(zip(names,fit.x))};p=k.fk(q,base)[f'arm_{side}_end_link']@tcp
 return q,float(np.linalg.norm(p[:3,3]-target[:3,3])),float(Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).magnitude())
