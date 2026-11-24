"""Local IK feedback around a symmetric planned posture, without nullspace drift."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
def solve_posture(k,side,target,nominal,base,tcp):
 names=[j['name'] for j in k.joints if f'arm_{side}_joint' in j['name'] and j['kind']!='fixed'];ref=np.array([nominal[n] for n in names]);radius=np.radians(20.)
 lower=np.maximum([k.by_name[n]['lower'] for n in names],ref-radius);upper=np.minimum([k.by_name[n]['upper'] for n in names],ref+radius);x0=np.clip(ref,lower+1e-8,upper-1e-8)
 def residual(x):
  q={**nominal,**dict(zip(names,x))};p=k.fk(q,base)[f'arm_{side}_end_link']@tcp
  return np.r_[10*(p[:3,3]-target[:3,3]),Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).as_rotvec(),.01*(x-ref)]
 fit=least_squares(residual,x0,bounds=(lower,upper),max_nfev=200,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 q={**nominal,**dict(zip(names,fit.x))};p=k.fk(q,base)[f'arm_{side}_end_link']@tcp
 return q,float(np.linalg.norm(p[:3,3]-target[:3,3])),float(Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).magnitude())
