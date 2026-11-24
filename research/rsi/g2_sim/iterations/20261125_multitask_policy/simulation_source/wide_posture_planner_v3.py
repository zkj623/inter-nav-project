"""Adaptive bounded planning IK; physical execution checks are unchanged."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
def solve_posture(k,side,target,nominal,base,tcp):
 names=[j['name'] for j in k.joints if f'arm_{side}_joint' in j['name'] and j['kind']!='fixed'];ref=np.array([nominal[n] for n in names]);best=None
 def pose(x):return k.fk({**nominal,**dict(zip(names,x))},base)[f'arm_{side}_end_link']@tcp
 def residual(x):
  p=pose(x)
  return np.r_[10*(p[:3,3]-target[:3,3]),Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).as_rotvec(),.005*(x-ref)]
 for degrees in [20.,35.,50.]:
  radius=np.radians(degrees);lower=np.maximum([k.by_name[n]['lower'] for n in names],ref-radius);upper=np.minimum([k.by_name[n]['upper'] for n in names],ref+radius)
  x0=np.clip(ref if best is None else best[0],lower+1e-8,upper-1e-8)
  fit=least_squares(residual,x0,bounds=(lower,upper),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9);p=pose(fit.x);pe=float(np.linalg.norm(p[:3,3]-target[:3,3]));re=float(Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).magnitude());best=(fit.x,pe,re)
  if pe<=.0015 and re<=.015:break
 return {**nominal,**dict(zip(names,best[0]))},best[1],best[2]
