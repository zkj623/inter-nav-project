import numpy as np
FINGER_NAMES=[f'gripper_{s}_{f}_link4' for s in ('l','r') for f in ('inner','outer')]
def causal_force(samples,now):
 past=[s for s in samples if float(s['physics_time_s'])<=now]
 if not past:raise ValueError('No past contact measurement')
 selected=past[-3:];stamp=float(selected[-1]['physics_time_s'])
 if not 0<=now-stamp<=.101:raise ValueError('Stale or future contact measurement')
 values=np.array([[s['finger_forces'][n] for n in FINGER_NAMES] for s in selected],dtype=np.float64)
 if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid contact measurement')
 return values.mean(0),stamp
def encode_force(force):
 values=np.asarray(force)
 if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid contact measurement')
 return np.log1p(np.minimum(values,100.))/np.log(36.)
