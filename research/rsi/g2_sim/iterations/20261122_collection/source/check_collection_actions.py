from pathlib import Path
import h5py,json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');bad=[]
for p in R.glob('dataset/accepted/*/*.hdf5'):
 with h5py.File(p,'r') as f:
  a=f['action'][:];ns=json.loads(f.attrs['joint_names'])
  for i,n in enumerate(ns):
   if a[:,i].min()<k.by_name[n]['lower']-.0001 or a[:,i].max()>k.by_name[n]['upper']+.0001:bad.append(dict(episode=str(p),joint=n,min=float(a[:,i].min()),max=float(a[:,i].max()),bounds=[k.by_name[n]['lower'],k.by_name[n]['upper']]))
print('ACTION_TARGET_LIMIT_CHECK',json.dumps(bad),flush=True)
