from pathlib import Path
import json,sys,h5py,numpy as np
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';sys.path.insert(0,str(R));from g2_kinematics import Kinematics
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');stats=json.loads((W/'datasets/pilot4/stats.json').read_text());names=stats['joint_names'];limits=np.array([[k.by_name[n]['lower'],k.by_name[n]['upper'],k.by_name[n]['velocity']] for n in names]);rows=json.loads((W/'datasets/pilot4/manifest.json').read_text());out=[]
for row in rows:
 with h5py.File(row['cache']) as f:
  actions=f['action'][:];q=f['qpos'][:];below=np.maximum(limits[:,0]-actions,0);above=np.maximum(actions-limits[:,1],0);violation=np.maximum(below,above);velocity=abs(np.diff(actions,axis=0))*20/limits[:,2]
  out.append(dict(asset=row['asset'],trial=row['trial'],target_bound_violations={n:float(violation[:,i].max()) for i,n in enumerate(names) if violation[:,i].max()>1e-5},max_target_velocity_fraction=float(velocity.max()),max_target_actual_lag_rad=float(abs(actions-q).max()),max_arm_target_actual_lag_rad=float(abs(actions[:,:14]-q[:,:14]).max()),initial_action_qpos_error_rad=float(abs(actions[0]-q[0]).max())))
(W/'action_contract_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
