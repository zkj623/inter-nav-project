from pathlib import Path
import json,h5py,numpy as np
R=Path(__file__).resolve().parent
for i,n in [(1,2),(2,2),(4,1)]:
 T=R/f'realpose_trials/placements/p{i:04d}/box_white/candidate0_repeat{n}';D=R/f'threecam_replay_trials/placements/p{i:04d}/box_white/candidate0_repeat{n}'
 if not (D/'acceptance.json').exists():continue
 with h5py.File(T/'episode.hdf5','r') as x,h5py.File(D/'episode.hdf5','r') as y:
  error=abs(x['observations/qpos'][:]-y['observations/qpos'][:]);row=dict(index=i,actions_equal=bool(np.array_equal(x['action'][:],y['action'][:])),max_arm_qpos_error_rad=float(error[:,:14].max()),max_gripper_qpos_error_rad=float(error[:,14:].max()),first_qpos_error_rad=float(error[0].max()),verified=json.loads((D/'acceptance.json').read_text())['accepted']);print(json.dumps(row),flush=True)
