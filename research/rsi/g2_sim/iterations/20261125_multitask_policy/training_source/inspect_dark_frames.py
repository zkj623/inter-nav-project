from pathlib import Path
import h5py,numpy as np,json,cv2
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124');W=Path(__file__).resolve().parent
T=R/'realpose_trials/placements/p0004/stool_wood_four_leg/candidate0_repeat1'
with h5py.File(T/'episode.hdf5') as f:
 ds=f['observations/images/wrist_left'];bad=[i for i in range(len(ds)) if ds[i,::16,::16].std()<1];print('DARK_FRAMES',len(bad),bad[:30],bad[-10:],flush=True)
 for i in sorted(set([0,1,5,10,*bad[:2],*bad[-2:]])):
  canvas=np.concatenate([cv2.resize(f['observations/images/'+name][i],(320,264)) for name in ['head_front','wrist_left','wrist_right']],axis=1);cv2.imwrite(str(W/f'dark_frame_{i}.jpg'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
