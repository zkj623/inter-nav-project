"""CPU-only input-sensitivity diagnostic; does not certify recovery actions."""
from pathlib import Path
import json
import cv2,h5py,numpy as np,torch
from g2_force_act_model import make_model,normalize_images
from g2_force_features import causal_force,encode_force
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
torch.set_num_threads(2)
c=torch.load(W/'runs/act_force_v1/step_008000.pt',map_location='cpu',weights_only=False);stats=c['stats'];model=make_model();model.load_state_dict(c['model']);model.eval()
terminal=json.loads((W/'progress_terminal_limits.json').read_text());results=[]
for asset in ['bin_lidded','bin_pedal','basket_rectangular']:
 trial=R/'policy_trials/act_progress_v1_8000/placements/p0001'/asset/'candidate0_repeat1';samples=json.loads((trial/'samples.json').read_text())
 file=trial/'episode.hdf5' if (trial/'episode.hdf5').exists() else trial/'episode.partial.hdf5'
 with h5py.File(file) as f:
  i=len(f['action'])-4;forces,stamp=causal_force(samples,float(f['timestamps'][i]));images=[]
  for name in stats['camera_names']:
   im=f['observations/images/'+name][i];h,w=im.shape[:2];scale=224/max(h,w);nw,nh=round(w*scale),round(h*scale);small=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA);tile=np.zeros((224,224,3),np.uint8);y=(224-nh)//2;x=(224-nw)//2;tile[y:y+nh,x:x+nw]=small;images.append(tile)
  q=(f['observations/qpos'][i]-stats['qpos_mean'])/stats['qpos_std'];q=np.r_[q,encode_force(forces),min(i/200.,terminal[asset])]
  inputs=np.stack([q,q.copy()]);inputs[1,16:20]=0
  with torch.inference_mode():pred=model(torch.tensor(inputs,dtype=torch.float32),normalize_images(torch.tensor(np.stack(images)).permute(0,3,1,2).unsqueeze(0).repeat(2,1,1,1,1).float()/255),None,task_ids=torch.tensor([stats['assets'].index(asset)]*2))[0].numpy()
  delta=(pred[0,0]-pred[1,0])*stats['action_std'];results.append(dict(asset=asset,frame=i,forces_n=forces.tolist(),first_action_abs_change_rad=np.abs(delta).tolist(),mean_arm_change_rad=float(np.abs(delta[:14]).mean()),scope='Measured forces versus zeroed-force ablation on old failed-policy observation; no recovery ground truth'))
(W/'force_input_effect_probe.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
