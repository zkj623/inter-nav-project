from pathlib import Path
import json,time,subprocess,sys
R=Path(__file__).resolve().parent
while True:
 f=R/'collection_progress.json'
 if f.exists() and json.loads(f.read_text()).get('complete'):break
 time.sleep(30)
subprocess.run([sys.executable,str(R/'finalize_threecam_dataset.py')],cwd=R,check=True)
from act_threecam_dataset import G2ThreeCameraDataset
for split,size in [('train',192),('validation',24),('test',24)]:
 dataset=G2ThreeCameraDataset(R/'dataset',split);assert len(dataset)==size
 for i in [0,len(dataset)//2,len(dataset)-1]:
  images,q,action,mask=dataset[i];assert images.shape==(3,3,528,640) and q.shape==(16,) and action.shape==(20,16) and mask.shape==(20,)
  import torch
  assert torch.isfinite(images).all() and torch.isfinite(q).all() and torch.isfinite(action).all()
(R/'dataset/loader_verification.json').write_text(json.dumps(dict(verified=True,splits={'train':192,'validation':24,'test':24},image_shape=[3,3,528,640],qpos_dim=16,action_shape=[20,16],scope='real frozen HDF5 samples from each split; no ACT model training has run'),indent=2));print('FROZEN_DATASET_AND_ACT_LOADER_VERIFIED',flush=True)
