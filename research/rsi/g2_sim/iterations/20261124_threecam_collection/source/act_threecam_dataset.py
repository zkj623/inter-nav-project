"""G2 ACT samples: images[3,3,528,640], qpos[16], action[K,16], is_pad[K]."""
from pathlib import Path
import json,h5py,numpy as np,torch
from torch.utils.data import Dataset
class G2ThreeCameraDataset(Dataset):
 def __init__(self,root,split='train',chunk_size=20):
  self.root=Path(root);self.rows=[json.loads(line) for line in (self.root/'manifest.jsonl').read_text().splitlines() if json.loads(line)['split']==split];self.stats=json.loads((self.root/'normalization.json').read_text());self.chunk_size=chunk_size
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  row=self.rows[index]
  with h5py.File(self.root/row['episode'],'r') as f:
   t=int(torch.randint(len(f['action']),(1,)).item());q=f['observations/qpos'][t];remaining=min(self.chunk_size,len(f['action'])-t);action=np.zeros((self.chunk_size,16),dtype=np.float32);action[:remaining]=f['action'][t:t+remaining];pad=np.arange(self.chunk_size)>=remaining;images=[]
   for name in self.stats['camera_names']:
    im=f['observations/images/'+name][t]
    if name=='head_front':im=np.pad(im,((64,64),(0,0),(0,0)),constant_values=0)
    assert im.shape==(528,640,3);images.append(im)
  q=(q-np.array(self.stats['qpos_mean']))/np.array(self.stats['qpos_std']);action=(action-np.array(self.stats['action_mean']))/np.array(self.stats['action_std']);action[pad]=0
  return torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255.,torch.tensor(q,dtype=torch.float32),torch.tensor(action,dtype=torch.float32),torch.from_numpy(pad)
