from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os
import cv2, h5py, numpy as np
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124')
W=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--name',default='pilot4');p.add_argument('--assets',nargs='+',default=['box_white','bin_lidded','bin_pedal','basket_rectangular']);p.add_argument('--per-asset',type=int,default=2);a=p.parse_args()
D=W/'datasets'/a.name;D.mkdir(parents=True,exist_ok=True)
cfg=json.loads((R/'ground_lift_training_config.json').read_text());all_assets=list(cfg['validated_simulation_baselines'])
rows=[]
for asset in a.assets:
 candidates=[]
 for folder in ['collection_state','wide_collection_state']:
  f=R/folder/(asset+'.json')
  if f.exists():candidates+=json.loads(f.read_text())['accepted']
 # Keep two local baselines then prioritize the wider-position episodes.
 local=[r for r in candidates if 'wide' not in r['group_id']]
 wide=[r for r in candidates if 'wide' in r['group_id']]
 selected=(local[:2]+wide+local[2:])[:a.per_asset]
 assert len(selected)>=1,asset+' has no accepted data'
 for i,row in enumerate(selected):
  row=dict(row);row.update(task_id=all_assets.index(asset),split='validation' if i>=2 and i==len(selected)-1 else 'train');rows.append(row)
def convert(row):
 T=Path(row['trial']);v=json.loads((T/'acceptance.json').read_text());assert v['accepted']
 key=hashlib.sha256(str(T).encode()).hexdigest()[:16];cache=W/'cache'/f'{key}.hdf5';cache.parent.mkdir(exist_ok=True)
 if not cache.exists():
  tmp=cache.with_suffix('.tmp.hdf5')
  with h5py.File(T/'episode.hdf5','r') as src,h5py.File(tmp,'w') as dst:
   n=len(src['action']);dst['qpos']=src['observations/qpos'][:];dst['action']=src['action'][:];dst['phase']=src['phase'][:]
   for name in ['head_front','wrist_left','wrist_right']:
    images=dst.create_dataset(name,(n,224,224,3),dtype='u1',chunks=(1,224,224,3),compression='lzf')
    for j in range(n):
     im=src['observations/images/'+name][j];h,w=im.shape[:2];scale=224/max(h,w);nw,nh=round(w*scale),round(h*scale);tile=np.zeros((224,224,3),np.uint8);small=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA);y=(224-nh)//2;x=(224-nw)//2;tile[y:y+nh,x:x+nw]=small;images[j]=tile
   dst.attrs['source_episode_sha256']=v['episode_sha256'];dst.attrs['source_trial']=str(T)
  tmp.replace(cache)
 result={**row,'cache':str(cache),'source_sha256':v['episode_sha256']}
 print('CACHED',row['asset'],row['index'],flush=True);return result
cv2.setNumThreads(1)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(convert,rows))
values={k:[] for k in ['qpos','action']}
for row in rows:
 if row['split']=='train':
  with h5py.File(row['cache']) as f:
   for k in values:values[k].append(f[k][:])
stats={}
for k,parts in values.items():
 x=np.concatenate(parts).astype(np.float64);stats[k+'_mean']=x.mean(0).tolist();stats[k+'_std']=np.maximum(x.std(0),.01).tolist();stats[k+'_q01']=np.quantile(x,.01,axis=0).tolist();stats[k+'_q99']=np.quantile(x,.99,axis=0).tolist()
stats.update(joint_names=cfg['policy_action_joint_names'],assets=all_assets,camera_names=['head_front','wrist_left','wrist_right'],image_size=224,image_processing='aspect-preserving resize with centered black padding; RGB',control_hz=20)
(D/'manifest.json').write_text(json.dumps(rows,indent=2));(D/'stats.json').write_text(json.dumps(stats,indent=2));(D/'scope.json').write_text(json.dumps(dict(episodes=len(rows),assets=a.assets,split_counts={s:sum(r['split']==s for r in rows) for s in ['train','validation']},purpose='Frozen accepted subset for overnight multi-object overfit and training; no claim of full 240 completion'),indent=2));print('DATASET_READY',D,flush=True)
