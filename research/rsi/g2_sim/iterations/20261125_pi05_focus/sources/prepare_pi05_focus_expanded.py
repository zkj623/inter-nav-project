from pathlib import Path
import json,hashlib,collections
import numpy as np,h5py,cv2
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
D=W/'datasets/focus_expanded';D.mkdir(exist_ok=True)
cfg=json.loads((R/'ground_lift_training_config.json').read_text());assets=list(cfg['validated_simulation_baselines'])
pilot=json.loads((W/'datasets/pilot4/manifest.json').read_text());pilot_sources={r['source_sha256'] for r in pilot}
rows=[];rejected=[];seen=set();positions={}
for asset in assets:
 candidates=[]
 for folder in ['collection_state','wide_collection_state']:
  p=R/folder/(asset+'.json')
  if p.exists():candidates+=json.loads(p.read_text())['accepted']
 accepted=[]
 for r in candidates:
  T=Path(r['trial'])
  try:
   a=json.loads((T/'acceptance.json').read_text());assert a['accepted'];sha=a['episode_sha256']
   if sha in seen:continue
   actual_hash=hashlib.sha256()
   with (T/'episode.hdf5').open('rb') as stream:
    for chunk in iter(lambda:stream.read(4*1024*1024),b''):actual_hash.update(chunk)
   assert actual_hash.hexdigest()==sha,'SOURCE_SHA256_CHANGED'
   with h5py.File(T/'episode.hdf5') as f:
    n=len(f['action']);assert n>60 and f['action'].shape==f['observations/qpos'].shape==(n,16)
    assert all(f['observations/images/'+c].shape[0]==n for c in ['head_front','wrist_left','wrist_right'])
    assert json.loads(f.attrs['joint_names'])==cfg['policy_action_joint_names']
    t=f['timestamps'][:];assert np.allclose(np.diff(t),.05,atol=1e-6)
    for camera in ['head_front','wrist_left','wrist_right']:
     ct=f['image_timestamps/'+camera][:];assert ct.shape==t.shape and np.max(abs(ct-t))<=1/120+.0001
    assert np.isfinite(f['action'][:]).all() and np.isfinite(f['observations/qpos'][:]).all()
   pos=tuple(round(float(r.get(k,0.)),7) for k in ['dx_m','dy_m','dyaw_deg'])
   row={**r,'source_sha256':sha,'frames':n,'position_key':pos,'task_id':assets.index(asset)}
   accepted.append(row);seen.add(sha)
  except Exception as e:rejected.append(dict(trial=str(T),reason=repr(e)))
 assert accepted,asset+' has no valid three-camera episode'
 # A holdout cannot share a position with any pretraining/pilot episode.
 pilot_positions={tuple(r['position_key']) for r in accepted if r['source_sha256'] in pilot_sources}
 candidates=[r for r in accepted if tuple(r['position_key']) not in pilot_positions]
 chosen=max(candidates,key=lambda r:abs(r.get('dx_m',0))+abs(r.get('dy_m',0))) if candidates and len(set(tuple(r['position_key']) for r in accepted))>1 else None
 held=tuple(chosen['position_key']) if chosen else None
 for r in accepted:r['split']='validation' if held is not None and tuple(r['position_key'])==held else 'train'
 assert any(r['split']=='train' for r in accepted)
 rows+=accepted
cv2.setNumThreads(1)
converted=[]
for row in rows:
 try:
  T=Path(row['trial']);key=hashlib.sha256(str(T).encode()).hexdigest()[:16];cache=W/'cache'/(key+'.hdf5');cache.parent.mkdir(exist_ok=True)
  if not cache.exists():
   tmp=cache.with_suffix('.focus.tmp.hdf5')
   with h5py.File(T/'episode.hdf5') as src,h5py.File(tmp,'w') as dst:
    n=len(src['action']);dst['qpos']=src['observations/qpos'][:];dst['action']=src['action'][:];dst['phase']=src['phase'][:]
    for camera in ['head_front','wrist_left','wrist_right']:
     images=dst.create_dataset(camera,(n,224,224,3),dtype='u1',chunks=(1,224,224,3),compression='lzf')
     for i in range(n):
      im=src['observations/images/'+camera][i];h,w=im.shape[:2];scale=224/max(h,w);nw,nh=round(w*scale),round(h*scale);tile=np.zeros((224,224,3),np.uint8);x=(224-nw)//2;y=(224-nh)//2;tile[y:y+nh,x:x+nw]=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA);images[i]=tile
    dst.attrs['source_episode_sha256']=row['source_sha256'];dst.attrs['source_trial']=str(T)
   tmp.replace(cache)
  with h5py.File(cache) as f:assert f.attrs['source_episode_sha256']==row['source_sha256']
  row['cache']=str(cache);print('CACHED',row['asset'],row['split'],row['group_id'],flush=True)
  converted.append(row)
 except Exception as ex:
  rejected.append(dict(trial=row['trial'],reason='CACHE_CONVERSION: '+repr(ex)))
  print('EXCLUDED',row['asset'],row['group_id'],repr(ex),flush=True)
rows=converted
assert all(any(r['asset']==a and r['split']=='train' for r in rows) for a in assets),'An object lost all readable training episodes'
values={k:[] for k in ['qpos','action']}
for row in rows:
 if row['split']!='train':continue
 with h5py.File(row['cache']) as f:
  for k in values:values[k].append(f[k][:])
stats={}
for k,v in values.items():
 x=np.concatenate(v).astype(np.float64);stats[k+'_mean']=x.mean(0).tolist();stats[k+'_std']=np.maximum(x.std(0),.01).tolist();stats[k+'_q01']=np.quantile(x,.01,axis=0).tolist();stats[k+'_q99']=np.quantile(x,.99,axis=0).tolist()
stats.update(joint_names=cfg['policy_action_joint_names'],assets=assets,camera_names=['head_front','wrist_left','wrist_right'],image_size=224,image_processing='aspect-preserving resize with centered black padding; RGB',control_hz=20)
train_keys={(r['asset'],tuple(r['position_key'])) for r in rows if r['split']=='train'};val_keys={(r['asset'],tuple(r['position_key'])) for r in rows if r['split']=='validation'};assert not train_keys&val_keys
scope=dict(episodes=len(rows),train_episodes=sum(r['split']=='train' for r in rows),validation_episodes=sum(r['split']=='validation' for r in rows),train_frames=sum(r['frames'] for r in rows if r['split']=='train'),validation_assets=sorted(set(r['asset'] for r in rows if r['split']=='validation')),no_holdout_assets=sorted(set(assets)-set(r['asset'] for r in rows if r['split']=='validation')),per_asset={a:dict(collections.Counter(r['split'] for r in rows if r['asset']==a)) for a in assets},rejected=rejected,position_ranges={a:{k:[min(float(r.get(k,0)) for r in rows if r['asset']==a),max(float(r.get(k,0)) for r in rows if r['asset']==a)] for k in ['dx_m','dy_m','dyaw_deg']} for a in assets},scope='Existing position/orientation distribution only; no new scene/background or unseen-object generalization claim')
for name,data in [('manifest.json',rows),('stats.json',stats),('scope.json',scope)]: (D/name).write_text(json.dumps(data,indent=2))
print('EXPANDED_READY',json.dumps(scope),flush=True)
