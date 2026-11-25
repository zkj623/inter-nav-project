from pathlib import Path
import json,hashlib,subprocess,sys
import h5py,numpy as np,cv2
from g2_force_features import causal_force
W=Path(__file__).resolve().parent;O=W/'datasets/pilot4_dagger_force';O.mkdir(exist_ok=True)
summary=json.loads((W/'dagger_collection_v1/summary.json').read_text());accepted=[r for r in summary['results'] if r['accepted']]
assert accepted,'At least one independently verified recovery required before retraining'
rows=json.loads((W/'datasets/pilot4_force_causal/manifest.json').read_text());stats=json.loads((W/'datasets/pilot4_force_causal/stats.json').read_text());stats['progress_terminal_limits']=json.loads((W/'progress_terminal_limits.json').read_text());stats['corrective_expert_episodes']=len(accepted)
records=[{**r,'source_role':'expert_recovery_suffix'} for r in accepted]
retention_exclusions=[]
retention_candidates=[]
for source_run in ['act_progress_v1_8000','act_force_v1_8000']:
 evaluations=json.loads((W/'evaluations'/source_run/'summary.json').read_text())
 retention_candidates.extend((source_run,result) for result in evaluations['results'] if result['success'])
for source_run,result in retention_candidates:
 if not result['success']:continue
 T=Path(result['trial']);v=json.loads((T/'verification.json').read_text());assert v['verified_success']
 if (T/'force_input_audit.json').exists():assert json.loads((T/'force_input_audit.json').read_text())['passed']
 with (O/(result['asset']+'_retention_camera_audit.log')).open('w') as log:
  code=subprocess.run([sys.executable,str(W.parent/'g2-act-threecam-20261124/verify_threecam.py'),str(T)],stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
 if code:
  retention_exclusions.append(dict(asset=result['asset'],reason='Three-camera retention-data audit failed'));continue
 base=next(r for r in rows if r['asset']==result['asset'])
 records.append({**base,'trial':str(T),'source_role':'verified_policy_distillation','retention_source_run':source_run,'teacher_start_frame':0,'source_sha256':v['episode_sha256'],'force_features_source':'Reconstructed only from past physical contact samples; the source policy may not have consumed these features'})
stats['verified_policy_retention_episodes']=len(records)-len(accepted)
(O/'retention_exclusions.json').write_text(json.dumps(retention_exclusions,indent=2))
for row in records:
 T=Path(row['trial']);samples=json.loads((T/'samples.json').read_text());cache=O/(row['asset']+'_'+row['source_role']+'_'+row.get('retention_source_run','recovery')+'.hdf5');start=row['teacher_start_frame'];assert not cache.exists()
 with h5py.File(T/'episode.hdf5') as raw,h5py.File(cache,'w') as f:
  if row['source_role']=='expert_recovery_suffix':
   labels=raw['behavior_source'][:];assert np.all(labels[:start]==0) and np.all(labels[start:]==1)
  total=len(raw['action']);n=total-start;times=raw['timestamps'][start:];values=[causal_force(samples,float(t)) for t in times]
  f['qpos']=raw['observations/qpos'][start:];f['action']=raw['action'][start:];f.create_dataset('phase',data=raw['phase'].asstr()[start:],dtype=h5py.string_dtype('utf-8'))
  f['finger_forces']=np.stack([v[0] for v in values]).astype('f4');f['progress']=np.arange(start,total)/200.;f.attrs['source_frame_start']=start;f.attrs['source_role']=row['source_role'];f.attrs['source_episode_sha256']=row['source_sha256']
  for name in stats['camera_names']:
   ds=f.create_dataset(name,shape=(n,224,224,3),dtype='u1',chunks=(1,224,224,3),compression='lzf')
   for j in range(n):
    im=raw['observations/images/'+name][start+j];h,w=im.shape[:2];scale=224/max(h,w);nw,nh=round(w*scale),round(h*scale);small=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA);tile=np.zeros((224,224,3),np.uint8);y=(224-nh)//2;x=(224-nw)//2;tile[y:y+nh,x:x+nw]=small;ds[j]=tile
  assert len(f['action'])==n and n>=60
 row={**row,'cache':str(cache),'split':'train','group_id':row['asset']+'_'+row['source_role']+'_'+row.get('retention_source_run','recovery')};rows.append(row)
(O/'manifest.json').write_text(json.dumps(rows,indent=2));(O/'stats.json').write_text(json.dumps(stats,indent=2));print('DAGGER_SUFFIXES_CACHED',len(accepted))
