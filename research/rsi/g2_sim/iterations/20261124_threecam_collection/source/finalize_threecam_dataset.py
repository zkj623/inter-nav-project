from pathlib import Path
import json,hashlib,os,shutil,h5py,numpy as np
from threecam_drive_profile import profile_for_trial
R=Path(__file__).resolve().parent;cfg=json.loads((R/'ground_lift_training_config.json').read_text());assets=list(cfg['validated_simulation_baselines']);rows=[];summary={};profiles={};D=R/'dataset';D.mkdir(exist_ok=True);totals={key:dict(n=0,sum=np.zeros(16),square=np.zeros(16)) for key in ['qpos','action']}
assert (R/'threecam_pilot_verified.json').exists(),'Pilot replay gate missing'
for a in assets:
 state=json.loads((R/'collection_state'/(a+'.json')).read_text());accepted=state['accepted'];assert len(accepted)==20,(a,len(accepted));rng=np.random.default_rng(20261124+assets.index(a));order=rng.permutation(20);split={int(j):'train' if i<16 else 'validation' if i<18 else 'test' for i,j in enumerate(order)}
 positions=[];O=D/'episodes'/a;O.mkdir(parents=True,exist_ok=True)
 for i,item in enumerate(accepted):
  T=Path(item['trial']);v=json.loads((T/'acceptance.json').read_text());assert v['accepted'];source=T/'episode.hdf5';target=O/f'episode_{i:04d}.hdf5'
  if not target.exists():os.link(source,target)
  h=hashlib.sha256()
  with target.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  assert h.hexdigest()==v['episode_sha256'];xy=np.array([item['dx_m'],item['dy_m']]);assert all(np.linalg.norm(xy-x)>=.001 for x in positions);positions.append(xy)
  profile=profile_for_trial(T)
  if a in profiles:assert profiles[a]['profile_id']==profile['profile_id'],'Mixed low-level drive profiles within one asset'
  else:profiles[a]=profile
  evidence=D/'metadata'/a/f'episode_{i:04d}';evidence.mkdir(parents=True,exist_ok=True)
  for name in ['metadata.json','acceptance.json','verification.json','result.json','execution_profile.json']:shutil.copy2(T/name,evidence/name)
  meta=json.loads((T/'metadata.json').read_text());result=json.loads((T/'result.json').read_text())
  row=dict(asset=a,episode_id=i,split=split[i],placement_group=item['group_id'],seed=item['seed'],dx_m=item['dx_m'],dy_m=item['dy_m'],dyaw_deg=0.,episode=str(target.relative_to(D)),sha256=h.hexdigest(),frames=v['frames'],trial=str(T),metadata=str((evidence/'metadata.json').relative_to(D)),acceptance=str((evidence/'acceptance.json').relative_to(D)),review_video=str(T/'review.mp4'),drive_profile_id=profile['profile_id'],execution_profile=str((evidence/'execution_profile.json').relative_to(D)),source_sha256=result['source_sha256'],threecam_recorder_sha256=meta['threecam_recorder_sha256'],baseline_commit=meta['source_commit']);rows.append(row)
  if split[i]=='train':
   with h5py.File(target,'r') as f:
    for key,path in [('qpos','observations/qpos'),('action','action')]:
     values=f[path][:].astype(np.float64);t=totals[key];t['n']+=len(values);t['sum']+=values.sum(0);t['square']+=(values**2).sum(0)
 summary[a]=dict(accepted=20,candidate_attempts=len(state['attempts']),completed_simulations=sum('trial' in x for x in state['attempts']),dx_range_m=[float(np.min(positions,axis=0)[0]),float(np.max(positions,axis=0)[0])],dy_range_m=[float(np.min(positions,axis=0)[1]),float(np.max(positions,axis=0)[1])],rejections={str(reason):sum(str(x.get('failure'))==str(reason) for x in state['attempts']) for reason in set(str(x.get('failure')) for x in state['attempts'] if not x['accepted'])})
assert len(rows)==240 and len({r['placement_group'] for r in rows})==240
stats={}
for key,t in totals.items():mean=t['sum']/t['n'];std=np.sqrt(np.maximum(t['square']/t['n']-mean**2,0));stats[key+'_mean']=mean.tolist();stats[key+'_std']=np.maximum(std,.01).tolist()
stats.update(joint_names=cfg['policy_action_joint_names'],camera_names=['head_front','wrist_left','wrist_right'],control_hz=20,action_units='absolute joint position radians',normalization_split='train only',image_processing='RGB float[0,1]; pad head400x640 to528x640 with64 black rows above/below; wrist images unchanged; no image rotation')
(D/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(D/'normalization.json').write_text(json.dumps(stats,indent=2));(D/'dataset_summary.json').write_text(json.dumps(dict(episodes=240,splits={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},assets=summary,old240_included=False,scope='current common low home, fixed body/base,1kg rigid objects; translated XY positions, fixed per-object yaw; head plus official wrist cameras'),indent=2));print('FROZEN240_THREECAM_EPISODES')
(D/'policy_execution_profiles.json').write_text(json.dumps(profiles,indent=2))
for name in ['threecam_pilot_verified.json','collection_source_manifest.json']:shutil.copy2(R/name,D/name)
shutil.copytree(R/'collection_state',D/'collection_state',dirs_exist_ok=True)
source_dir=D/'source_versions';source_dir.mkdir(exist_ok=True)
for sha in {r['source_sha256'] for r in rows}:
 src=R/'source_versions'/(sha+'.py');assert hashlib.sha256(src.read_bytes()).hexdigest()==sha;shutil.copy2(src,source_dir/src.name)
for name in ['threecam_recorder.py','collection_recorder.py','official_wrist_config.json','act_threecam_dataset.py','verify_threecam.py','verify_realpose.py','threecam_drive_profile.py']:shutil.copy2(R/name,source_dir/name)
