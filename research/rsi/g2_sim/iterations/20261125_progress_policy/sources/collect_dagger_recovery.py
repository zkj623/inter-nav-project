from pathlib import Path
import argparse,subprocess,json,sys,hashlib
import h5py,numpy as np
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19035);a=p.parse_args()
checkpoint=W/'runs/act_force_v1/step_008000.pt';run='dagger_force_v1';O=W/'dagger_collection_v1';O.mkdir(exist_ok=True)
source=json.loads((W/'evaluations/act_force_v1_8000/summary.json').read_text());assert source['total']==4
prefixes=dict(box_white=55.,bin_lidded=38.,bin_pedal=25.,basket_rectangular=35.)
rows=json.loads((W/'datasets/pilot4_one_each/manifest.json').read_text());results=[]
def invoke(command,name,timeout):
 with (O/(name+'.log')).open('w') as log:return subprocess.run([sys.executable,*map(str,command)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
for failed in source['results']:
 if any(r['accepted'] for r in results):break  # Minimal verified corrective pilot before retraining.
 if failed['success']:continue
 asset=failed['asset'];row=next(r for r in rows if r['asset']==asset);T=R/'dagger_trials'/run/row['plan_dir']/asset/'candidate0_repeat1'
 previous=Path(failed['trial']);episode=previous/'episode.hdf5'
 if not episode.exists():episode=previous/'episode.partial.hdf5'
 with h5py.File(episode) as f:elapsed=len(f['action'])/20.
 prefix=min(prefixes[asset],max(5.,elapsed-3.))
 if not (T/'result.json').exists():invoke([R/f'dagger_force_{asset}_trial.py','--asset',asset,'--plan-dir',row['plan_dir'],'--gpu','1','--run',run,'--checkpoint',checkpoint,'--policy-url',f'http://127.0.0.1:{a.port}','--duration',prefix,'--execute','2'],asset+'_simulation',2400)
 accepted=False;reasons=[]
 if not (T/'result.json').exists():reasons=['NO_RESULT']
 else:
  invoke([R/'verify_realpose.py','--trial-dir',T],asset+'_physics_audit',1200)
  invoke([R/'verify_threecam.py',T],asset+'_camera_audit',900)
  frc=invoke([W/'audit_force_episode.py','--trial',T],asset+'_force_audit',120)
  physical=json.loads((T/'verification.json').read_text());camera=json.loads((T/'acceptance.json').read_text());result=json.loads((T/'result.json').read_text())
  reasons=camera['rejections'][:]
  if frc:reasons.append('FORCE_INPUT_AUDIT')
  if result.get('collection_role')!='DAGGER_EXPERT_RECOVERY_NOT_POLICY_EVALUATION':reasons.append('COLLECTION_ROLE')
  start=result.get('teacher_start_frame');episode=T/'episode.hdf5' if (T/'episode.hdf5').exists() else T/'episode.partial.hdf5'
  with h5py.File(episode) as f:
   labels=f['behavior_source'][:];valid=isinstance(start,int) and start>0 and len(labels)-start>=60 and np.all(labels[:start]==0) and np.all(labels[start:]==1)
  if not valid:reasons.append('EXPERT_SUFFIX_LABELS')
  accepted=not reasons and physical['verified_success'] and camera['accepted']
 record={**row,'trial':str(T),'accepted':bool(accepted),'failure':reasons,'prefix_duration_s':prefix,'collection_role':'expert_recovery_suffix','source_policy_checkpoint':str(checkpoint)}
 if accepted:record.update(teacher_start_frame=start,source_sha256=physical['episode_sha256'])
 results.append(record);(O/'summary.json').write_text(json.dumps(dict(accepted=sum(r['accepted'] for r in results),total=len(results),results=results),indent=2));print(json.dumps(dict(asset=asset,accepted=accepted,failure=reasons)),flush=True)
print('DAGGER_COLLECTION_COMPLETE_NOT_POLICY_EVALUATION',flush=True)
