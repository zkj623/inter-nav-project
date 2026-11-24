from pathlib import Path
import time,json,subprocess,sys,h5py,numpy as np,hashlib
R=Path(__file__).resolve().parent;deadline=time.monotonic()+7200
while True:
 rows=[]
 for i,repeat in [(1,2),(2,2),(4,1)]:
  T=R/f'realpose_trials/placements/p{i:04d}/box_white/candidate0_repeat{repeat}';D=R/f'threecam_replay_trials/placements/p{i:04d}/box_white/candidate0_repeat{repeat}'
  if not all((x/'acceptance.json').exists() for x in [T,D]):continue
  if not all(json.loads((x/'acceptance.json').read_text())['accepted'] for x in [T,D]):continue
  with h5py.File(T/'episode.hdf5','r') as x,h5py.File(D/'episode.hdf5','r') as y:
   if x['action'].shape!=y['action'].shape or not np.array_equal(x['action'][:],y['action'][:]):continue
   error=float(np.max(abs(x['observations/qpos'][:]-y['observations/qpos'][:])))
  if error>.005:continue
  rows.append(dict(index=i,trial=str(T),replay=str(D),actual_qpos_max_difference_rad=error,recorded_actions_equal=True,physical_and_threecam_verified=True))
 if len(rows)>=2:
  (R/'threecam_pilot_verified.json').write_text(json.dumps(rows,indent=2));break
 if time.monotonic()>deadline:raise RuntimeError('PILOT_GATE_NOT_PASSED_WITHIN2H')
 time.sleep(10)
paths=[*R.glob('realpose_*_trial.py'),R/'threecam_recorder.py',R/'collection_recorder.py',R/'plan_threecam.py',R/'verify_threecam.py',R/'verify_realpose.py',R/'official_wrist_config.json'];(R/'collection_source_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print('TWO_PILOTS_VERIFIED_STARTING12x20',flush=True)
subprocess.run([sys.executable,str(R/'run_threecam_collection.py')],cwd=R,check=True)
