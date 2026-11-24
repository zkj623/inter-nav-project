from pathlib import Path
import json,sys,subprocess,time,concurrent.futures,hashlib
import numpy as np,h5py
R=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--repeat',type=int,default=1);args=parser.parse_args()
assets=['case_handheld','basket_woven','bin_pedal']
def verify(a):
 D=R/'realpose_trials/realpose_plans'/a/f'candidate0_repeat{args.repeat}';deadline=time.time()+1800
 while not (D/'result.json').exists():
  if time.time()>deadline:raise RuntimeError('Pilot result timeout '+a)
  time.sleep(3)
 p=subprocess.run([sys.executable,str(R/'verify_realpose.py'),'--trial-dir',str(D)],cwd=R,capture_output=True,text=True);(D/'independent_audit.log').write_text(p.stdout+p.stderr)
 result=json.loads((D/'verification.json').read_text());report=dict(asset=a,physics_independently_verified=result.get('verified_success'),physics_failure=result.get('failure'),physics_rejections=result.get('rejections'),head_stream_verified=False)
 path=D/'episode.hdf5'
 if not path.exists():path=D/'episode.partial.hdf5'
 with h5py.File(path,'r') as f,h5py.File(D/'head_pilot.hdf5','r') as h:
  t=h['timestamps'][:];ct=h['camera_timestamps'][:];n=len(t);errors={key:float(np.max(abs(h[key][:]-f[key][:]))) for key in ['observations/qpos','observations/qvel','action','timestamps']};images=h['observations/images/head_front'];report.update(head_frames=n,head_image_shape=list(images.shape),max_camera_time_offset_s=float(np.max(abs(t-ct))),primary_stream_alignment_errors=errors,twenty_hz=bool(np.allclose(np.diff(t),.05,atol=1e-6)),images_nonempty=bool(all(images[i].std()>2 for i in [0,n//2,n-1])),first_phase=h['phase'].asstr()[0],last_phase=h['phase'].asstr()[-1])
  report['head_stream_verified']=bool(n>60 and images.shape==(n,400,640,3) and max(errors.values())<1e-6 and report['max_camera_time_offset_s']<=1/120+.0001 and report['twenty_hz'] and report['images_nonempty'] and report['last_phase']=='hold')
  if result.get('verified_success'):report.update(min_hold_lift_m=result['min_hold_lift_m'],max_hold_tilt_deg=result['max_hold_tilt_deg'])
 (D/'head_pilot_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return report
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(verify,assets))
(R/'realpose_pilot_validation.json').write_text(json.dumps(rows,indent=2));print('PILOTS_AUDITED',sum(x['physics_independently_verified'] and x['head_stream_verified'] for x in rows),len(rows),flush=True)
