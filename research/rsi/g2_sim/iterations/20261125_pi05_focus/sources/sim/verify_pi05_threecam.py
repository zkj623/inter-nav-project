from pathlib import Path
import sys,json,hashlib,h5py,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;T=Path(sys.argv[1]);physical=json.loads((T/'stable_verification.json').read_text());res=json.loads((T/'result.json').read_text());out=dict(asset=res['asset'],accepted=False,physical_verified=physical['verified_success'],rejections=[],auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if not physical['verified_success']:out['rejections'].append('PHYSICAL_VERIFICATION')
else:
 fpath=T/'episode.hdf5';cfg=json.loads((R/'ground_lift_training_config.json').read_text());k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');ss=json.loads((T/'samples.json').read_text());mirror=0.;height=-99.;span=0.
 for s in ss:
  q=s['joint_positions'];mirror=max(mirror,max(abs(np.degrees(q[f'idx{61+i}_arm_r_joint{i+1}']-sg*q[f'idx{21+i}_arm_l_joint{i+1}'])) for i,sg in enumerate([-1,1,-1,1,-1,1,-1])))
  if s['phase'] in ['close','contact_refine','lift','contact_settle','hold']:
   fk=k.fk(q,np.array(res['plan']['base_pose']));height=max(height,max(fk[f'arm_{s}_link4'][2,3]-fk[f'arm_{s}_link1'][2,3] for s in ['l','r']));span=max(span,abs(fk['arm_l_link4'][1,3]-fk['arm_r_link4'][1,3]))
 out.update(max_actual_mirror_joint_error_deg=mirror,max_elbow_above_shoulder_m=height,max_elbow_span_m=span)
 out['posture_review_needed']=bool(mirror>15 or height>.1) # diagnostic, not a stable-lift failure
 if res.get('posture_anchor_sha256') and res['posture_anchor_sha256']!=hashlib.sha256((R/'posture_anchor.py').read_bytes()).hexdigest():out['rejections'].append('POSTURE_HELPER_HASH')
 meta=json.loads((T/'metadata.json').read_text());out['threecam_recorder_source_matches']=meta['threecam_recorder_sha256'] in {hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['threecam_recorder.py','threecam_recorder_v2.py','threecam_recorder_v3.py']}
 if not out['threecam_recorder_source_matches']:out['rejections'].append('CAMERA_RECORDER_HASH')
 if meta['recorder_sha256']!=hashlib.sha256((R/'collection_recorder.py').read_bytes()).hexdigest():out['rejections'].append('BASE_RECORDER_HASH')
 if meta['official_wrist_calibration']!=json.loads((R/'official_wrist_config.json').read_text()):out['rejections'].append('WRIST_CALIBRATION_CHANGED')
 with h5py.File(fpath,'r') as f:
  t=f['timestamps'][:];N=len(t);out['frames']=N;out['cameras']={};assert json.loads(f.attrs['camera_names'])==['head_front','wrist_left','wrist_right']
  for name,shape in [('head_front',(400,640,3)),('wrist_left',(528,640,3)),('wrist_right',(528,640,3))]:
   images=f['observations/images/'+name];ct=f['image_timestamps/'+name][:];indices=np.unique(np.linspace(0,N-1,12,dtype=int));stds=[float(images[i].std()) for i in indices];differences=[float(np.abs(images[i].astype(float)-images[j].astype(float)).mean()) for i,j in zip(indices,indices[1:])]
   every_frame_std=[float(images[i,::8,::8,:].std()) for i in range(N)]
   good=images.shape==(N,*shape) and images.dtype==np.uint8 and ct.shape==t.shape and np.max(abs(ct-t))<=1/120+.0001 and min(stds)>1 and min(every_frame_std)>1 and max(differences)>.1
   out['cameras'][name]=dict(shape=list(images.shape),max_timestamp_error_s=float(np.max(abs(ct-t))),min_image_std=min(stds),min_all_frame_subsample_std=min(every_frame_std),all_frames_checked=N,max_between_sample_change=max(differences),verified=bool(good))
   if not good:out['rejections'].append('CAMERA_'+name)
  action=f['action'][:];order=cfg['policy_action_joint_names'];errors=[]
  for s in ss:
   ts=s.get('physics_time_s');j=np.searchsorted(t,ts-.5/120,side='right')-1 if ts is not None else -1
   if 0<=j<N and 0<ts-t[j]<=.05+1e-6:errors.append(float(max(abs(action[j,n]-s['joint_targets'][key]) for n,key in enumerate(order))))
  out['actual_command_alignment_witness_count']=len(errors);out['actual_command_alignment_max_error_rad']=max(errors,default=None)
  if len(errors)<N//4 or max(errors,default=1)>1e-5:out['rejections'].append('ACTION_ALIGNMENT_WITNESS')
 out['episode_path']=str(fpath);out['episode_sha256']=physical['episode_sha256'];out['placement']=res['plan']['collection_placement'];out['accepted']=not out['rejections']
(T/'stable_threecam_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);raise SystemExit(0 if out['accepted'] else 2)
