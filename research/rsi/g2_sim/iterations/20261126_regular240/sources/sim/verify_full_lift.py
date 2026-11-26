from pathlib import Path
from full_lift_rules import allowed_base_contact, hold_metrics
import argparse,json,hashlib,os
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,CookedGeometry,ConvexPart
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;pa=argparse.ArgumentParser();pa.add_argument('--trial-dir',required=True);args=pa.parse_args();D=Path(args.trial_dir);d=json.loads((D/'result.json').read_text());row=dict(audit_version='full_lift_v1',auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trial=str(D),asset=d['asset'],harness_success=d['success'],verified_success=False,failure=d.get('failure'));ss=json.loads((D/'samples.json').read_text());cfg=json.loads((R/'ground_lift_training_config.json').read_text());source=R/'source_versions'/(d['source_sha256']+'.py');row['source_matches']=source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==d['source_sha256']
if d['success']:
 reasons=[];hold=[s for s in ss if s['phase']=='hold'];work=[s for s in ss if s['phase'] not in ('default_settle','prepare_home','initial_settle')];a=d['asset'];pr=R/'object_assets'/(a+'_cache_provenance.json');path=Path(json.loads(pr.read_text())['path']) if pr.exists() else R/'object_assets'/(a+'_cooked.json')
 if a in ['stool_rectangular','stool_wood_four_leg','table_side','basket_rectangular']:path=R/'strategy15_surface_assets'/(a+'_cooked.json')
 c=json.loads(path.read_text());obj=CookedGeometry([ConvexPart(x['vertices_object_m'],str(i)) for i,x in enumerate(c['hulls'])]);mesh=np.load(R/'object_assets'/(a+'.npz'))['vertices'];hits=[];selfhits=[];floor=[];robotfloor=[];limits=[];maximp={}
 for kind in ['self','nonhand']:
  events=json.loads((D/(kind+'_contacts.json')).read_text());maximp[kind]=max((e['impulse'] for e in events),default=0.)
 root=max(np.linalg.norm(np.array(s['base_position'])-[0,0,.04]) for s in ss);frozen=max(s['frozen_joint_max_error_rad'] for s in work);margin=min(np.degrees(min(s['joint_positions'][n]-j['lower'],j['upper']-s['joint_positions'][n])) for s in work for n,j in k.by_name.items() if 'body_joint' in n)
 names=[n for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic']]
 for s in ss:
  q=s['joint_positions'];base=np.eye(4);base[:3,3]=s['base_position'];fk=k.fk(q,base);rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix();pos=np.array(s['object_position']);sh=intersections(q)
  if sh:selfhits.append(dict(t=s['t'],pairs=sh))
  for n in names:
   if not k.by_name[n]['lower']-.005<=q.get(n,0)<=k.by_name[n]['upper']+.005:limits.append(dict(t=s['t'],joint=n))
  for n,g in robot.items():
   if n!=k.root and not n.startswith('chassis_'):
    ft=fk[n];robotfloor.append(float(ft[2,3]+ft[2,:3]@g.bounds.mean(0)-abs(ft[2,:3])@((g.bounds[1]-g.bounds[0])/2)))
   if n.startswith('gripper_'):continue
   t=fk[n].copy();t[:3,3]=rot.T@(t[:3,3]-pos);t[:3,:3]=rot.T@t[:3,:3];ctr=t[:3,:3]@g.bounds.mean(0)+t[:3,3];ext=abs(t[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
   if np.any(ctr+ext<obj.bounds[0]) or np.any(ctr-ext>obj.bounds[1]):continue
   gap=obj.compound_gap(g,t)
   if gap<-.001:hits.append(dict(t=s['t'],link=n,gap_m=gap))
  if s['phase']=='hold':floor.append(float((mesh@rot.T+pos)[:,2].min()))
 row.update(root_drift_m=float(root),frozen_error_rad=float(frozen),min_body_margin_deg=float(margin),self_intersections=selfhits,nonhand_geometry_hits=hits,joint_limit_violations=limits,max_impulses_ns=maximp,min_hold_floor_clearance_m=min(floor),min_hold_lift_m=d['min_hold_lift_m'],max_hold_tilt_deg=d['max_hold_tilt_deg'],hold_relative_drift_m=d['hold_relative_drift_m'],four_contact_fraction=d['hold_four_finger_contact_fraction'],max_contact_surface_error_m=d['max_contact_surface_error_m'])
 if not row['source_matches']:reasons.append('SOURCE_HASH_CHANGED')
 if d['plan']['q_start']!=cfg['common_initial_joint_positions_rad']:reasons.append('COMMON_HOME_CHANGED')
 if selfhits:reasons.append('VISIBLE_SELF_INTERSECTION')
 if any(h['link']!='base_link' for h in hits):reasons.append('NONHAND_GEOMETRY')
 if min(robotfloor)<=.001:reasons.append('ROBOT_FLOOR_SUPPORT')
 row['min_robot_nonwheel_aabb_floor_gap_m']=min(robotfloor)
 if limits:reasons.append('JOINT_LIMIT')
 if root>=.001 or frozen>=.005 or margin<cfg.get('minimum_body_margin_acceptance_deg',9.9):reasons.append('FROZEN_ROOT_MARGIN')
 nonhand_events=json.loads((D/'nonhand_contacts.json').read_text())
 if maximp['self']>.001 or any(e['impulse']>.001 and not allowed_base_contact(e) for e in nonhand_events):reasons.append('CONTACT_IMPULSE')
 if not(hold_metrics(hold)['accepted'] and min(floor)>.005 and d['max_contact_surface_error_m']<=.0025):reasons.append('HOLD_ACCEPTANCE')
 dataset=D/'episode.partial.hdf5'
 if not dataset.exists():dataset=D/'episode.hdf5'
 with h5py.File(dataset,'r') as f:
  t=f['timestamps'][:];ct=f['camera_timestamps'][:];q=f['observations/qpos'][:];act=f['action'][:];images=f['observations/images/workspace'];ph=f['phase'].asstr()[:];T=len(t);order=json.loads(f.attrs['joint_names']);finite=bool(np.isfinite(q).all() and np.isfinite(act).all());timing=bool(T>60 and np.allclose(np.diff(t),.05,atol=1e-6) and np.max(abs(t-ct))<=1/120+.0001);shapes=bool(q.shape==act.shape==(T,16) and images.shape==(T,480,640,3) and images.dtype==np.uint8 and order==cfg['policy_action_joint_names']);nonempty=all(images[i].std()>1 for i in [0,T//2,T-1]);row['data']=dict(frames=T,finite=finite,timing=timing,shapes=shapes,nonempty=bool(nonempty),duration_s=float(t[-1]-t[0]+.05),max_camera_offset_s=float(np.max(abs(t-ct))),first_phase=str(ph[0]),last_phase=str(ph[-1]))
  continued=act[ph=='hold']; info=d.get('policy_continuation') or {}
  calls=d.get('policy_evaluation',{}).get('inference_calls',[])
  continuation_ok=bool(len(continued)>=200 and info.get('inference_calls_after_trigger',0)>0 and any(x['frame']>=info.get('frame',1e30) for x in calls))
  row['continued_feedback_verified']=continuation_ok;row['full_metrics']=hold_metrics(hold)
  row['hold_command_max_change_rad']=float(np.max(abs(continued-continued[0]))) if len(continued) else None
  if not continuation_ok:reasons.append('POLICY_CONTINUATION_INVALID')
  velocities=abs(f['observations/qvel'][:]);row['data']['max_actual_velocity_fraction']=float(np.max(velocities/np.array([k.by_name[n]['velocity'] for n in order])))
  if row['data']['max_actual_velocity_fraction']>1.0001:reasons.append('ACTUAL_VELOCITY_LIMIT')
  # Cross-check actual pre-action observations against audit samples at the same simulation time.
  sam={round(s.get('physics_time_s',s['t']),6):s for s in ss};errors=[];actionerrors=[]
  for i,ti in enumerate(t):
   x=sam.get(round(float(ti),6))
   if x:errors.append(max(abs(q[i,j]-x['joint_positions'][n]) for j,n in enumerate(order)))
   nextsample=next((sam[round(float(ti+dt),6)] for dt in [1/120,7/120] if round(float(ti+dt),6) in sam and dt<.05),None)
   if nextsample:actionerrors.append(max(abs(act[i,j]-nextsample['joint_targets'][n]) for j,n in enumerate(order)))
  row['data']['same_time_qpos_max_error_rad']=max(errors,default=None);row['data']['same_step_action_max_error_rad']=max(actionerrors,default=None)
  if not(finite and timing and shapes and nonempty):reasons.append('DATA_INTEGRITY')
  if errors and max(errors)>1e-5:reasons.append('OBSERVATION_ALIGNMENT')
  if actionerrors and max(actionerrors)>1e-5:reasons.append('ACTION_ALIGNMENT')
 row['verified_success']=not reasons;row['rejections']=reasons
 if row['verified_success']:
  if dataset.name=='episode.partial.hdf5':dataset.rename(D/'episode.hdf5')
  with h5py.File(D/'episode.hdf5','r+') as f:
   if not f.attrs.get('success',False):f.attrs['success']=True
   if not f.attrs.get('independently_verified',False):f.attrs['independently_verified']=True
  row['episode_path']=str(D/'episode.hdf5');row['episode_sha256']=hashlib.sha256((D/'episode.hdf5').read_bytes()).hexdigest()
(D/'stable_verification.tmp.json').write_text(json.dumps(row,indent=2));(D/'stable_verification.tmp.json').replace(D/'stable_verification.json');print(json.dumps({n:v for n,v in row.items() if n not in ['self_intersections','nonhand_geometry_hits','joint_limit_violations']}),flush=True)
raise SystemExit(0 if row['verified_success'] else 2)



