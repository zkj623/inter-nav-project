from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
s=(R/'verify_liftpair.py').read_text().replace("R.glob('liftpair*_trials/**/result.json')","R.glob('restpose_v2_trials/**/result.json')").replace("'liftpair_verified_results.json'","'restpose_verified_results.json'")
s=s.replace("if d['success']:","if d['success'] and d['test_type']!='empty_pose_hold':")
s=s.replace("for s in samples for n,j in k.by_name.items() if 'body_joint' in n","for s in samples if s['phase'] not in ('default_settle','prepare_home','initial_settle') for n,j in k.by_name.items() if 'body_joint' in n")
exec(compile(s,'restpose_lift_review','exec'))
rows=json.loads((R/'restpose_verified_results.json').read_text());cfg=json.loads((R/'restpose_candidate_config.json').read_text());rest=cfg['standby_joint_positions_rad'];home=cfg['work_joint_positions_rad']
for r in rows:
 f=R/r['result'];d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());p=d.get('plan',{});pose_only=d['test_type']=='empty_pose_hold'
 r['workflow']='standby_hold_and_preparation' if pose_only else 'standby_preparation_and_grasp'
 r['custom_standby_targets_match']=p.get('q_default')==rest;r['work_pose_unchanged']=p.get('q_start')==home
 r['standby_max_tracking_error_rad']=d.get('standby_max_tracking_error_rad');r['prepared_max_tracking_error_rad']=d.get('prepared_max_tracking_error_rad')
 pts=d.get('standby_link_positions_m',{});mirror=[];arms_down=True
 for side in ['l','r']:
  sh=np.array(pts.get(f'arm_{side}_link2',[0,0,0]));el=np.array(pts.get(f'arm_{side}_link4',[0,0,0]));wr=np.array(pts.get(f'arm_{side}_end_link',[0,0,0]))
  arms_down=arms_down and wr[2]+.12<el[2] and el[2]+.12<sh[2]
 for name in ['link2','link4','end_link']:
  mirror.append(float(np.linalg.norm(np.array(pts.get('arm_r_'+name,[999,999,999]))-np.array(pts.get('arm_l_'+name,[0,0,0]))*[1,-1,1])))
 r['arms_down_verified']=bool(arms_down);r['standby_arm_mirror_error_m']=max(mirror);r['standby_link_positions_m']=pts
 fk=k.fk(d.get('standby_actual_q',{}));zero=k.fk({});reflection=np.diag([1.,-1.,1.]);local_reflection=zero['arm_l_end_link'][:3,:3].T@reflection@zero['arm_r_end_link'][:3,:3]
 rr=fk['arm_r_end_link'][:3,:3].T@reflection@fk['arm_l_end_link'][:3,:3]@local_reflection
 r['standby_end_rotation_mirror_error_rad']=float(np.arccos(np.clip((np.trace(rr)-1)/2,-1,1)))
 before=[x for x in ss if x['phase']=='prepare_home'];r['preparation_max_tracking_error_rad']=max((x['max_tracking_error_rad'] for x in before),default=999.)
 names=[n for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic']]
 r['original_limit_violations']=[(x['t'],n) for x in ss for n in names if x['joint_positions'].get(n,0)<k.by_name[n]['lower']-.005 or x['joint_positions'].get(n,0)>k.by_name[n]['upper']+.005][:10]
 r['preparation_fd_peak_velocity_fraction']=max((abs(y['joint_positions'].get(n,0)-x['joint_positions'].get(n,0))/((y['t']-x['t'])*k.by_name[n]['velocity']) for x,y in zip(before,before[1:]) for n in names),default=999.)
 sc=json.loads(f.with_name('self_contacts.json').read_text());r['max_self_contact_impulse_ns']=max((x['impulse'] for x in sc),default=0.)
 peaks=[]
 for seg in d.get('preparation_motion',[]):
  dv={n:abs(seg['end'].get(n,0)-seg['start'].get(n,0)) for n in rest};t=seg['duration_s'];peaks.append(dict(duration_s=t,max_velocity_fraction=max(2.1875*v/t/k.by_name[n]['velocity'] for n,v in dv.items()),max_acceleration_rad_s2=max(dv.values())*7.514/t**2,max_jerk_rad_s3=max(dv.values())*52.5/t**3))
 r['preparation_profile_peaks']=peaks
 ready=bool(r['custom_standby_targets_match'] and r['work_pose_unchanged'] and arms_down and r['standby_arm_mirror_error_m']<.005 and r['standby_end_rotation_mirror_error_rad']<np.radians(.5) and r['standby_max_tracking_error_rad'] is not None and r['standby_max_tracking_error_rad']<.03 and r['prepared_max_tracking_error_rad'] is not None and r['prepared_max_tracking_error_rad']<.03 and r['preparation_max_tracking_error_rad']<.25 and r['preparation_fd_peak_velocity_fraction']<=1. and not r['original_limit_violations'] and r['max_self_contact_impulse_ns']<=.001 and peaks and all(x['max_velocity_fraction']<=.400001 and x['max_acceleration_rad_s2']<=.500001 and x['max_jerk_rad_s3']<=2.000001 for x in peaks))
 r['standby_and_transition_verified']=ready
 ready=bool(ready and r.get('root_displacement_m',999)<.001 and r.get('frozen_error_rad',999)<.005 and r.get('min_body_margin_deg',0)>=9.9)
 r['standby_and_transition_verified']=ready
 if pose_only:
  held=[x for x in ss if x['phase']=='pose_hold'];r['verified_success']=bool(d['success'] and ready and len(held)>=49 and max(x['max_tracking_error_rad'] for x in held)<.08)
 else:r['verified_success']=bool(r['verified_success'] and ready)
 print('RESTPOSE_REVIEW',r['asset'],r['workflow'],r['verified_success'],r['prepared_max_tracking_error_rad'],r['standby_arm_mirror_error_m'],flush=True)
(R/'restpose_verified_results.json').write_text(json.dumps(rows,indent=2))
