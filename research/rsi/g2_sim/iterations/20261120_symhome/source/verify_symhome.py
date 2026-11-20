from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
s=(R/'verify_liftpair.py').read_text().replace("R.glob('liftpair*_trials/**/result.json')","R.glob('symhome_v4*_trials/**/result.json')").replace("'liftpair_verified_results.json'","'symhome_verified_results.json'")
s=s.replace("row['frozen_error_rad']=max(s.get('frozen_joint_max_error_rad',0.) for s in samples)","row['frozen_error_rad']=max(s.get('frozen_joint_max_error_rad',0.) for s in samples)")
# Body margin is intentionally near the original lower limit at the official default. Assess work posture margin after preparation, and original limits across the full path separately.
s=s.replace("for s in samples for n,j in k.by_name.items() if 'body_joint' in n","for s in samples if s['phase'] not in ('default_settle','prepare_home','initial_settle') for n,j in k.by_name.items() if 'body_joint' in n")
exec(compile(s,'lift_geometry_review','exec'))
rows=json.loads((R/'symhome_verified_results.json').read_text());search=json.loads((R/'symhome_search.json').read_text());home=search['candidates'][0]['q'];default=search['default']
for r in rows:
 f=R/r['result'];d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());p=d.get('plan',{})
 before=[x for x in ss if x['phase']=='prepare_home'];ready=[x for x in ss if x['phase']=='settle' and x['t']>10]
 r['identical_symmetric_home']=p.get('q_start')==home;r['official_default_targets']=p.get('q_default')==default
 r['preparation_recorded']=bool(before and ready and any(x['phase']=='common_approach' for x in ss))
 r['prepared_error_rad']=d.get('prepared_max_tracking_error_rad')
 r['preparation_max_tracking_error_rad']=max((x['max_tracking_error_rad'] for x in before),default=999.)
 profile=d.get('preparation_motion',[]);peaks=[]
 for segment in profile:
  t=segment['duration_s'];dv={n:abs(segment['end'].get(n,0.)-segment['start'].get(n,0.)) for n in home}
  peaks.append(dict(duration_s=t,max_velocity_fraction=max(2.1875*v/t/k.by_name[n]['velocity'] for n,v in dv.items()),max_acceleration_rad_s2=max(dv.values())*7.514/t**2,max_jerk_rad_s3=max(dv.values())*52.5/t**3))
 r['preparation_profile_peaks']=peaks
 r['preparation_actual_peak_velocity_fraction']=max((abs(x['joint_velocities'].get(n,0.))/k.by_name[n]['velocity'] for x in before for n in home),default=999.)
 r['preparation_fd_peak_velocity_fraction']=max((abs(y['joint_positions'].get(n,0.)-x['joint_positions'].get(n,0.))/((y['t']-x['t'])*k.by_name[n]['velocity']) for x,y in zip(before,before[1:]) for n in home),default=999.)
 sc=json.loads(f.with_name('self_contacts.json').read_text());r['max_self_contact_impulse_ns']=max((x['impulse'] for x in sc),default=0.)
 active=[n for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic']]
 violations=[]
 for x in ss:
  for n in active:
   j=k.by_name[n];v=x['joint_positions'].get(n,0.)
   if v<j['lower']-.005 or v>j['upper']+.005:violations.append([x['t'],n,v])
 r['original_limit_violations']=violations[:10]
 fk=k.fk(ready[-1]['joint_positions']) if ready else None
 if fk:
  r['elbow_lateral_span_m']=abs(fk['arm_l_link4'][1,3]-fk['arm_r_link4'][1,3])
  r['arm_position_mirror_error_m']=max(float(np.linalg.norm(fk[f'arm_r_link{i}'][:3,3]-fk[f'arm_l_link{i}'][:3,3]*[1,-1,1])) for i in range(2,8))
  r['wrist_mirror_error_m']=float(np.linalg.norm(fk['arm_r_end_link'][:3,3]-fk['arm_l_end_link'][:3,3]*[1,-1,1]))
 # Smooth motion and precise following are separate observations. Retain the
 # runtime's original0.25rad tracking rejection for acceptance, and disclose
 # the stricter exploratory0.1rad precision goal independently; do not equate
 # a smooth delayed response with exact following of the command trajectory.
 r['precision_tracking_within_0p1_rad']=r['preparation_max_tracking_error_rad']<.10
 r['smooth_transition_verified']=bool(r['identical_symmetric_home'] and r['official_default_targets'] and r['preparation_recorded'] and r['prepared_error_rad'] is not None and r['prepared_error_rad']<.03 and r['preparation_max_tracking_error_rad']<.25 and not violations and r.get('arm_position_mirror_error_m',999)<.005 and peaks and all(x['max_velocity_fraction']<=.400001 and x['max_acceleration_rad_s2']<=.500001 and x['max_jerk_rad_s3']<=2.000001 for x in peaks) and r['preparation_actual_peak_velocity_fraction']<=1. and r['preparation_fd_peak_velocity_fraction']<=1. and r['max_self_contact_impulse_ns']<=.001)
 r['verified_success']=bool(r['verified_success'] and r['smooth_transition_verified'])
 print('SYMHOME_REVIEW',r['asset'],r['verified_success'],r.get('prepared_error_rad'),r.get('arm_position_mirror_error_m'),flush=True)
(R/'symhome_verified_results.json').write_text(json.dumps(rows,indent=2))
