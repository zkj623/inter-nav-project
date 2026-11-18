"""Independently recompute final push criteria from recorded 10 Hz samples."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
rows=[]
for p in sorted(R.glob('strategy15_push*_trials/*/*/*/result.json')):
 d=json.loads(p.read_text());parts=p.relative_to(R).parts;variant=parts[0].removeprefix('strategy15_').removesuffix('_trials')
 harness=R/(variant+'_trial_strategy15.py');assert harness.exists(),harness
 assert hashlib.sha256(harness.read_bytes()).hexdigest()==d['source_sha256'],str(p)
 samples=json.loads(p.with_name('samples.json').read_text());hold=[r for r in samples if r['phase']=='hold'];moving=[r for r in samples if r['phase']!='settle'];push=[r for r in samples if r['phase']=='push']
 both=sum(all(sum(f for n,f in r['finger_forces'].items() if n.startswith('gripper_'+s+'_'))>.5 for s in ['l','r']) for r in push)
 checks=dict(completed_hold=len(hold)>=29,no_runtime_failure=d.get('failure') in [None,'PUSH_STOP_CRITERIA_NOT_MET'])
 if moving:
  checks.update(forward=bool(hold and hold[-1]['push_forward_m']>=.10),lateral=max(abs(r['push_lateral_m']) for r in moving)<=.03,yaw=max(abs(r['yaw_error_deg']) for r in moving)<=10.,tilt=max(r['tilt_deg'] for r in moving)<=10.,stopped=bool(hold and max(r['object_speed_m_s'] for r in hold[-10:])<=.02),released=bool(hold and max(r['total_hand_force_n'] for r in hold[-10:])<.5),contact_samples=sum(r['total_hand_force_n']>.5 for r in moving)>=5,witnesses=d.get('contact_witness_points',0)>0)
 recomputed=all(checks.values());assert recomputed==d['success'],(p,checks,d['success'])
 dual=d.get('plan',{}).get('mode','').startswith('bimanual_')
 nonhand=json.loads(p.with_name('nonhand_contacts.json').read_text())
 nonhand_peak=max((c['impulse'] for c in nonhand),default=0.)
 force_run_start=None;longest_force_run=0.
 for sample in moving:
  if sample['total_hand_force_n']>20.:
   if force_run_start is None:force_run_start=sample['t']
   longest_force_run=max(longest_force_run,sample['t']-force_run_start+.1)
  else:force_run_start=None
 force_review=longest_force_run<=1.05 and max((r['total_hand_force_n'] for r in moving),default=0.)<=35.
 accepted=bool(recomputed and (not dual or both>=5) and nonhand_peak<=.001 and len(nonhand)<500 and force_review)
 row=dict(result=str(p.relative_to(R)),asset=d['asset'],variant=variant,candidate=d['candidate'],repeat=d['repeat'],harness_success=d['success'],review_accepted=accepted,failure=d.get('failure'),checks=checks,both_hands_push_samples=both,initialization_corrected=('init' in variant or 'start' in variant),source_sha256=d['source_sha256'],video_server_path=str(p.with_name('review.mp4')))
 if variant=='push':row['classification']='superseded initialization order; not a valid capability conclusion'
 elif d.get('failure')=='HARNESS_ERROR':row['classification']='software error'
 else:row['classification']='physical trial'
 row.update({k:d.get(k) for k in ['push_forward_m','max_push_tilt_deg','max_yaw_error_deg','max_lateral_drift_m','max_hand_force_n','final_hold_max_speed_m_s','final_hold_max_hand_force_n','object_initial_position_error_m','object_initial_orientation_dot']})
 row.update(nonhand_contact_events=len(nonhand),max_nonhand_impulse_ns=nonhand_peak,nonhand_contact_buffer_not_full=len(nonhand)<500,nonhand_contact_review_passed=nonhand_peak<=.001 and len(nonhand)<500)
 row.update(max_continuous_force_above_20n_seconds=longest_force_run,force_review_passed=force_review)
 for metric,key in [('max_yaw_error_deg','yaw_error_deg'),('max_lateral_drift_m','push_lateral_m'),('max_push_tilt_deg','tilt_deg')]:
  row[metric]=max((abs(r[key]) for r in moving),default=None)
 rows.append(row)
(R/'push_result_verification.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
