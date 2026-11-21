from pathlib import Path
import json,hashlib,numpy as np
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,CookedGeometry,ConvexPart
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;records=[]
source_for={'torquelow_track_trials':'torquelow_single_track_trial.py','torquelow_budget_trials':'torquelow_single_budget_trial.py','torquelow_single_trials':'torquelow_single_trial.py','torquelow_direct_trials':'torquelow_single_direct_trial.py','torquelow_firm_trials':'torquelow_single_firm_trial.py','torquelow_centered_trials':'torquelow_single_centered_trial.py'}
for folder,source in source_for.items():
 for f in sorted((R/folder).glob('**/result.json')):
  d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());hold=[s for s in ss if s['phase']=='hold'];sc=json.loads(f.with_name('self_contacts.json').read_text());nc=json.loads(f.with_name('nonhand_contacts.json').read_text());ec=json.loads(f.with_name('environment_contacts.json').read_text());a=d['asset'];row=dict(result=str(f.relative_to(R)),asset=a,source=source,source_sha256=d['source_sha256'],harness_success=d['success'],failure=d.get('failure'),max_lift_m=d.get('max_lift_m'),min_hold_lift_m=d.get('min_hold_lift_m'),max_hold_tilt_deg=d.get('max_hold_tilt_deg'),hold_relative_drift_m=d.get('hold_relative_drift_m'),required_contact_count=2,hold_required_contact_fraction=d.get('hold_required_contact_fraction'),hold_samples=len(hold),waist_effort_limits_nm=d.get('waist_effort_limits_nm'),waist_unlocked=d.get('waist_hold_is_unlocked'),standing_transition_validated=False,verified_success=False)
  row['source_matches']=hashlib.sha256((R/source).read_bytes()).hexdigest()==d['source_sha256'];row['max_self_impulse_ns']=max((x['impulse'] for x in sc),default=0.);row['max_nonhand_impulse_ns']=max((x['impulse'] for x in nc),default=0.);row['max_environment_impulse_ns']=max((x['impulse'] for x in ec),default=0.);row['inactive_hand_peak_force_n']=max((sum(v for n,v in s['finger_forces'].items() if n.startswith('gripper_r_')) for s in ss),default=0.);row['frozen_error_rad']=max((s['frozen_joint_max_error_rad'] for s in ss),default=999.);row['root_displacement_m']=max((float(np.linalg.norm(np.array(s['base_position'])-[0,0,.04])) for s in ss),default=999.);row['original_joint_limit_violations']=[dict(t=s['t'],joint=n) for s in ss for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic'] and (s['joint_positions'].get(n,0)<j['lower']-.005 or s['joint_positions'].get(n,0)>j['upper']+.005)][:10]
  row['max_contact_surface_error_m']=d.get('max_contact_surface_error_m');row['positive_contact_fraction_independent']=sum(all(sum(v for n,v in s['finger_forces'].items() if n.startswith(f'gripper_l_{finger}_'))>.5 for finger in ('inner','outer')) for s in hold)/max(1,len(hold))
  if folder in ('torquelow_budget_trials','torquelow_track_trials'):
   row['combined_native_clamp_verified']=bool(all(all(abs(v-50.)<1e-8 for v in s.get('waist_native_drive_caps_nm',{}).values()) and len(s.get('waist_native_drive_caps_nm',{}))==5 and all(abs(v)<1e-10 for v in s.get('waist_manual_feedforward_nm',{}).values()) and len(s.get('waist_manual_feedforward_nm',{}))==5 for s in ss))
   assert row['combined_native_clamp_verified']
  if d['success']:
   mesh=np.load(R/'object_assets'/f'{a}.npz')['vertices'];c=json.loads((R/'strategy15_surface_assets'/f'{a}_cooked.json').read_text());obj=CookedGeometry([ConvexPart(h['vertices_object_m'],str(i)) for i,h in enumerate(c['hulls'])]);hits=[];selfhits=[];worst=0.;floors=[];robotfloor=[]
   for s in ss:
    b=np.eye(4);b[:3,3]=s['base_position'];fk=k.fk(s['joint_positions'],b);rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix();pos=np.array(s['object_position']);sh=intersections(s['joint_positions'])
    if sh:selfhits.append(dict(t=s['t'],pairs=sh))
    for n,geo in robot.items():
     if n!=k.root and not n.startswith('chassis_'):robotfloor.append(float(fk[n][2,3]+fk[n][2,:3]@geo.bounds.mean(0)-abs(fk[n][2,:3])@((geo.bounds[1]-geo.bounds[0])/2)))
     if n.startswith('gripper_'):continue
     t=fk[n].copy();t[:3,3]=rot.T@(t[:3,3]-pos);t[:3,:3]=rot.T@t[:3,:3];center=t[:3,3]+t[:3,:3]@geo.bounds.mean(0);ext=abs(t[:3,:3])@((geo.bounds[1]-geo.bounds[0])/2)
     if np.any(center+ext<obj.bounds[0]) or np.any(center-ext>obj.bounds[1]):continue
     gap=obj.compound_gap(geo,t);worst=min(worst,gap)
     if gap<-.001:hits.append(dict(t=s['t'],link=n,gap_m=gap))
    if s['phase']=='hold':floors.append(float((mesh@rot.T+pos)[:,2].min()))
   row.update(nonhand_geometry_hits=hits,visible_self_intersections=selfhits,min_nonhand_gap_m=worst,min_held_floor_clearance_m=min(floors),min_robot_nonwheel_aabb_floor_gap_m=min(robotfloor))
   row['verified_success']=bool(row['source_matches'] and row['waist_unlocked'] and all(v==50 for v in row['waist_effort_limits_nm'].values()) and len(hold)>=29 and min(s['lift_m'] for s in hold)>=.15 and max(s['tilt_deg'] for s in hold)<=20 and row['hold_relative_drift_m']<=.02 and row['positive_contact_fraction_independent']>=.9 and row['max_self_impulse_ns']<=.001 and row['max_nonhand_impulse_ns']<=.001 and row['max_environment_impulse_ns']<=.001 and row['inactive_hand_peak_force_n']<=.1 and row['frozen_error_rad']<.005 and row['root_displacement_m']<.001 and not row['original_joint_limit_violations'] and row['max_contact_surface_error_m']<=.0025 and not hits and not selfhits and min(floors)>.005 and min(robotfloor)>.001)
  for name in ['samples.json','review.mp4']:row[name]=dict(server_path=str(f.with_name(name)),sha256=hashlib.sha256(f.with_name(name).read_bytes()).hexdigest())
  records.append(row);print('STRICT50_REVIEW',row['result'],row['verified_success'],row['failure'],row['max_lift_m'],flush=True)
(R/'torquelow_single_verified_results.json').write_text(json.dumps(records,indent=2))

