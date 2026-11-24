"""Audit a saved G2 run without importing Isaac Sim."""
import argparse,json,math
from pathlib import Path

def audit(folder):
 folder=Path(folder);r=json.loads((folder/'result.json').read_text());t=json.loads((folder/'trajectory.json').read_text());contacts=json.loads((folder/'contacts.json').read_text())
 support=0;non_support=[]
 for event in contacts:
  for p in event.get('points',[]):
   impulse=math.sqrt(sum(v*v for v in p['impulse']))
   if impulse<=.001:continue
   if abs(p['normal'][2])>.7 and p['position'][2]<.08:support+=1
   else:non_support.append({'step':event['step'],'actor0':event['actor0'],'actor1':event['actor1'],'impulse_ns':impulse,'position':p['position'],'normal':p['normal']})
 quiet=[row for row in t[-10:] if row['distance']<=.5 and row['linear_speed']<.03 and row['angular_speed']<.05]
 jumps=[math.dist(a['position'][:2],b['position'][:2]) for a,b in zip(t,t[1:])]
 checks={'completed':r.get('success') is True and r.get('reason')=='goal_reached_and_stopped','within_task_goal_tolerance':r.get('final_distance_m',math.inf)<=.5,'final_linear_speed':r.get('final_speed_m_s',math.inf)<.03,'final_angular_speed':r.get('final_angular_speed_rad_s',math.inf)<.05,'ten_quiet_control_samples':len(quiet)==10,'stable_body':r.get('maximum_tilt_deg',math.inf)<20,'no_non_support_force_contacts':not non_support,'continuous_sampled_motion':max(jumps,default=0.)<.10,'no_reference_path_or_grid':not r.get('reference_path_supplied',True) and not r.get('offline_grid_supplied',True),'no_base_teleport':r.get('base_pose_teleport_after_reset') is False}
 if 'physics_elapsed_seconds' in r:checks['physics_timebase']=abs(r['physics_elapsed_seconds']-r['simulation_seconds'])<.02
 report={'passed':all(checks.values()),'checks':checks,'support_force_points':support,'non_support_force_points':len(non_support),'largest_non_support_impulse_ns':max((p['impulse_ns'] for p in non_support),default=0.),'first_non_support_contacts':non_support[:10],'final_distance_m':r.get('final_distance_m'),'actual_path_length_m':r.get('actual_path_length_m'),'simulation_seconds':r.get('simulation_seconds'),'physics_elapsed_seconds':r.get('physics_elapsed_seconds'),'maximum_tilt_deg':r.get('maximum_tilt_deg'),'maximum_sample_displacement_m':max(jumps,default=0.),'stationary_sample_time_span_s':t[-1]['t']-t[-10]['t'] if len(t)>=10 else None,'scope':'This task, this saved run; contact classification uses measured point normal, height and impulse, not actor names.'}
 (folder/'navigation-audit.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();report=audit(a.folder);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
