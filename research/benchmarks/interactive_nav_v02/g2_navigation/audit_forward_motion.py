"""Measure actual translation directions relative to the measured robot heading."""
import argparse,json,math
from pathlib import Path
import numpy as np

def audit(folder):
 folder=Path(folder);rows=json.loads((folder/'trajectory.json').read_text());distances={'forward':0.,'lateral':0.,'reverse':0.,'other':0.};turn=0.;stationary_turn=0.;modes={};sampled_distance=0.;low_speed_distance=0.
 for a,b in zip(rows,rows[1:]):
  delta=np.asarray(b['position'][:2])-a['position'][:2];length=float(np.linalg.norm(delta));angle=(b['yaw']-a['yaw']+math.pi)%(2*math.pi)-math.pi;turn+=abs(angle)
  if max(a['linear_speed'],b['linear_speed'])<.03:stationary_turn+=abs(angle)
  sampled_distance+=length;heading=a['yaw']+.5*angle;f=float(delta@[math.cos(heading),math.sin(heading)])/max(length,1e-12)
  if max(a['linear_speed'],b['linear_speed'])>.015:
   key='forward' if f>=math.cos(math.pi/6) else 'reverse' if f<-.5 else 'lateral' if abs(f)<=.5 else 'other';distances[key]+=length
  else:low_speed_distance+=length
  mode=a.get('motion_mode','unclassified');modes[mode]=modes.get(mode,0)+1
 total=sum(distances.values());result={'total_sampled_path_length_m':sampled_distance,'low_speed_distance_m':low_speed_distance,'moving_distance_m':total,'actual_moving_distance_by_direction_m':distances,'forward_fraction_of_total_path':distances['forward']/max(sampled_distance,1e-9),'lateral_fraction_of_total_path':distances['lateral']/max(sampled_distance,1e-9),'forward_distance_fraction':distances['forward']/max(total,1e-9),'lateral_distance_fraction':distances['lateral']/max(total,1e-9),'total_absolute_yaw_change_deg':math.degrees(turn),'stationary_yaw_change_deg':math.degrees(stationary_turn),'motion_mode_samples':modes,'definition':'Classify measured displacement against midpoint measured body yaw when either endpoint linear speed exceeds 0.015 m/s. Forward is within 30 degrees; lateral is 60-120 degrees; reverse is more than 120 degrees. Fractions of total path keep low-speed displacement in the denominator. Stationary turns have both endpoint speeds below 0.03 m/s.'};(folder/'forward-motion-audit.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');print(json.dumps(audit(p.parse_args().folder),indent=2))
