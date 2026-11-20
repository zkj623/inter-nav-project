from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
dsearch=json.loads((R/'narrowrest_search_v2.json').read_text());candidate=dsearch['candidates'][0]
(R/'narrowrest_candidate_config.json').write_text(json.dumps(dict(label='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL',standby_joint_positions_rad=candidate['q'],work_joint_positions_rad=dsearch['work_home'],preparation_waypoints=candidate['waypoints'],preparation_seconds_per_segment=20,work_pose_changed=False,planned_visual_arm_width_m=candidate['visual_arm_bounds']['width_m'],planned_visual_base_width_m=dsearch['base_bounds']['width_m'],previous_visual_arm_width_m=dsearch['previous_arm_bounds']['width_m']),indent=2))
s=(R/'verify_restpose.py').read_text().replace("R.glob('restpose_v2_trials/**/result.json')","R.glob('narrowrest_trials/**/result.json')").replace("'restpose_verified_results.json'","'narrowrest_verified_results.json'").replace("'restpose_candidate_config.json'","'narrowrest_candidate_config.json'")
exec(compile(s,'narrowrest_motion_review','exec'))
from audit_self_collision import intersections,shapes,near
import fcl
rows=json.loads((R/'narrowrest_verified_results.json').read_text());v={p.stem:np.load(p)['vertices'] for p in (R/'robot_assets').glob('*.npz') if p.stem in k.links}
for r in rows:
 d=json.loads((R/r['result']).read_text());q=d['standby_actual_q'];base=np.eye(4);base[2,3]=.04;fk=k.fk(q,base)
 b={n:[float((vertices@fk[n][:3,:3].T+fk[n][:3,3])[:,1].min()),float((vertices@fk[n][:3,:3].T+fk[n][:3,3])[:,1].max())] for n,vertices in v.items()}
 arms=[n for n in b if n.startswith('arm_') or n.startswith('gripper_')];chassis=[n for n in b if n==k.root or n.startswith('chassis_')]
 amin=min(b[n][0] for n in arms);amax=max(b[n][1] for n in arms);bmin=min(b[n][0] for n in chassis);bmax=max(b[n][1] for n in chassis)
 r.update(visual_arm_width_m=amax-amin,visual_base_width_m=bmax-bmin,left_inset_m=bmax-amax,right_inset_m=amin-bmin,standby_visual_intersections=intersections(q))
 r['arms_within_chassis_projection']=bool(r['left_inset_m']>=.001 and r['right_inset_m']>=.001)
 body=[n for n in v if n not in arms];points={n:vertices@fk[n][:3,:3].T+fk[n][:3,3] for n,vertices in v.items()};boxes={n:(p.min(0),p.max(0)) for n,p in points.items()};candidates=[]
 for a in arms:
  for n in body:
   if near(a,n):continue
   lower=float(np.linalg.norm(np.maximum(np.maximum(boxes[a][0]-boxes[n][1],boxes[n][0]-boxes[a][1]),0.)))
   candidates.append((lower,a,n))
 objects={n:fcl.CollisionObject(shapes[n],fcl.Transform(fk[n][:3,:3],fk[n][:3,3])) for n in v};best=(float('inf'),None)
 for lower,a,n in sorted(candidates):
  if lower>best[0]:break
  gap=float(fcl.distance(objects[a],objects[n],fcl.DistanceRequest(),fcl.DistanceResult()))
  if gap<best[0]:best=(gap,[a,n])
 r['nonadjacent_arm_body_visual_gap_m']=best[0];r['closest_arm_body_pair']=best[1]
 r['verified_success']=bool(r['verified_success'] and r['arms_within_chassis_projection'] and not r['standby_visual_intersections'])
 print('NARROWREST_REVIEW',r['asset'],r['verified_success'],r['visual_arm_width_m'],r['visual_base_width_m'],r['left_inset_m'],r['right_inset_m'],flush=True)
(R/'narrowrest_verified_results.json').write_text(json.dumps(rows,indent=2))
