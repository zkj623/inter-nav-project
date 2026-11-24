from pathlib import Path
import json,numpy as np,sys,hashlib
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;T=Path(sys.argv[1]);v=json.loads((T/'verification.json').read_text());res=json.loads((T/'result.json').read_text());ss=json.loads((T/'samples.json').read_text());k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');sign=np.array([-1,1,-1,1,-1,1,-1]);nl=[f'idx{21+i}_arm_l_joint{i+1}' for i in range(7)];nr=[f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)];by_phase={};maxdelta=0.;height=-99.;span=0.
for s in ss:
 q=s['joint_positions'];delta=float(np.max(np.abs(np.degrees(np.array([q[n] for n in nr])-sign*np.array([q[n] for n in nl])))));maxdelta=max(maxdelta,delta);by_phase[s['phase']]=max(by_phase.get(s['phase'],0.),delta)
 if s['phase'] in ['close','contact_refine','lift','contact_settle','hold']:
  fk=k.fk(q,np.array(res['plan']['base_pose']));height=max(height,max(fk[f'arm_{side}_link4'][2,3]-fk[f'arm_{side}_link1'][2,3] for side in ['l','r']));span=max(span,abs(fk['arm_l_link4'][1,3]-fk['arm_r_link4'][1,3]))
source_ok=hashlib.sha256((R/'posture_anchor.py').read_bytes()).hexdigest()==res.get('posture_anchor_sha256')
waypoints=[res['plan']['q_close']]+res['plan']['q_lift'];max_lift_jump=max(float(np.degrees(max(abs(x[n]-y[n]) for n in nl+nr))) for x,y in zip(waypoints,waypoints[1:]))
out={'asset':res['asset'],'physics_verified':v['verified_success'],'helper_source_matches':source_ok,'max_actual_mirror_joint_error_deg':maxdelta,'by_phase_deg':by_phase,'max_elbow_above_shoulder_m':height,'max_elbow_span_m':span,'criteria':{'max_actual_mirror_joint_error_deg':15.,'max_elbow_above_shoulder_m':.10},'natural_verified':bool(v['verified_success'] and source_ok and maxdelta<=15. and height<=.10),'scope':'all sampled joint records; elbow posture evaluated during loaded interaction; visual review also required'}
out['max_planned_lift_waypoint_delta_deg']=max_lift_jump;out['criteria']['max_planned_lift_waypoint_delta_deg']=30.;out['natural_verified']=out['natural_verified'] and max_lift_jump<=30.
(T/'posture_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
