from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
p=json.loads((R/'directprep_white_validation_plans/box_white.json').read_text())['plans'][0]
(R/'directprep_candidate_config.json').write_text(json.dumps(dict(standby_joint_positions_rad=p['q_default'],work_joint_positions_rad=p['q_start']),indent=2))
s=(R/'verify_restpose.py').read_text().replace("restpose_v2_trials/**/result.json","directprep_trials/**/result.json").replace('restpose_verified_results.json','directprep_verified_results.json').replace('restpose_candidate_config.json','directprep_candidate_config.json')
exec(compile(s,'directprep_independent_review','exec'))
s=(R/'verify_narrowrest.py').read_text().split('from audit_self_collision import')[1];s='from audit_self_collision import'+s;s=s.replace('narrowrest_verified_results.json','directprep_verified_results.json').replace('NARROWREST_REVIEW','DIRECTPREP_REVIEW');exec(compile(s,'directprep_width_review','exec'))
# Observed rather than commanded path metrics, at recorded 10 Hz.
rows=json.loads((R/'directprep_verified_results.json').read_text());comparison={}
for label,f in [('old',R.parent/'g2-narrowrest-20261120/narrowrest_trials/narrowrest_plans/box_white/candidate0_repeat1/samples.json'),('new',R/'directprep_trials/directprep_white_validation_plans/box_white/candidate0_repeat1/samples.json')]:
 ss=[s for s in json.loads(f.read_text()) if s['phase']=='prepare_home'];fs=[k.fk(s['joint_positions']) for s in ss];record={}
 for side in ('l','r'):
  pts=np.array([f[f'arm_{side}_end_link'][:3,3] for f in fs]);el=np.array([f[f'arm_{side}_link4'][:3,3] for f in fs]);record[side]=dict(wrist_path_m=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),wrist_upward_excursion_m=float(pts[:,2].max()-pts[0,2]),elbow_abs_y_peak_m=float(abs(el[:,1]).max()))
 armnames=[n for n in k.by_name if 'arm_' in n and k.by_name[n]['kind']=='revolute'];record['actual_total_arm_travel_deg']=float(sum(abs(np.degrees(y['joint_positions'][n]-x['joint_positions'][n])) for x,y in zip(ss,ss[1:]) for n in armnames));comparison[label]=record
(R/'directprep_observed_comparison.json').write_text(json.dumps(comparison,indent=2));print('OBSERVED',json.dumps(comparison),flush=True)
