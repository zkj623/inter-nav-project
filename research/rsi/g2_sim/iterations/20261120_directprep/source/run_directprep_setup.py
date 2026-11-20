from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;d=json.loads((R/'narrowrest_candidate_config.json').read_text());q=d['standby_joint_positions_rad'].copy()
for side,sgn in [('l',1),('r',-1)]:
 ns=[next(n for n in q if f'arm_{side}_joint{i}' in n) for i in range(1,4)];q[ns[0]]-=sgn*np.pi;q[ns[1]]=-q[ns[1]];q[ns[2]]+=sgn*np.pi
p=json.loads((R/'narrowrest_plans/box_white.json').read_text());p['plans'][0].update(q_default=q,q_prepare=[d['work_joint_positions_rad']]);(R/'directprep_white_validation_plans').mkdir(exist_ok=True);(R/'directprep_white_validation_plans/box_white.json').write_text(json.dumps(p,indent=2))
s=(R/'narrowrest_box_white_trial.py').read_text().replace("ROOT/'narrowrest_trials'","ROOT/'directprep_trials'");s=s.replace("unchanged symmetric tucked common home","unchanged archived low work home");(R/'directprep_white_validation_trial.py').write_text(s)
