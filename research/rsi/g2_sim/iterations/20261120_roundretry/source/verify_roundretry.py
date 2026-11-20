from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
p=json.loads((R/'directprep_plans/bin_round_lidded.json').read_text())['plans'][0];(R/'roundretry_candidate_config.json').write_text(json.dumps(dict(standby_joint_positions_rad=p['q_default'],work_joint_positions_rad=p['q_start']),indent=2))
s=(R/'verify_restpose.py').read_text().replace('restpose_v2_trials/**/result.json','roundretry_trials/**/result.json').replace('restpose_verified_results.json','roundretry_verified_results.json').replace('restpose_candidate_config.json','roundretry_candidate_config.json');exec(compile(s,'roundretry_physics_review','exec'))
s='from audit_self_collision import'+(R/'verify_narrowrest.py').read_text().split('from audit_self_collision import')[1];s=s.replace('narrowrest_verified_results.json','roundretry_verified_results.json').replace('NARROWREST_REVIEW','ROUNDRETRY_REVIEW');exec(compile(s,'roundretry_width_review','exec'))
