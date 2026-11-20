from pathlib import Path
import json,copy,re,ast
R=Path(__file__).resolve().parent;d=json.loads((R/'narrowrest_search_v2.json').read_text());chosen=d['candidates'][0];cfg=json.loads((R/'standby_preparation_config.json').read_text()) if (R/'standby_preparation_config.json').exists() else None
out=R/'narrowrest_plans';out.mkdir(exist_ok=True)
for f in (R/'restpose_v2_plans').glob('*.json'):
 p=copy.deepcopy(json.loads(f.read_text()))
 for x in p['plans']:x.update(q_default=chosen['q'],q_prepare=chosen['waypoints'],standby_label='CUSTOM narrow arms-down standby; visual arm width constrained to chassis envelope')
 (out/f.name).write_text(json.dumps(p,indent=2))
for f in R.glob('restpose_v2_*_trial.py'):
 a=f.name.removeprefix('restpose_v2_').removesuffix('_trial.py');s=f.read_text().replace("ROOT/'restpose_v2_trials'","ROOT/'narrowrest_trials'")
 needle="    result['standby_link_positions_m']={n:standby_fk[n][:3,3].tolist() for side in ('l','r') for n in [f'arm_{side}_link2',f'arm_{side}_link4',f'arm_{side}_end_link']}";assert needle in s
 widths="""
    visual_vertices={p.stem:np.load(p)['vertices'] for p in (ROOT/'robot_assets').glob('*.npz') if p.stem in k.links}
    result['standby_visual_y_bounds_m']={n:[float((v@standby_fk[n][:3,:3].T+standby_fk[n][:3,3])[:,1].min()),float((v@standby_fk[n][:3,:3].T+standby_fk[n][:3,3])[:,1].max())] for n,v in visual_vertices.items()}
"""
 s=s.replace(needle,needle+widths);ast.parse(s);(R/f'narrowrest_{a}_trial.py').write_text(s)
settings=dict(label='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL',standby_joint_positions_rad=chosen['q'],work_joint_positions_rad=d['work_home'],preparation_waypoints=chosen['waypoints'],preparation_seconds_per_segment=20,work_pose_changed=False,planned_visual_arm_width_m=chosen['visual_arm_bounds']['width_m'],planned_visual_base_width_m=d['base_bounds']['width_m'],previous_visual_arm_width_m=d['previous_arm_bounds']['width_m'])
(R/'narrowrest_candidate_config.json').write_text(json.dumps(settings,indent=2))
print('NARROWREST_BUILT',len(list(out.glob('*.json'))))
