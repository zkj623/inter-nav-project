from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_narrowrest';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in json.loads((I/'manifest.json').read_text()).items():assert sha(I/n)==h,n
for p in (I/'source').glob('*.py'):ast.parse(p.read_text())
cfg=json.loads((root/'ground_lift_training_config.json').read_text());old=json.loads((R/'ground_lift_training_config.json').read_text());stand=json.loads((root/'standby_preparation_config.json').read_text());rows=json.loads((I/'trial_index.json').read_text());assert len(rows)==1 and rows[0]['verified_success']
for key in ['common_initial_joint_positions_rad','fixed_joint_positions_rad','policy_action_joint_names','validated_simulation_baselines']:assert cfg[key]==old[key],key
assert cfg['default_joint_positions_rad']==stand['standby_joint_positions_rad'];assert stand['standby_is_official'] is False;assert cfg['standby_end_to_end_validated_assets']==['box_white']
assert stand['width_validation']['arms_within_chassis_projection'] and min(stand['width_validation'][k] for k in ['left_inset_m','right_inset_m'])>=.001
for g in json.loads((I/'geometry_input_index.json').read_text()):assert sha(Path(g['server_path']))==g['sha256']
for r in rows:
 assert r['source_sha256'] in {sha(p) for p in (I/'source').glob('narrowrest_*trial.py')}
 for key in ['samples.json','review.mp4']:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
for p in (I/'fixtures/narrowrest_plans').glob('*.json'):
 for plan in json.loads(p.read_text())['plans']:assert plan['q_default']==cfg['default_joint_positions_rad'] and plan['q_start']==cfg['common_initial_joint_positions_rad']
print('NARROWREST_ARCHIVE_OK')
