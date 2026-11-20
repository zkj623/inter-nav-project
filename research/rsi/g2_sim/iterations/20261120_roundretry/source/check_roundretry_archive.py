from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_roundretry';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cfg=json.loads((root/'ground_lift_training_config.json').read_text());previous=json.loads((root/'iterations/20261120_directprep/fixtures/directprep_ground_lift_config.json').read_text());st=json.loads((root/'standby_preparation_config.json').read_text())
for n in ['default_joint_positions_rad','common_initial_joint_positions_rad','fixed_joint_positions_rad','policy_action_joint_names','policy_action_dimension','preparation_trajectory']:assert cfg[n]==previous[n],n
for a,v in previous['validated_simulation_baselines'].items():assert cfg['validated_simulation_baselines'][a]==v,a
assert len(cfg['validated_assets'])==12 and set(cfg['validated_assets'])==set(cfg['geometry_screen_pass_assets']) and not cfg['pending_adaptation_assets'];assert set(cfg['standby_end_to_end_validated_assets'])=={'box_white','bin_round_lidded'};assert len(st['other_assets_not_end_to_end_retested'])==10
rows=json.loads((I/'trial_index.json').read_text());assert sum(r['verified_success'] for r in rows)==1;sel=json.loads((R/'roundretry_selection.json').read_text());r=next(r for r in rows if r['result']==sel['selected_result']);assert r['verified_success'];result=json.loads((R/r['result']).read_text());assert sha(R/'roundretry_trial.py')==result['source_sha256'];assert sha(I/'source/roundretry_trial.py')==result['source_sha256'];assert result['plan']['q_start']==cfg['common_initial_joint_positions_rad'];assert result['plan']['q_default']==cfg['default_joint_positions_rad'];assert result['hold_relative_drift_m']<=.02 and result['max_contact_surface_error_m']<=.0025
for row in rows:
 for name in ['samples.json','review.mp4']:
  if name in row:assert sha(Path(row[name]['server_path']))==row[name]['sha256']
for p in (I/'source').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for x in json.loads((I/'geometry_input_index.json').read_text()):assert sha(Path(x['server_path']))==x['sha256']
for rel,h in json.loads((I/'manifest.json').read_text()).items():assert sha(I/rel)==h
print('ROUNDRETRY_ARCHIVE_CHECK_OK',flush=True)
