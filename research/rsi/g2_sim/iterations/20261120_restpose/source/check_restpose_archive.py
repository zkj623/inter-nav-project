from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_restpose';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in json.loads((I/'manifest.json').read_text()).items():assert sha(I/n)==h,n
for p in (I/'source').glob('*.py'):ast.parse(p.read_text())
cfg=json.loads((root/'ground_lift_training_config.json').read_text());old=json.loads((R/'ground_lift_training_config.json').read_text());stand=json.loads((root/'standby_preparation_config.json').read_text());rows=json.loads((I/'trial_index.json').read_text())
assert cfg['common_initial_joint_positions_rad']==old['common_initial_joint_positions_rad']==stand['work_joint_positions_rad']
assert cfg['fixed_joint_positions_rad']==old['fixed_joint_positions_rad'];assert cfg['policy_action_joint_names']==old['policy_action_joint_names']
assert cfg['validated_simulation_baselines']==old['validated_simulation_baselines']
assert cfg['default_joint_positions_rad']==stand['standby_joint_positions_rad'];assert stand['standby_is_official'] is False
assert cfg['plan_directory']=='restpose_v2_plans';assert all(v=='restpose_v2_plans' for v in cfg['per_asset_plan_directory_overrides'].values())
assert cfg['standby_end_to_end_validated_assets']==sorted({r['asset'] for r in rows if r['verified_success'] and r['workflow']=='standby_preparation_and_grasp'})
for r in rows:
 assert r['source_sha256'] in {sha(p) for p in (I/'source').glob('restpose_v2_*trial.py')}
 for key in ['samples.json','review.mp4']:assert sha(Path(r[key]['server_path']))==r[key]['sha256']
for p in (I/'fixtures/restpose_v2_plans').glob('*.json'):
 for plan in json.loads(p.read_text())['plans']:
  assert plan['q_default']==stand['standby_joint_positions_rad'];assert plan['q_start']==cfg['common_initial_joint_positions_rad']
print('RESTPOSE_ARCHIVE_OK',len(rows),cfg['standby_end_to_end_validated_assets'])
