from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_directprep';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=json.loads((I/'trial_index.json').read_text());assert len(rows)==1 and rows[0]['verified_success'];r=rows[0]
for rel,h in json.loads((I/'manifest.json').read_text()).items():assert sha(I/rel)==h,rel
result=json.loads((R/r['result']).read_text());assert sha(R/'directprep_white_validation_trial.py')==result['source_sha256'];assert sha(I/'source/directprep_white_validation_trial.py')==result['source_sha256']
for n in ['samples.json','review.mp4']:assert sha(Path(r[n]['server_path']))==r[n]['sha256']
for x in json.loads((I/'geometry_input_index.json').read_text()):assert sha(Path(x['server_path']))==x['sha256']
prior=json.loads((root/'iterations/20261120_narrowrest/fixtures/narrowrest_ground_lift_config.json').read_text());cfg=json.loads((root/'ground_lift_training_config.json').read_text());st=json.loads((root/'standby_preparation_config.json').read_text())
for n in ['common_initial_joint_positions_rad','fixed_joint_positions_rad','validated_assets']:
 assert cfg[n]==prior[n],n
for n in prior:
 if 'action' in n:assert cfg[n]==prior[n],n
for p in (R/'directprep_plans').glob('*.json'):
 x=json.loads(p.read_text())['plans'][0];assert x['q_default']==st['standby_joint_positions_rad'];assert x['q_start']==cfg['common_initial_joint_positions_rad'];assert x['q_prepare']==[x['q_start']]
assert all(x['clear'] for x in json.loads((R/'directprep_audit.json').read_text())['paths']);assert r['arms_within_chassis_projection'] and not r['standby_visual_intersections'];assert st['end_to_end_verified_assets']==['box_white']
for p in (I/'source').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
print('DIRECTPREP_ARCHIVE_CHECK_OK',flush=True)
