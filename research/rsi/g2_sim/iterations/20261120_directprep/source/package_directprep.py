from pathlib import Path
import json,hashlib,shutil,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_directprep'
rows=json.loads((R/'directprep_verified_results.json').read_text());assert len(rows)==1 and rows[0]['verified_success'];r=rows[0];audit=json.loads((R/'directprep_audit.json').read_text());assert len(audit['paths'])==12 and all(x['clear'] for x in audit['paths']);candidate=json.loads((R/'directprep_candidate_config.json').read_text());observed=json.loads((R/'directprep_observed_comparison.json').read_text())
stand=json.loads((root/'standby_preparation_config.json').read_text());oldcfg=json.loads((root/'ground_lift_training_config.json').read_text());cfg=json.loads(json.dumps(oldcfg));assert candidate['work_joint_positions_rad']==cfg['common_initial_joint_positions_rad'];assert candidate['work_joint_positions_rad']==stand['work_joint_positions_rad']
stand.update(candidate,source_iteration='iterations/20261120_directprep',preparation_waypoints=[candidate['work_joint_positions_rad']],status='Equivalent shoulder IK branch removes shoulder reconfiguration detour;20s smooth transition and one white-box full grasp verified;others geometry only',previous_config_archive='iterations/20261120_narrowrest/fixtures/narrowrest_preparation_config.json',end_to_end_verified_assets=['box_white'],end_to_end_tested_assets=['box_white'],geometrically_audited_assets=[x['asset'] for x in audit['paths']],other_assets_not_end_to_end_retested=[x['asset'] for x in audit['paths'] if x['asset']!='box_white'])
stand['width_validation']={n:r[n] for n in ['visual_arm_width_m','visual_base_width_m','left_inset_m','right_inset_m','arms_within_chassis_projection','nonadjacent_arm_body_visual_gap_m']};stand['width_validation']['scope']='settled standby visual projection only';stand['execution']={a:dict(iteration='20261120_directprep',harness='directprep_white_validation_trial.py' if a=='box_white' else f'directprep_{a}_trial.py',plan_dir='directprep_white_validation_plans' if a=='box_white' else 'directprep_plans',candidate=0,verified_end_to_end=a=='box_white') for a in cfg['validated_assets']}
stand['transition_comparison']=observed
cfg.update(default_joint_positions_rad=candidate['standby_joint_positions_rad'],previous_standby_joint_positions_rad=oldcfg['default_joint_positions_rad'],source_iteration='iterations/20261120_directprep',previous_config_archive='iterations/20261120_narrowrest/fixtures/narrowrest_ground_lift_config.json',status=stand['status'],plan_directory='directprep_plans',standby_width_validation=stand['width_validation'],initialization_scope='Custom narrow arms-down standby in work-compatible shoulder IK branch ->20s preparation -> unchanged archived low work home;torso locked for arm/gripper policy')
cfg['per_asset_plan_directory_overrides']={a:('directprep_white_validation_plans' if a=='box_white' else 'directprep_plans') for a in cfg['validated_assets']};cfg['preparation_trajectory']['waypoints']=[candidate['work_joint_positions_rad']]
cfg['validation_scope']='Fixed-root ideal-torso simulation. Only one complete white-box workflow rerun from equivalent shoulder-branch standby;12 preparation paths geometry screened. Archived11/12 baselines retain prior startup. Original torso torque/hardware feasibility not verified.'
for n in ['common_initial_joint_positions_rad','fixed_joint_positions_rad','action_joint_names','validated_assets']:
 if n in oldcfg:assert cfg[n]==oldcfg[n]
(R/'directprep_preparation_config.json').write_text(json.dumps(stand,indent=2));(R/'directprep_ground_lift_config.json').write_text(json.dumps(cfg,indent=2))
I.mkdir(exist_ok=True)
def cp(p,rel):
 t=I/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
for p in R.glob('*directprep*.py'):ast.parse(p.read_text(encoding='utf-8-sig'));cp(p,Path('source')/p.name)
for pattern in ['directprep*.json','directprep_plans/*.json','directprep_white_validation_plans/*.json']:
 for p in R.glob(pattern):cp(p,Path('fixtures')/p.relative_to(R))
f=R/r['result']
for n in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:cp(f.with_name(n),Path('results')/f.with_name(n).relative_to(R))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(I/'trial_index.json').write_text(json.dumps(rows,indent=2));(I/'geometry_input_index.json').write_text(json.dumps([dict(server_path=str(p),sha256=sha(p)) for p in [R/'generated/g2_crs_omnipicker.urdf']+list((R/'robot_assets').glob('*.npz'))],indent=2))
(I/'README.md').write_text('''# Reduce standby-to-work shoulder detour

The prior narrow standby and archived work home used different spherical-shoulder IK branches. Straight joint interpolation was monotonic but rotated shoulder joints1/3 about176/143deg, causing hands to rise and elbows to spread before reaching the low home. Changing duration alone would not remove this geometric detour.

For left shoulder joints1/2/3 use q1-pi,-q2,q3+pi; mirror the right with q1+pi,-q2,q3-pi. The shoulder-axis URDF rounding leaves only micrometre-scale end-pose differences. Shoulder joint1 travel becomes about3.5deg and joint3 about37.4deg. Complete original visual meshes and collisions were checked again; no external asset changes. This changes the joint representation of standby, preserving its arms-down end pose and chassis width requirement, and keeps the entire work home, body targets,16-dimensional policy and grasp plans unchanged. It does not reduce every joint's travel; elbow flexion and body lowering remain required.

See fixtures/directprep_audit.json for85 samples per object preparation route, old/new planned paths and total joint travel; fixtures/directprep_observed_comparison.json for10Hz observed motion from physics; trial_index.json for independent limits/clearance/full-lift verification. All12 preparation routes passed geometry screening. Only box_white completed a new full physics grasp/lift; other complete workflows were not rerun. The seventh-order20s preparation is retained, without intermediate stops. Original ideal torso drive, pinned root and lift/contact thresholds are unchanged. This is a custom simulation standby, not an official factory state or real-hardware validation. No data collection or policy training started.

Overlay20261120_narrowrest runtime dependencies plus source/fixtures here. Run directprep_white_validation_trial.py --asset box_white --plan-dir directprep_white_validation_plans --gpu GPU. Consult root standby_preparation_config.json.execution for asset-specific sources. Source/sample/video SHA256 and geometry hashes are archived. Videos are provided in the ongoing Feishu review document.
''')
(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\n');(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
(root/'standby_preparation_config.json').write_text(json.dumps(stand,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(cfg,indent=2));f=root/'README.md';s=f.read_text();marker='<!-- directprep -->'
if marker not in s:f.write_text(marker+'\nPreferred startup uses a work-compatible shoulder IK branch to remove the arm swing detour. [Verification](iterations/20261120_directprep/README.md). Low work home unchanged; one white-box full workflow verified.\n<!-- /directprep -->\n\n'+s)
for p in I.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for rel,digest in json.loads((I/'manifest.json').read_text()).items():assert sha(I/rel)==digest
print('DIRECTPREP_PACKAGE_OK',r['verified_success'],flush=True)
