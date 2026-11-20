from pathlib import Path
import json,shutil,hashlib,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_restpose';I.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((R/'restpose_verified_results.json').read_text());old=json.loads((R/'ground_lift_training_config.json').read_text());start=json.loads((R/'restpose_candidate_config.json').read_text());paths=json.loads((R/'restpose_v2_path_audit.json').read_text())
pose=next(r for r in rows if r['workflow']=='standby_hold_and_preparation');assert pose['verified_success'];assert len(paths)==12 and all(r['standby_preparation_approach_clear'] for r in paths)
full=[r for r in rows if r['workflow']=='standby_preparation_and_grasp'];assert {r['asset'] for r in full}=={'box_white','case_handheld'},'Finish both example runs before publishing'
verified=sorted({r['asset'] for r in full if r['verified_success']});config={**start,**dict(source_iteration='iterations/20261120_restpose',preferred_standby=True,standby_is_official=False,preparation_verified=True,end_to_end_verified_assets=verified,end_to_end_tested_assets=sorted(r['asset'] for r in full),other_assets_not_end_to_end_retested=sorted(set(r['asset'] for r in paths)-{r['asset'] for r in full}),geometrically_audited_assets=[r['asset'] for r in paths],status=f'custom arms-down standby and20s preparation verified; {len(verified)}/2 complete example runs verified; remaining workflows not rerun',physics_scope='pinned root, existing ideal torso drive; not hardware verification; work home and original angular limits/arm effort/contact acceptance retained')}
config['execution']={a:dict(iteration='20261120_restpose',harness=f'restpose_v2_{a}_trial.py',plan_dir='restpose_v2_plans',candidate=0,verified_end_to_end=a in verified) for a in old['validated_assets']}
(R/'restpose_preparation_config.json').write_text(json.dumps(config,indent=2));(root/'standby_preparation_config.json').write_text(json.dumps(config,indent=2))
cfg=json.loads(json.dumps(old));cfg['previous_validation_default_joint_positions_rad']=cfg['default_joint_positions_rad'];cfg['default_joint_positions_rad']=start['standby_joint_positions_rad'];cfg['standby_preparation_config']='standby_preparation_config.json';cfg['standby_label']='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL';cfg['standby_end_to_end_validated_assets']=verified
cfg['validated_assets_scope']='Archived lift baselines from the unchanged low work pose using prior scene startup; NOT all revalidated from the new arms-down standby. See standby_end_to_end_validated_assets.'
cfg['validation_scope']=f'Fixed-root ideal-torso simulation. Archived11/12 low-work-home lift baselines used prior scene startup; current custom arms-down standby/preparation and {len(verified)} complete examples are separately verified. Original50Nm torso effort feasibility and hardware execution are not verified.'
cfg['previous_validation_plan_directory']=cfg['plan_directory'];cfg['previous_validation_plan_directory_overrides']=cfg['per_asset_plan_directory_overrides'];cfg['plan_directory']='restpose_v2_plans';cfg['per_asset_plan_directory_overrides']={a:'restpose_v2_plans' for a in cfg['per_asset_plan_directory_overrides']}
cfg['initialization_scope']='Preferred startup: custom symmetric arms-down standby ->20s preparation -> unchanged symmetric low work home; lock torso before arm/gripper policy'
cfg['preparation_trajectory']['source_default_yaml']=None;cfg['preparation_trajectory']['source_standby_config']='standby_preparation_config.json';cfg['preparation_trajectory']['waypoints']=start['preparation_waypoints'];cfg['preparation_trajectory']['standby_is_official']=False
cfg['plan_resolution']='Archived validated_simulation_baselines reproduce the previous startup; current preferred arms-down startup uses execution entries in standby_preparation_config.json.'
cfg['source_iteration']='iterations/20261120_restpose';cfg['previous_config_archive']='iterations/20261120_symhome/fixtures/symhome_ground_lift_config.json';cfg['status']=f'arms-down standby/preparation verified; {len(verified)} complete example workflows verified from new standby;11/12 archived low-work-home lift baselines; collection awaits review'
(R/'restpose_ground_lift_config.json').write_text(json.dumps(cfg,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(cfg,indent=2))
def cp(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*restpose*.py'):ast.parse(p.read_text());cp(p,Path('source')/p.name)
for pat in ['restpose*.json','restpose_v2_plans/*.json']:
 for p in R.glob(pat):cp(p,Path('fixtures')/p.relative_to(R))
for r in rows:
 for n in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=(R/r['result']).with_name(n)
  if f.exists():cp(f,Path('results')/f.relative_to(R))
(I/'trial_index.json').write_text(json.dumps(rows,indent=2));(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\n')
table='\n'.join(f"| {r['asset']} | {r['workflow']} | {r['verified_success']} | {r.get('failure')} | {r['standby_arm_mirror_error_m']} | {r['prepared_max_tracking_error_rad']} |" for r in rows)
(I/'README.md').write_text(f'''# Custom arms-down simulation standby

This replaces the preferred raised-hand scene startup with a custom upright, arms-down, mirror-symmetric standby. It is NOT an official G2 factory standby. Work-home joint targets are byte-for-byte unchanged from20261120_symhome; policy action dimensions remain16 and torso freezes only after preparation. Grippers idle at-0.3rad. Twenty-second seventh-order position and analytical velocity targets connect standby directly to the existing low work home, without pose resets after initialization.

Standby hold and preparation independently passed. All12 object-inclusive standby/preparation/approach paths passed native and visible collision screening at<=0.06rad increments. This revision physically tested only white box and hard case full workflows: {len(verified)}/2 verified; other full workflows are not automatically revalidated from the new start. The historical11/12 lift baselines retain their prior startup and remain archived. The root validated_assets_scope and separate standby_end_to_end_validated_assets make that distinction explicit. No data collection/training started.

The first exploration mirrored arm link origins but missed the final wrist offset: full end-link matching showed joint6 uses the SAME angle sign across sides. The corrected map is[-1,+1,-1,+1,-1,+1,-1] for joints1..7; two generic configurations audit both end-link position and orientation using the local reflected end frame. The old v1 preparation sample failed the independent wrist mirror check (~3cm) and is excluded; two v1 example runs were stopped. Final v2 samples verify actual arm/end positions and orientation, original joint ranges, real sampled speed, self contact<=0.001Ns, settling errors and original full lift acceptance.

**Scope:** existing pinned root and ideal torso drive. Upright standby is modeled dynamically; preparation to the low work home still has the previously documented original-50Nm torso limit mismatch. No hardware feasibility claim. Command smoothness does not mean zero tracking lag; actual measured errors are indexed. The additional0.1rad precise-following goal from the preceding iteration is not asserted here.

| Asset | Workflow | Verified | Failure | Standby mirror error m | Prepared tracking rad |
| --- | --- | --- | --- | --- | --- |
{table}

Overlay previous symhome/caseyaw runtime dependencies, then source/fixtures here into one workspace. Preview: ISAAC_PY restpose_v2_box_white_trial.py --asset box_white --plan-dir restpose_v2_plans --pose-only --gpu GPU. Complete example: same command without --pose-only and with --repeat 2. Each asset's preferred runtime is indexed by ../../standby_preparation_config.json.execution. Original mass/materials/arm effort/contact limits remain unchanged. Samples/videos stay at SHA-indexed server paths; images and videos are in the existing Feishu review document.
''')
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
f=root/'README.md';s=f.read_text();banner='<!-- restpose -->\nPreferred startup now uses a [custom arms-down standby](standby_preparation_config.json), then the unchanged low work home. [Preparation and example validation](iterations/20261120_restpose/README.md). Old11 lift baselines retain their archived startup; new-start complete examples are separately indexed. Simulation only, no collection.\n<!-- /restpose -->\n\n'
if '<!-- restpose -->' not in s:f.write_text(banner+s)
out=R/'restpose_review';out.mkdir(exist_ok=True)
for name in ['standby_front.png','standby_oblique.png','initial_topview.png','review.mp4']:
 shutil.copy2((R/pose['result']).with_name(name),out/('standby_transition.mp4' if name=='review.mp4' else name))
for r in full:shutil.copy2(r['review.mp4']['server_path'],out/(r['asset']+'.mp4'))
(out/'selection.json').write_text(json.dumps(rows,indent=2));print('RESTPOSE_PACKAGED',len(rows),verified)
