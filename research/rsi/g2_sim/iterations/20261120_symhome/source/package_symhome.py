from pathlib import Path
import json,hashlib,shutil,ast
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261120_symhome';I.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((R/'symhome_verified_results.json').read_text());old=json.loads((R/'ground_lift_training_config.json').read_text());search=json.loads((R/'symhome_search.json').read_text());home=search['candidates'][0]['q'];default=search['default'];cfg=json.loads(json.dumps(old));passed=sorted({r['asset'] for r in rows if r['verified_success']});assets=sorted(old['validated_assets'])
assert set(r['asset'] for r in rows)==set(assets),'Complete all12 physical trials before changing active home'
cfg.update(name='g2_ground_lift_symmetric_home_v3',common_initial_joint_positions_rad=home,default_joint_positions_rad=default,plan_directory='symhome_plans',initialization_scope='official scene_flat_g2_op defaults -> continuous seventh-order preparation -> identical symmetric common home -> per-object approach',joint_lock_method='root pinned throughout; original torso angular limits during preparation; torso angular limits fixed at new home only after settling; arms/grippers actuated',validated_assets=passed,pending_adaptation_assets=sorted(set(assets)-set(passed)),source_iteration='iterations/20261120_symhome',previous_config_archive='iterations/20261119_caseyaw/fixtures/caseyaw_ground_lift_config.json',validation_scope='Fixed-root simulation with existing ideal torso drives; official default preparation and symmetric home; not feasible under unmodified 50Nm URDF torso effort caps, not hardware validation',status=f'{len(passed)}/12 default preparation + symmetric home + lift simulations independently verified; collection awaits review')
cfg['validated_simulation_baselines']={}
for a in passed:
 r=next(x for x in rows if x['asset']==a and x['verified_success']);parts=Path(r['result']).parts
 matches=[p.name for p in R.glob('symhome_v4*trial.py') if sha(p)==r['source_sha256']]
 harness=next((n for n in matches if n.endswith('_'+a+'_trial.py')),matches[0])
 cfg['validated_simulation_baselines'][a]={**old['validated_simulation_baselines'][a],**dict(iteration='20261120_symhome',harness=harness,plan_dir=parts[1],candidate=0,repeat=int(parts[3].split('_repeat')[1]))}
 d=json.loads((R/r['result']).read_text())
 for key in ['pinch_force_feedback','feedback_controller']:
  if key in d:cfg['validated_simulation_baselines'][a][key]=d[key]
cfg['per_asset_plan_directory_overrides']={a:'symhome_plans' for a in assets};cfg['latest_baseline_verified_repeat_counts']={a:sum(r['asset']==a and r['verified_success'] for r in rows) for a in passed}
cfg['preparation_trajectory']=dict(profile='7th order 35u^4-84u^5+70u^6-20u^7; endpoint velocity/acceleration/jerk zero',minimum_duration_s=20,maximum_command_velocity_fraction=.4,maximum_command_acceleration_rad_s2=.5,maximum_command_jerk_rad_s3=2,waypoints=search['candidates'][0]['waypoints'],source_default_yaml='vendor/genie_sim_drive_audit/source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config/scene_flat_g2_op.yaml',root_pinned=True,ideal_torso_drive=True)
cfg['preparation_trajectory'].update(analytical_joint_velocity_feedforward=True,outside_policy=True,policy_start_phase='settle after preparation and torso lock')
cfg['assumptions']['original_torso_effort_transition_verified']=False
cfg['preparation_trajectory']['tracking_acceptance_rad']=.25
cfg['preparation_trajectory']['precision_tracking_goal_rad']=.1
cfg['preparation_trajectory']['precision_tracking_goal_met']=all(r['precision_tracking_within_0p1_rad'] for r in rows)
cfg['original_torso_effort_audit']='iterations/20261120_symhome/fixtures/symhome_original_effort_audit.json'
(R/'symhome_ground_lift_config.json').write_text(json.dumps(cfg,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(cfg,indent=2))
def cp(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*symhome*.py'):ast.parse(p.read_text());cp(p,Path('source')/p.name)
for pat in ['symhome*.json','symhome_plans/*.json']:
 for p in R.glob(pat):cp(p,Path('fixtures')/p.relative_to(R))
default_source=R/cfg['preparation_trajectory']['source_default_yaml'];cp(default_source,Path('fixtures/official_scene_flat_g2_op.yaml'))
for r in rows:
 for n in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  p=(R/r['result']).with_name(n)
  if p.exists():cp(p,Path('results')/p.relative_to(R))
(I/'trial_index.json').write_text(json.dumps(rows,indent=2));(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\n')
table='\n'.join(f"| {r['asset']} | {r['smooth_transition_verified']} | {r['verified_success']} | {r.get('failure')} | {r.get('min_hold_lift_m')} | {r.get('max_hold_tilt_deg')} | {r.get('four_contact_fraction')} |" for r in rows)
(I/'README.md').write_text(f'''# Symmetric tucked common home and continuous preparation

The old common reset was shared across objects but asymmetric between arms. This revision mirrors arm positions and reduces elbow-axis lateral span from1.053m to0.762m. Work torso angles remain[-40,142,-87,0,0]deg. Both wrists at about[x0.552,y+/-0.276,z0.726]m. Exact initial targets are identical across all twelve objects.

Preparation starts at the shipped scene_flat_g2_op.yaml joint defaults (degrees converted to radians; omitted joints zero; root at z0.04 fixed to world, unlike that YAML free-base mode). It drives all preparation joints through original angular ranges with a seventh-order profile, >=20s, peak speed<=40% published velocity, commanded acceleration<=0.5rad/s2 and jerk<=2rad/s3, zero endpoint velocity/acceleration/jerk. Analytical joint-velocity targets accompany position targets during preparation; zero velocity targets are restored afterward. After settling, torso joint limits lock at the work posture. Preparation is outside the16-dimensional arm/gripper policy. No reset or robot/object pose writes after default initialization. Hands approach the old audited per-object pregrasp. Original grasp/lift controllers are retained except the explicitly indexed basket force feedback and round-lidded-bin balance comparisons. Runtime camera-only top-view snapshots do not advance physics.

**Scope limitation:** existing ideal torso drives and pinned root are retained. Independent static gravity audit under original URDF50Nm torso caps gives maximum required226.0Nm (4.52x) along this preparation. Thus smooth simulation is NOT a demonstrated motor-feasible or hardware trajectory. This limitation is explicit in active configuration. Arms retain original effort caps; held non-arm joints/root are fixed as before. Do not use this preparation for hardware without a corrected torque model and separate validation.

All12 object-inclusive preparation/approach paths passed native and visual collision screening. Full physical results below; {len(passed)}/12 independently verified with new home. Old12 successful asymmetric-home baselines remain archived, and are not counted as new-home success. New results do not establish randomized robustness. No collection/training started.

| Asset | Smooth preparation | Full lift verified | Failure | Hold lift min m | Tilt max deg | Four force fraction |
| --- | --- | --- | --- | --- | --- | --- |
{table}

Original full lift acceptance is retained. Independent verification checks original angular limits for the entire trajectory, actual preparation tracking under the existing0.25rad runtime rejection, mirror error<5mm, original velocity bounds (including finite differences of actual sampled positions), commanded smoothness caps, non-assembly self-contact impulse<=0.001Ns, visual floor clearance, object nonhand collisions, frozen work joints/root, original work torso margins and force/tilt/lift/drift criteria. The additional exploratory0.1rad precise-following goal is recorded separately as precision_tracking_within_0p1_rad; it remains unmet in some20s trials. Smooth limited-speed movement and exact following are distinct observations; tracking lag is disclosed, never treated as zero. Physical samples/videos are SHA-indexed server artifacts; code/results/plans archived here. Source runtime_v1 stopped on invalid USD-prim references after preparation. Runtime_v2 had9.8-10.6deg maximum position-only tracking lag; runtime_v3 added velocity targets; runtime_v4 extends preparation to20s. Only v4-family outputs count toward this revision. The basket precise comparison raises articulation solver64/16 to128/32, retaining original geometry, materials, effort caps and visible-contact thresholds. Earlier incomplete debugging trials were stopped; they are not successful validations or new repeat evidence.

Basket force feedback retains solver64/16 and original collisions/materials/effort limits. It bounds gripper master target offsets to[-0.12,0]rad and opens against excessive force, targeting20N per hand. The prior right-side force exceeded100N at first lift and witness error reached2.622mm; the visible2.5mm threshold is unchanged. The round-bin comparisons use the existing case-style force/tilt balancing, either throughout feedback or only after original preload is established; all original force preload, settling and final acceptance remain unchanged. Tilt balancing uses privileged simulated object orientation. One basket-force launch ran out of GPU memory before producing a completed physical result; it was rerun in a fresh repeat2 folder and the incomplete launch is not repeat evidence.

Overlay prior commonfix/caseyaw dependencies, source and fixtures in one workspace; run the asset entry in ../../ground_lift_training_config.json. The simulator is IsaacSim4.2, 120Hz physics/10Hz sampled evidence and video; no hardware claim.
''')
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
readme=root/'README.md';txt=readme.read_text();marker='<!-- symhome -->'
if marker not in txt:readme.write_text(marker+f'\nLatest symmetric home: {len(passed)}/12 full simulations verified, continuous preparation from shipped default. [Results and scope](iterations/20261120_symhome/README.md). Ideal torso only; no hardware or collection.\n<!-- /symhome -->\n\n'+txt)
out=R/'symhome_review';out.mkdir(exist_ok=True)
chosen=next((r for r in rows if r['asset']=='box_white' and r['smooth_transition_verified']),next(r for r in rows if r['smooth_transition_verified']))
shutil.copy2(chosen['review.mp4']['server_path'],out/'full_sequence.mp4');shutil.copy2((R/chosen['result']).with_name('initial_topview.png'),out/'initial_topview.png');(out/'selected.json').write_text(json.dumps(chosen,indent=2))
print('SYMHOME_PACKAGED',len(rows),len(passed),flush=True)
