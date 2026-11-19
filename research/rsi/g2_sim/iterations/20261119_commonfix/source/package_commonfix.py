from pathlib import Path
import json,shutil,hashlib,ast,html
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_commonfix';I.mkdir(exist_ok=True)
rows=json.loads((R/'commonfix_verified_results.json').read_text());old=json.loads((root/'iterations/20261119_commonstart/fixtures/commonstart_ground_lift_config.json').read_text());cfg=json.loads(json.dumps(old));best={}
for r in sorted(rows,key=lambda r:(r['result'].startswith('commonfix_solver_trials'),(R/r['result']).stat().st_mtime)):
 if r['asset'] not in best or r['verified_success']:best[r['asset']]=r
for a in list(best):
 if not best[a]['verified_success']:
  candidates=[r for r in rows if r['asset']==a and not r.get('diagnostic_only_reason')]
  best[a]=max(candidates,key=lambda r:(r['failure'] not in ['HARNESS_ERROR','NONHAND_OBJECT_CONTACT','CONTACT_POINT_OFF_VISIBLE_SURFACE','LOADED_VISIBLE_SURFACE_GAP'],r.get('max_lift_m') or 0.))
assert set(best)=={'case_handheld','basket_rectangular'}
new=[a for a,r in best.items() if r['verified_success']];cfg['validated_assets']=sorted(set(old['validated_assets'])|set(new));cfg['pending_adaptation_assets']=sorted({'case_handheld','basket_rectangular'}-set(new))
for a in new:
 r=best[a];parts=Path(r['result']).parts;h=next(p.name for p in R.glob('commonfix*trial.py') if p.name.endswith(f'_{a}_trial.py') and hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256'])
 cfg['validated_simulation_baselines'][a]=dict(iteration='20261119_commonfix',harness=h,plan_dir=parts[1],candidate=int(parts[3].split('_')[0].replace('candidate','')),repeat=int(parts[3].split('_repeat')[1]))
cfg.update(source_iteration='iterations/20261119_commonfix',previous_config_archive='iterations/20261119_commonstart/fixtures/commonstart_ground_lift_config.json',status=f"{len(cfg['validated_assets'])}/12 common-start full approach+grasp+lift validations; collecting data awaits user review")
cfg['controller_scope']='Per-asset archived baseline including controller gains and solver iterations; physical geometry/material/actuator caps recorded in protocol; all start from identical full q_start.'
cfg['per_asset_plan_directory_overrides']={a:b['plan_dir'] for a,b in cfg['validated_simulation_baselines'].items()}
for a in new:cfg['latest_baseline_verified_repeat_counts'][a]=sum(r['asset']==a and r['verified_success'] for r in rows)
(R/'commonfix_ground_lift_config.json').write_text(json.dumps(cfg,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(cfg,indent=2))
readme=root/'README.md';body=readme.read_text();marker='<!-- commonfix -->'
if marker not in body:readme.write_text(marker+f"\nLatest common-start validation: {len(cfg['validated_assets'])}/12 passed. [Remaining-object iteration](iterations/20261119_commonfix/README.md); collection awaits review.\n<!-- /commonfix -->\n\n"+body)
def copy(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*commonfix*.py'):ast.parse(p.read_text());copy(p,Path('source')/p.name)
for p in R.glob('commonfix*.json'):copy(p,Path('fixtures')/p.name)
copy(R/'commonstart_planning.json',Path('fixtures/commonstart_planning.json'))
for pat in ['commonfix*plans/*.json']:
 for p in R.glob(pat):copy(p,Path('fixtures')/p.relative_to(R))
for folder in {Path(r['result']).parts[1] for r in rows}:
 for p in (R/folder).glob('*.json'):copy(p,Path('fixtures')/p.relative_to(R))
for r in rows:
 p=R/r['result']
 for name in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=p.with_name(name)
  if f.exists():copy(f,Path('results')/f.relative_to(R))
for p in R.glob('commonfix*.log'):
 if p.stat().st_size<1000000:copy(p,Path('diagnostics')/p.name)
(I/'trial_index.json').write_text(json.dumps(rows,indent=2))
table='\n'.join(f"| {r['asset']} | {r['result']} | {r['verified_success']} | {r['failure']} | {r['min_hold_lift_m']} | {r['four_contact_fraction']} |" for r in rows)
doc=f'''# Remaining two common-start grasps

{len(rows)} physical trials; {len(new)} new objects independently verified; {len(cfg['validated_assets'])}/12 total complete common-start baselines. The full initial q_start, body posture, mass, friction and actuator torque limits are unchanged. Existing ten successes use their archived baselines.

Case geometry candidates compare contact heights, wrist angles and stance distances, including forward clearance during lift. Basket candidates examine rim pinch and external four-finger support. Every candidate is connected to the identical common home through collision-audited joint motion; no teleportation or object pose driving. Runtime witness, collision, tracking and acceptance gates are retained. Load-control targets are controller parameters within original motor limits, not a change to the physical object.

Additional diagnostics compare softer arm gains (kp600/kd80) and gripper gains (kp15/kd2) with unchanged effort caps. Precise fingertip variants replace only the four distal collision meshes in each trial's prepared_scene: max384 hulls, 2M voxels, 0.1% requested error, shrinkwrap. Shared assets remain untouched. The first precision prototype omitted explicit material binding and is diagnostic-only; the accurate variant restores each original ContactMaterial (static1/dynamic0.9), and only material-preserving variants can be accepted. These geometry experiments are not a regression of the previous ten objects.

The solver variant retains original collision meshes/materials and raises articulation solver iterations from 32 position/8 velocity to 64/16; solverfine tests 256/64. They log native mimic residuals during actual simulation. The configuration prefers passing original-geometry solver trials over experimental replacement collision meshes.

The micro-load variant, only after bilateral contact, commands a 2cm forward/1cm upward wrist motion over a slow segment before four-finger refinement. This is a physical load-transfer experiment with the same IK/static/tracking/collision checks, not object pose driving; final lift/hold thresholds remain unchanged.

The transfer revision triggers on first bilateral contact during refinement; the earlier post-close micro-load variant did not trigger if bilateral contact was absent at that instant. The plane revision measures actual simulator distal link transforms and the visual mesh support patches, rather than assuming exact mimic geometry, and uses bounded wrist corrections. All executed variants are individually archived.

The holdangle variant uses gripper master position gains kp80/kd2 (from kp20/kd1), retaining original motor effort caps and target velocity limits. It tests whether load-induced master aperture error is the source of loss of one support finger. This is separately archived from the softer-gain variant.

The transfer8 revision waits for >8N bilateral contact before commanding 4cm forward/1cm up. The guide revision also retains at least 4cm forward displacement in the first two subsequent lift waypoints, so the following lift cannot immediately undo the clearance; its full joint path passes IK, effort, visible and native collision screening before simulation.

The center variant reads actual object x/y during preload, low-pass filters and clamps a shared wrist offset to +/-3cm, then freezes it after capture; it never follows object z or writes object state. This is privileged simulation-state feedback for the demonstration expert. Narrow/wide geometric searches regenerate pad support patches at master angles -0.4/-0.75 rad rather than merely changing the opening in an old trajectory; all approach paths are re-audited.

| Asset | Trial | Verified | Failure | Hold min lift m | Four-contact coverage |
| --- | --- | --- | --- | --- | --- |
'''+table+'''

Acceptance: lift >=15cm, hold >=3s, tilt <=20deg, relative drift <=2cm, four contact fraction >=90%; visible contact surfaces and runtime non-hand contacts checked. Independent review checks full visual floor clearance >5mm, sampled non-hand penetration <=1mm, fixed-body joint errors, original torso margins, root drift, actual initial pose and identical initial targets.

Prepare the preceding commonstart workspace, overlay this source and fixtures; shared assets remain read-only. Per-asset entries in ../../ground_lift_training_config.json specify the iteration/harness/plan/candidate/repeat. trial_index.json records all successes and failures and full video/sample server paths and SHA256. Fixed root/non-arm joints, assumed 1kg objects; no randomized robustness or hardware validation. Batch data collection awaits video review.
'''
(I/'README.md').write_text(doc);(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\ndiagnostics/** -text -whitespace\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
out=R/'commonfix_review';out.mkdir(exist_ok=True);names={'case_handheld':'手提硬箱','basket_rectangular':'方篮'}
x=['<h1>2026-11-19｜统一初姿：方篮与硬箱继续迭代</h1>',f"<p><b>本轮新增 {len(new)} 种通过完整过程复核，当前共 {len(cfg['validated_assets'])}/12 种通过统一初姿的接近、抓取、抬升和保持。</b>本轮保留 {len(rows)} 次物理试验。</p>",'<p>完整初始关节状态与上一轮完全相同。每次实际执行接近路径后才抓取，没有改变物体质量、摩擦或驱动力矩上限。成功标准沿用至少抬高 15 cm、保持 3 秒、倾斜不超过 20°、相对漂移不超过 2 cm、四接触承力覆盖至少 90%，并保留可视接触表面和非手部接触检查。</p>']
for a,r in sorted(best.items()):
 shutil.copy2(r['review.mp4']['server_path'],out/(a+'.mp4'));state='通过' if r['verified_success'] else '尚未通过';metrics=f"保持最低抬高 {r['min_hold_lift_m']*100:.1f} cm，最大保持倾斜 {r['max_hold_tilt_deg']:.1f}°，四接触覆盖 {r['four_contact_fraction']*100:.1f}%" if r['min_hold_lift_m'] is not None else f"本条最大抬高 {(r.get('max_lift_m') or 0)*100:.1f} cm，未完成合格保持；失败原因：{r['failure']}"
 d=json.loads((R/r['result']).read_text());change=d.get('constraint_accuracy',d.get('precision_collision_experiment',d.get('feedback_revision','原控制器配合新几何候选')))
 if d.get('gripper_gain_revision'):change+='; '+d['gripper_gain_revision']
 if 'captured_planar_offset_m' in d:change+='; bounded measured planar centering during preload, frozen offset '+str(d['captured_planar_offset_m'])+' m; no object state writes'
 x += [f'<h2>{names[a]}：{state}</h2>',f'<p>{html.escape(metrics)}</p>',f'<p>本条设置：{html.escape(change)}。本轮该物体通过独立复核的试验共 {sum(s["asset"]==a and s["verified_success"] for s in rows)} 次；同初始条件重复不代表随机鲁棒性。</p>',f'<p>归档：{html.escape(r["result"])}</p>',f'<figure view-type="Preview"><source path="@./{a}.mp4" name="{names[a]}_统一初姿继续迭代_{state}.mp4"/></figure>']
x.append('<p>当前仍为固定底盘和躯干、统一初始双臂姿态、1 kg 假设物体的仿真验证；同初始条件重复不代表随机鲁棒性。尚未开始批量收数据或 policy 训练。代码分支 genie2-rsi，本轮归档 research/rsi/g2_sim/iterations/20261119_commonfix。</p>')
x.insert(2,'<p>对照包含接触点和站位、夹持目标、手臂柔顺性、指尖碰撞精度、求解器精度以及小幅试提。方篮的有效方案保留原碰撞模型，只将求解精度从位置 32/速度 8 次提高到 64/16；原先最大联动误差约 1.9°，有效方案抬升阶段最大约 0.76°。最终判据没有放宽。第一版精细碰撞测试因未显式绑定原材质，只作诊断，正式复核已排除。</p>')
x.insert(3,'<p>若物体仍未通过，下方展示本轮较接近目标的未通过轨迹，便于观察失败过程；全部其他失败和参数均保存在 trial_index.json，不代表该轨迹可以用于成功示范。</p>')
(out/'report.xml').write_text('\n'.join(x),encoding='utf-8');(out/'selected.json').write_text(json.dumps(best,indent=2));print('PACKAGED',len(rows),new,len(cfg['validated_assets']))
