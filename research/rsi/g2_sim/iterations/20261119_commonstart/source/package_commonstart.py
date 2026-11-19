from pathlib import Path
import json,shutil,ast,hashlib,html
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_commonstart';I.mkdir(exist_ok=True)
rows=json.loads((R/'commonstart_verified_results.json').read_text());planning=json.loads((R/'commonstart_planning.json').read_text());best={}
for r in sorted(rows,key=lambda r:(R/r['result']).stat().st_mtime):
 if r['asset'] not in best or r['verified_success']:best[r['asset']]=r
passed=sorted(a for a,r in best.items() if r['verified_success']);previous=json.loads((R/'ground_lift_training_config.json').read_text());config=json.loads(json.dumps(previous))
config['previous_config_archive']='iterations/20261119_liftpair/fixtures/liftpair_ground_lift_config.json';config['source_iteration']='iterations/20261119_commonstart'
config['common_initial_joint_positions_rad']=planning['home'];config['initialization_scope']='one identical full joint target for all objects; executed common-start arm/gripper approach before per-object grasp'
config['validated_assets']=passed;config['pending_adaptation_assets']=sorted(set(previous['validated_assets'])-set(passed));config['validated_simulation_baselines']={}
for a in passed:
 r=best[a];parts=Path(r['result']).parts
 harness=next(p.name for p in R.glob('*commonstart*trial.py') if p.name.endswith(f'_{a}_trial.py') and hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256'])
 config['validated_simulation_baselines'][a]=dict(iteration='20261119_commonstart',harness=harness,plan_dir=parts[1],candidate=int(parts[3].split('_')[0].replace('candidate','')),repeat=1)
config['status']=f'{len(passed)}/12 complete common-start approach+grasp+lift validations; collecting training data awaits user video review'
config['plan_directory']='commonstart_plans';config['per_asset_plan_directory_overrides']={a:b['plan_dir'] for a,b in config['validated_simulation_baselines'].items()};config['latest_baseline_verified_repeat_counts']={a:1 for a in passed}
config['validation_scope']='One common reset target including arms/grippers; object-specific collision-audited approach then lift. Fixed root/non-arm joints, assumed mass 1kg. No random robustness or hardware validation.'
config['controller_scope']='Per-object previously validated lift controller, extended with common initial state and joint approach; results revalidated over full physical trajectory.'
(R/'commonstart_ground_lift_config.json').write_text(json.dumps(config,indent=2))
def copy(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*commonstart*.py'):ast.parse(p.read_text());copy(p,Path('source')/p.name)
for pat in ['commonstart*.json','commonstart*plans/*.json']:
 for p in R.glob(pat):copy(p,Path('fixtures')/p.relative_to(R))
for r in rows:
 p=R/r['result']
 for name in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=p.with_name(name)
  if f.exists():copy(f,Path('results')/f.relative_to(R))
for p in R.glob('commonstart*.log'):
 if p.stat().st_size<1000000:copy(p,Path('diagnostics')/p.name)
(I/'trial_index.json').write_text(json.dumps(rows,indent=2))
doc=f'# Unified initial arm and gripper pose\n\n{len(passed)}/12 objects pass the full common-start approach, grasp and lift trial. Full target q_start is identical across plans, including 14 arm and two gripper master joints. Body/root fixed as before.\n\n'
doc+='Initial measured error is checked after 1.5 seconds of settling (<0.05 rad); collision paths sampled with <=0.06 rad increments; all physical collision/contact gates and lift acceptance retained. Previous successful pregrasp lifts are historical and do not substitute for these complete trials.\n\n'
doc+='| Asset | Verified | Failure | Actual initial max error rad |\n| --- | --- | --- | --- |\n'+'\n'.join(f"| {a} | {r['verified_success']} | {r['failure']} | {r['initial_actual_max_error_rad']} |" for a,r in sorted(best.items()))
doc+='\n\nPrepare the workspace using ../20261119_liftpair/README.md, then overlay source and fixtures. Use each per-asset harness in config. Full videos and samples with SHA256 are indexed in trial_index.json. Mass 1kg assumed; fixed root/non-arm joints; random robustness/hardware not tested. Training collection awaits user review.\n'
(I/'README.md').write_text(doc);(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\ndiagnostics/** -text -whitespace\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
if len(best)==12:
 (root/'ground_lift_training_config.json').write_text(json.dumps(config,indent=2))
 f=root/'README.md';s=f.read_text();marker='<!-- commonstart -->'
 if marker not in s:f.write_text(marker+f'\nCommon full initial joint pose and executed approach: {len(passed)}/12 passed [validation](iterations/20261119_commonstart/README.md). Earlier pregrasp results remain historical. Training collection awaits video review.\n<!-- /commonstart -->\n\n'+s)
out=R/'commonstart_review';out.mkdir(exist_ok=True);names={'bin_lidded':'带盖桶','bin_round_lidded':'圆盖桶','bin_pedal':'脚踏桶','bin_round_pedal':'圆形脚踏桶','box_white':'白箱','box_reinforced':'加固箱','case_handheld':'手提硬箱','basket_woven':'提手编织篮','basket_rectangular':'方篮','stool_rectangular':'长方木凳','stool_wood_four_leg':'四腿木凳','table_side':'边桌'}
x=['<h1>2026-11-19｜统一初始状态：完整接近、抓取与抬升复测</h1>',f'<p><b>本轮 {len(passed)}/12 种物体通过完整过程验证。</b>所有物体使用完全相同的双臂、夹爪和躯干初始目标。先稳定初始状态，再实际执行接近路径，到达各自预抓取位置，随后闭爪、抬升并保持。</p>','<p>此前视频仅统一躯干，本节补齐统一双臂初姿与接近过程。初始实际关节角在稳定后复核，最大误差要求小于 0.05 rad；接近路径包含原生/可视自碰撞及物体碰撞检查，物理执行继续检查接触与跟踪误差。</p>','<p>最终验收沿用抬高至少 15 cm、保持 3 秒、倾斜不超过 20°、相对漂移不超过 2 cm、四接触承力覆盖至少 90%，以及可视接触表面、非手部接触和固定关节检查。物体质量统一假设 1 kg，根部及非手臂关节固定。尚未测试随机扰动，也未开始批量收数据，等待本轮视频审阅。</p>']
x.insert(2,'<img path="@./common_start.png" caption="所有物体共用的机器人初始姿态（示例画面中的物体为四腿木凳）"/>')
x.insert(1,'<p>统一初始姿态与接近路径已为全部 12 种物体补齐并实际测试；完整抬升成功集需以本节结果为准。方篮的接触表面与反馈逆解问题、硬箱的四指预载问题仍需调整；此前 12/12 是从各自预抓取状态开始的历史验证，不能替代统一初姿的结果。</p>')
for a,r in sorted(best.items()):
 shutil.copy2(r['review.mp4']['server_path'],out/(a+'.mp4'));x += [f'<h2>{names[a]}（{a}）</h2>',f"<p>完整过程：{'通过' if r['verified_success'] else '未通过'}；初始实际最大关节误差 {r['initial_actual_max_error_rad']:.5f} rad；失败原因 {html.escape(str(r['failure']))}。</p>",f'<figure view-type="Preview"><source path="@./{a}.mp4" name="{names[a]}_统一初始状态.mp4"/></figure>']
(out/'report.xml').write_text('\n'.join(x),encoding='utf-8');(out/'selected.json').write_text(json.dumps(best,indent=2));print('PACKAGED',len(rows),'trials',len(passed),'passed')
