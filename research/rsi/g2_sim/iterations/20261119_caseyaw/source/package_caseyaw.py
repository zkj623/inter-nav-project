from pathlib import Path
import json,hashlib,shutil,ast,datetime
from xml.sax.saxutils import escape
R=Path(__file__).resolve().parent
root=R.parent/'g2-publish-20261118/research/rsi/g2_sim'
I=root/'iterations/20261119_caseyaw';I.mkdir(exist_ok=True)
rows=json.loads((R/'caseyaw_verified_results.json').read_text())
diagnostics=json.loads((R/'caseyaw_diagnostic_index.json').read_text())
old=json.loads((root/'iterations/20261119_commonfix/fixtures/commonfix_ground_lift_config.json').read_text())
cfg=json.loads(json.dumps(old));passing=[r for r in rows if r['verified_success']]
groups={r['result'].split('_repeat')[0]:sum(s['verified_success'] and s['result'].split('_repeat')[0]==r['result'].split('_repeat')[0] for s in rows) for r in rows}
best=min(passing,key=lambda r:(-groups[r['result'].split('_repeat')[0]],r['max_hold_tilt_deg'])) if passing else max(rows,key=lambda r:r.get('max_lift_m') or 0.)
best_data=json.loads((R/best['result']).read_text());best_yaw=best_data['plan']['object_yaw_deg']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
best_harness=next(p.name for p in R.glob('caseyaw*trial.py') if sha(p)==best['source_sha256'])
if passing:
 parts=Path(best['result']).parts
 cfg['validated_assets']=sorted(set(cfg['validated_assets'])|{'case_handheld'});cfg['pending_adaptation_assets']=[]
 cfg['validated_simulation_baselines']['case_handheld']=dict(iteration='20261119_caseyaw',harness=best_harness,plan_dir=parts[1],candidate=int(parts[3].split('_')[0].replace('candidate','')),repeat=int(parts[3].split('_repeat')[1]),object_yaw_deg=best_yaw)
 if best_harness in ('caseyaw_level_trial.py','caseyaw_damped_trial.py'):
  cfg['validated_simulation_baselines']['case_handheld'].update(controller_feedback='privileged measured object orientation; opposite tilt gain0.4 bounded8deg, rate1deg/s',capture_gate=best_data.get('capture_gate'),final_acceptance='original four positive forces >=90%; original lift/tilt/drift/contact limits')
  cfg['validated_simulation_baselines']['case_handheld']['damped_feedback']=best_data.get('damped_feedback')
 cfg['latest_baseline_verified_repeat_counts']['case_handheld']=groups[best['result'].split('_repeat')[0]]
 cfg['per_asset_plan_directory_overrides']['case_handheld']=parts[1]
cfg.update(source_iteration='iterations/20261119_caseyaw',previous_config_archive='iterations/20261119_commonfix/fixtures/commonfix_ground_lift_config.json',status=f"{len(cfg['validated_assets'])}/12 common-start full approach+grasp+lift validations; collecting data awaits review")
(R/'caseyaw_ground_lift_config.json').write_text(json.dumps(cfg,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(cfg,indent=2))
for r in rows:r['result_mtime_utc']=datetime.datetime.fromtimestamp((R/r['result']).stat().st_mtime,datetime.timezone.utc).isoformat()
def copy(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*caseyaw*.py'):ast.parse(p.read_text());copy(p,Path('source')/p.name)
for pat in ['caseyaw*.json','caseyaw*plans/*.json']:
 for p in R.glob(pat):copy(p,Path('fixtures')/p.relative_to(R))
copy(R/'commonstart_planning.json',Path('fixtures/commonstart_planning.json'))
for r in rows+diagnostics:
 p=R/r['result']
 for name in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=p.with_name(name)
  if f.exists():copy(f,Path('results')/f.relative_to(R))
for p in R.glob('caseyaw*.log'):
 if p.stat().st_size<1000000:copy(p,Path('diagnostics')/p.name)
(I/'trial_index.json').write_text(json.dumps(rows,indent=2));(I/'diagnostic_index.json').write_text(json.dumps(diagnostics,indent=2))
table='\n'.join(f"| {r['result']} | {r['verified_success']} | {r['failure']} | {r['min_hold_lift_m']} | {r['max_hold_tilt_deg']} | {r['four_contact_fraction']} |" for r in rows)
doc=f'''# Hard case: contact layout, capture and leveling

This independent workspace contains {len(rows)} fresh regular physical trials, {len(passing)} independently verified passes, plus {len(diagnostics)} diagnostic-only trial(s). Historical commonfix outputs are not new trials. Wall timestamps are retained in trial_index.json. Diagnostic trials are excluded from accepted baselines.

Old candidates already included yaw90. New yaw90 trials reproduce the earlier failure; explicit yaw180 planning found no viable path. Yaw270, narrower opening and steeper approach also failed. Reverse finger-pair orientation moves wrists from about 2.6cm ahead of the object center to about 2.5cm behind it; four common-start paths were planned and three distances physically tested.

The level harness changes capture to bilateral force >12N for three updates, a physical 15mm test lift, and four measured contact manifolds inside existing visual gap/witness tolerances. It adds opposite object-tilt feedback (gain0.4, <=8deg, <=1deg/s), commanding only robot joints. The damped variant reduces load-balance wrist rate from1.5 to0.4deg/s, averages five force samples, targets28N/side, and requires four forces>2N for eight settling updates. This uses privileged simulated object orientation. Final acceptance still requires four positive distal forces >=90%; geometric contact availability alone is not success. No attachment or object pose writes.

| Trial | Verified | Failure | Hold min lift m | Max hold tilt deg | Four force fraction |
| --- | --- | --- | --- | --- | --- |
{table}

Acceptance remains >=15cm lift, >=3s hold, <=20deg tilt, <=2cm relative drift, four distal forces >0.5N in >=90% of hold samples, visual gap<=1.5mm and witness<=2.5mm. Independent review checks visual floor clearance>5mm, nonhand geometry penetration<=1mm, nonhand impulse<=0.001Ns, fixed joints/root, torso margins and actual initial pose. Exact full q_start matches the other eleven objects. Original geometry/materials, 1kg mass assumption and effort limits retained; articulation solver64/16 as in basket baseline. Same-condition repeats are not randomized robustness tests. No hardware validation, policy training or data collection.

Prepare commonfix dependencies, overlay source/fixtures, run the active per-asset baseline in ../../ground_lift_training_config.json. Old successful baselines remain in their original archives. Full samples/videos remain at SHA-indexed server paths; selected review videos are appended to the existing Feishu document.
'''
(I/'README.md').write_text(doc);(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\ndiagnostics/** -text -whitespace\n')
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
f=root/'README.md';s=f.read_text();marker='<!-- caseyaw -->'
if marker not in s:f.write_text(marker+f"\nLatest hard-case iteration: {len(cfg['validated_assets'])}/12 common-start baselines verified. [Contact-layout and leveling tests](iterations/20261119_caseyaw/README.md). Data collection awaits review.\n<!-- /caseyaw -->\n\n"+s)
out=R/'caseyaw_review';out.mkdir(exist_ok=True);shutil.copy2(best['review.mp4']['server_path'],out/'case_handheld.mp4');(out/'selected.json').write_text(json.dumps(best,indent=2))
state='通过' if best['verified_success'] else '尚未通过'
metric=f"保持最低抬高 {best['min_hold_lift_m']*100:.1f} cm，最大倾斜 {best['max_hold_tilt_deg']:.1f}°，四指承力覆盖 {best['four_contact_fraction']*100:.1f}%" if best['min_hold_lift_m'] is not None else f"最大抬高 {(best.get('max_lift_m') or 0)*100:.1f} cm；停止原因 {best['failure']}"
xml=f'''<h1>2026-11-19｜最后一个硬箱：接触布局与倾斜反馈迭代</h1>
<p><b>手提硬箱：{state}；目前 {len(cfg['validated_assets'])}/12 种物体具有统一机器人初姿的完整过程通过基线。</b></p>
<p>本轮独立工作区新增 {len(rows)} 次常规物理测试、{len(diagnostics)} 次诊断测试；常规测试中 {len(passing)} 次通过独立复核。之前复盘已有输出不计入新增测试。旧候选已有90°朝向，此次90°是重新测试；180°未找到可行规划路径，未计物理试验。</p>
<p>尝试了另一侧面、较窄张爪、更陡接近，以及反向手指排列。反向排列把手腕从箱体中心前方约2.6厘米移到后方约2.5厘米，改善可达间隙。进一步加入双侧预紧后15毫米实际试提、四个真实接触点确认和有界反向倾斜反馈。全部从与前11种物体相同的完整初始关节目标开始；原始几何、材质、力矩上限和1千克质量假设保持不变。</p>
<p>最终调平版本把手腕承力平衡速度从1.5°/秒降到0.4°/秒，用5个采样平滑接触力，双侧各以28牛为目标，并在四指均超过2牛、连续8次检查成立后开始保持计时。控制器使用仿真物体姿态反馈，仅控制机器人关节；没有物体位置控制或绑定。此反馈信息属于仿真教师可用信息，尚未训练感知policy。</p>
<p>下方常规测试：箱体朝向 {best_yaw}°，{metric}。{escape(best['result'])}</p>
<p>最终配置同初始条件独立启动重复通过 {groups[best['result'].split('_repeat')[0]]} 次；这不等同于随机位置、姿态、质量或摩擦扰动下的成功率。其它11种物体的通过基线保留，本轮没有重新测试全部12种。</p>
<figure view-type="Preview"><source path="@./case_handheld.mp4" name="手提硬箱_本轮常规测试_{state}.mp4"/></figure>
'''
for i,d in enumerate(diagnostics):
 name=f'case_diagnostic_{i}.mp4';shutil.copy2(d['review.mp4']['server_path'],out/name)
 xml+=f'''<p>诊断视频（未通过，不用于训练基线）：保持最低抬高 {(d.get('min_hold_lift_m') or 0)*100:.1f}厘米，最大倾斜 {(d.get('max_hold_tilt_deg') or 0):.1f}°，四指同时承力覆盖 {(d.get('hold_four_finger_contact_fraction') or 0)*100:.1f}%，四个几何接触可用覆盖 {(d.get('diagnostic_four_available_fraction') or 0)*100:.1f}%。几何接触可用不能替代四指承力和倾斜验收。</p><figure view-type="Preview"><source path="@./{name}" name="硬箱_接触诊断_未通过.mp4"/></figure>'''
xml+='''<p>最终验收不变：抬高至少15厘米、保持至少3秒、倾斜≤20°、相对漂移≤2厘米、四个指尖正承力采样覆盖≥90%，同时满足可视接触、非手部碰撞、整体离地及关节余量检查。未启动批量收数据或policy训练。代码归档：genie2-rsi 分支 research/rsi/g2_sim/iterations/20261119_caseyaw。</p>'''
(out/'report.xml').write_text(xml,encoding='utf-8');print('CASEYAW_PACKAGED',len(rows),len(passing),len(cfg['validated_assets']))
