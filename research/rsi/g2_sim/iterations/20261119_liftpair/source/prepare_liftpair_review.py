from pathlib import Path
import json,shutil,html
R=Path(__file__).resolve().parent;out=R/'liftpair_review';out.mkdir(exist_ok=True)
rows=json.loads((R/'liftpair_verified_results.json').read_text());config=json.loads((R/'liftpair_ground_lift_config.json').read_text());selected={}
for a in ['box_reinforced','case_handheld']:
 passing=[r for r in rows if r['asset']==a and r['verified_success']]
 if passing:
  b=config['validated_simulation_baselines'][a];target=f"{b['harness'].replace('_trial.py','_trials')}/{b['plan_dir']}/{a}/candidate{b['candidate']}_repeat{b['repeat']}/result.json"
  selected[a]=next(r for r in passing if r['result']==target)
 else:selected[a]=max([r for r in rows if r['asset']==a],key=lambda r:(R/r['result']).stat().st_mtime)
names={'box_reinforced':'加固箱','case_handheld':'手提硬箱'}
reasons={'FOUR_FINGER_PRELOAD_NOT_ESTABLISHED':'四指预载未建立','LIFT_HOLD_CRITERIA_NOT_MET':'保持判据未全部满足','NO_SETTLED_FOUR_FINGER_GRASP':'四指承力未稳定','NONHAND_OBJECT_CONTACT':'发生非手部接触','CONTACT_POINT_OFF_VISIBLE_SURFACE':'接触点与可视表面不符'}
new=sum(r['verified_success'] for r in selected.values())
x=['<h1>2026-11-19｜两种箱体继续验证：接触姿态与承力控制</h1>',f"<p><b>本轮新增 {new} 种通过独立复核；当前共 {len(config['validated_assets'])}/12 种抬升候选拥有通过基线。</b>本轮保留 {len(rows)} 次物理试验，失败轨迹完整归档。</p>",
 '<p>躯干保持 [-40°,142°,-87°,0°,0°]，仍只有双臂和夹爪运动。没有改变物体质量、摩擦系数、驱动力矩上限或碰撞资产。</p>',
 '<p>针对内外指负载反复切换，本轮比较连续腕部承力平衡，以及更缓慢的总接触载荷调节。另按几何重新选择接触高度和腕部倾角，并保留抬升时向前避让的路径。部分诊断把有界手臂挤压位移从 12 mm 扩到 20 mm，保留接触力释放及碰撞检查；不是提高电机力矩。</p>',
 '<table><thead><tr><th><p>物体</p></th><th><p>结果</p></th><th><p>保持最低抬高 / 最大倾斜</p></th><th><p>四接触覆盖 / 原因</p></th></tr></thead><tbody>']
for a,r in selected.items():
 state='通过' if r['verified_success'] else '尚未通过'
 metrics=f"{r['min_hold_lift_m']*100:.1f} cm / {r['max_hold_tilt_deg']:.1f}°" if r['min_hold_lift_m'] is not None else '未完成合格保持'
 why=f"{r['four_contact_fraction']*100:.1f}%" if r['four_contact_fraction'] is not None else reasons.get(r['failure'],r['failure'] or '复核通过')
 x.append('<tr>'+''.join('<td><p>'+html.escape(t)+'</p></td>' for t in [names[a]+' '+a,state,metrics,why])+'</tr>')
x+=['</tbody></table>',
 '<p>验收未放宽：抬升至少 15 cm、保持至少 3 秒、倾斜不超过 20°、相对漂移不超过 2 cm、四指承力覆盖至少 90%。保留非手部接触、可视表面、关节锁定与物理时钟检查；成功结果再独立复核整体离地、几何穿透和原始关节余量。</p>']
for a in selected:
 passing=[r for r in rows if r['asset']==a and r['verified_success']]
 x.append(f'<p>{names[a]}：本轮 {len(passing)} 次试验通过独立复核。不同候选或重复试验的完整参数见归档表，不能据此当作随机扰动成功率。</p>')
x+=['<p>仍是 1 kg 假设、固定底盘和非手臂关节、按物体预抓取双臂初姿的仿真。此前十种物体沿用各自冻结基线，不声称已在新控制器上全部回归。尚未开始 policy 训练，也未验证统一双臂 reset、随机初始误差或实机能力。</p>','<h2>本轮视频</h2>']
for a,r in selected.items():
 shutil.copy2(r['review.mp4']['server_path'],out/(a+'.mp4'));state='通过' if r['verified_success'] else '尚未通过'
 x += [f'<h3>{names[a]}：{state}</h3>',f"<p>归档试验：{html.escape(r['result'])}</p>",f'<figure view-type="Preview"><source path="@./{a}.mp4" name="{names[a]}_继续验证_{state}.mp4"/></figure>']
x.append('<p>代码分支 genie2-rsi；本轮归档 research/rsi/g2_sim/iterations/20261119_liftpair。当前训练配置按物体列明 iteration、harness、plan_dir、candidate、repeat；trial_index.json 保留全部成功与失败、原始视频和采样路径及 SHA256。</p>')
(out/'report.xml').write_text('\n'.join(x),encoding='utf-8');shutil.copy2(R/'liftpair_verified_results.json',out/'liftpair_verified_results.json');(out/'selected.json').write_text(json.dumps(selected,indent=2))
print('REVIEW_READY',new,len(rows))
