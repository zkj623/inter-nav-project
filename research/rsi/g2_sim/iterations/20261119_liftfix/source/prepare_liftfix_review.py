from pathlib import Path
import json,shutil,html
R=Path(__file__).resolve().parent;out=R/'liftfix_review';out.mkdir(exist_ok=True)
rows=json.loads((R/'liftfix_verified_results.json').read_text());config=json.loads((R/'liftfix_ground_lift_config.json').read_text());selected={}
for r in sorted(rows,key=lambda r:(R/r['result']).stat().st_mtime):
 if r['asset'] not in selected or not selected[r['asset']]['verified_success']:selected[r['asset']]=r
names={'basket_rectangular':'方篮','basket_woven':'提手编织篮','bin_round_pedal':'圆形脚踏桶','box_reinforced':'加固箱','case_handheld':'手提硬箱','bin_pedal':'脚踏桶'}
reasons={'NONHAND_OBJECT_CONTACT':'发生非手部接触','NO_SETTLED_FOUR_FINGER_GRASP':'抬后四接触未稳定','FOUR_FINGER_PRELOAD_NOT_ESTABLISHED':'抬升前四指预载不足','LIFT_HOLD_CRITERIA_NOT_MET':'保持判据未全部满足','FIXED_TORSO_ARM_IK_UNREACHABLE':'反馈末端逆解未满足误差要求'}
new=sum(r['verified_success'] for r in selected.values())
x=['<h1>2026-11-19｜固定姿态抓取调整：六种物体复测</h1>',f"<p><b>本轮新增 {new} 种通过；加上上一轮 6 种，当前共 {len(config['validated_assets'])}/12 种通过各自归档基线。</b>本轮共保留 {len(rows)} 次物理试验，失败记录完整保留。固定躯干仍为 [-40°,142°,-87°,0°,0°]，未用更深弯腰换取覆盖率。</p>",
 '<p>方篮减浅插入深度，编织篮微调右夹爪对中；箱桶类调整前后站位、接触高度和双手间距。脚踏桶重新按实际几何生成倾斜腕部四指支撑路径。硬箱也搜索了上部特征夹持，但开放夹爪几何检查未通过，继续调整侧面支撑。</p>',
 '<p>通过标准未放宽：抬升至少 15 cm、保持至少 3 秒、倾斜不超过 20°、相对漂移不超过 2 cm、四接触承力覆盖至少 90%。非手部接触、可视表面与固定关节检查全部保留。独立复核确认整体可视模型离地与躯干关节余量。</p>',
 '<table><thead><tr><th><p>本轮对象</p></th><th><p>结果</p></th><th><p>保持最低抬升 / 最大倾斜</p></th><th><p>承力覆盖 / 原因</p></th></tr></thead><tbody>']
for a,r in sorted(selected.items()):
 state='通过' if r['verified_success'] else '尚未通过';metrics=f"{r['min_hold_lift_m']*100:.1f} cm / {r['max_hold_tilt_deg']:.1f}°" if r['min_hold_lift_m'] is not None else '未完成合格保持'
 why=f"四接触 {r['four_contact_fraction']*100:.0f}%" if r['verified_success'] else reasons.get(r['failure'],r['failure'] or '独立复核未通过')
 if not r['verified_success'] and r['four_contact_fraction'] is not None and r['four_contact_fraction']<.9:why=f"四接触覆盖仅 {r['four_contact_fraction']*100:.0f}%，要求至少 90%"
 x.append('<tr>'+''.join('<td><p>'+html.escape(t)+'</p></td>' for t in [names[a]+' '+a,state,metrics,why])+'</tr>')
x+=['</tbody></table>',
 '<p>部分试验使用独立的承力反馈版本：弱指负载不足且强指低于 30 N 时继续有限挤压，保留位移与腕部调整上限；保持阶段继续反馈，避免直接切成开环。夹持类对中反馈限制在 ±8 mm，并先检查四分支预载。不同版本的结果分别归档，不把某个物体通过外推到全部物体。</p>',
 '<p>另外测试了更高预载与更长稳定确认，以及前 10 cm 抬升同步向前避让、随后继续竖直抬升的路径。只驱动手臂，物体仍由真实接触带动；这些实验不自动视为通过，具体以表格与视频结果为准。</p>',
 '<p>手提硬箱还将高预载实验的进入保持门控恢复为原基线（各指大于 1 N、速度小于 15 mm/s、连续四次更新），实际执行保持后仍按原有完整判据判定。加固箱和手提硬箱最后一轮都保持了抬升高度，但四指持续承力仅 60%，没有加入成功示范集。后续重点应是接触点分布与承力均衡，而不是单纯增加总夹持力。</p>',
 '<p>上一轮六种已通过物体：带盖桶、圆盖桶、白箱、四腿木凳、长方木凳、边桌，本轮沿用其冻结基线。新结果仍是统一假设 1 kg、根部和非手臂关节固定、各物体独立预抓取初姿的仿真；没有物体绑定，未验证随机扰动、统一双臂 reset 或实机能力。尚未开始 policy 训练。</p>',
 '<h2>本轮逐物体视频</h2>']
for a,r in sorted(selected.items()):
 shutil.copy2(r['review.mp4']['server_path'],out/(a+'.mp4'));state='通过' if r['verified_success'] else '尚未通过'
 x+=[f'<h3>{names[a]}（{a}）：{state}</h3>',f"<p>归档试验：{html.escape(r['result'])}</p>",f'<figure view-type="Preview"><source path="@./{a}.mp4" name="{names[a]}_调整后_{state}.mp4"/></figure>']
x.append('<p>代码分支 genie2-rsi；本轮归档 research/rsi/g2_sim/iterations/20261119_liftfix。当前 ground_lift_training_config.json 按物体列出已复核基线的 iteration、harness、plan_dir 和 candidate。trial_index.json 包含全部失败、成功记录，以及原始视频/采样路径与 SHA256。</p>')
(out/'report.xml').write_text('\n'.join(x),encoding='utf-8');shutil.copy2(R/'liftfix_verified_results.json',out/'liftfix_verified_results.json');(out/'selected.json').write_text(json.dumps(selected,indent=2))
print('REVIEW_READY',len(selected),'videos',new,'new passes')
