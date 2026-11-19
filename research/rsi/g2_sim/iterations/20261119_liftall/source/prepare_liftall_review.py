from pathlib import Path
import json,shutil,html
R=Path(__file__).resolve().parent;out=R/'liftall_review';out.mkdir(exist_ok=True)
rows=json.loads((R/'liftall_verified_results.json').read_text());latest={r['asset']:r for r in rows if r['failure']!='HARNESS_ERROR'}
assert len(latest)==11
names={'bin_lidded':'带盖桶','bin_round_lidded':'圆盖桶','box_white':'白箱','box_reinforced':'加固箱','basket_woven':'提手编织篮','basket_rectangular':'方篮','stool_wood_four_leg':'四腿木凳','stool_rectangular':'长方木凳','table_side':'边桌','bin_round_pedal':'圆形脚踏桶','case_handheld':'手提硬箱'}
reasons={'NONHAND_OBJECT_CONTACT':'碰到腕部或底盘，未通过非手部接触检查','NO_SETTLED_FOUR_FINGER_GRASP':'能离地，但四个指面/分支未稳定承力','FOUR_FINGER_PRELOAD_NOT_ESTABLISHED':'抬升前未建立四指预载，未进入正式抬升','OBJECT_TILT_EXCEEDED':'物体倾斜超限','HOLD_OBJECT_TILT_EXCEEDED':'保持期间倾斜超限'}
passed=[r for r in latest.values() if r['verified_success']]
x=['<h1>2026-11-19｜固定新姿态：全部抬升候选验证</h1>',f'<p><b>当前 {len(passed)}/12 种候选通过本轮检查，并非全部可以稳定抬升。</b>11 种完成物理试验；脚踏桶 bin_pedal 的 81 个抓法/站位组合均未找到合格路径，未进入物理试验。两把大椅子仍为推移任务，软包圆凳不属于本轮。</p>',
 '<p>保持当前躯干 [-40°,142°,-87°,0°,0°]，肩高约 82.3 cm，所有躯干关节至少约 10° 原始限位余量。仅双臂和夹爪运动；每种物体使用各自预抓取双臂初姿，并未验证统一双臂 reset 到抓取位置。</p>',
 '<p>统一判据：抬高至少 15 cm 后保持 3 秒，倾斜不超过 20°，相对漂移不超过 2 cm，四接触承力覆盖率至少 90%。本轮增加运行时非手部接触冲量检查；把手/边缘抓取也要求左右内外四分支承力，比历史仅左右合力检查更严格。旧成功结果不能直接视为本轮通过。</p>',
 '<p>成功样本另外复核可视模型整体离地、采样非手部几何穿透、实际躯干关节余量和锁定误差。物体统一假设 1 kg、理想躯干驱动、根部固定，没有绑定物体。每种仅一次确定初始条件试验，尚不代表随机鲁棒性或实机能力；尚未开始训练。</p>',
 '<table><thead><tr><th><p>物体</p></th><th><p>结果</p></th><th><p>保持最低抬高 / 最大倾斜</p></th><th><p>原因</p></th></tr></thead><tbody>']
for a,r in sorted(latest.items()):
 status='通过' if r['verified_success'] else '未通过';metrics=f"{r['min_hold_lift_m']*100:.1f} cm / {r['max_hold_tilt_deg']:.1f}°" if r['min_hold_lift_m'] is not None else '未完成合格保持'
 reason='独立复核通过' if r['verified_success'] else reasons.get(r['failure'],r['failure'] or '独立复核不通过')
 x.append('<tr>'+''.join('<td><p>'+html.escape(s)+'</p></td>' for s in [names[a]+' '+a,status,metrics,reason])+'</tr>')
x+=['<tr><td><p>脚踏桶 bin_pedal</p></td><td><p>路径未通过</p></td><td><p>未进行物理抬升</p></td><td><p>固定躯干逆解不满足或碰到底盘；有限搜索结果，不代表所有抓法均不可能</p></td></tr></tbody></table>',
 '<p>方篮在闭爪时撞到左腕；圆形脚踏桶和加固箱在抬升途中出现底盘接触。编织篮已经抬离地面约 20 cm，但右夹爪内分支没有持续承力，暂不作为四接触成功示范。后续优先调整接触位置、腕部朝向与前后站位，并保留当前躯干姿态，必要时再判断是否需要第二种固定姿态。</p>',
 '<p>圆盖桶旧方案缺少接触中心字段造成首轮程序错误，已按真实指尖表面补齐元数据重跑，关节目标未改；以下仅展示重跑结果。首次程序错误完整保留在代码归档中，不计作物理抓取失败。</p>',
 '<h2>逐物体视频</h2>']
for a,r in sorted(latest.items()):
 shutil.copy2(r['review.mp4']['server_path'],out/(a+'.mp4'))
 status='通过' if r['verified_success'] else '未通过'
 x += [f'<h3>{names[a]}（{a}）：{status}</h3>',f'<figure view-type="Preview"><source path="@./{a}.mp4" name="{names[a]}_{status}.mp4"/></figure>']
x.append('<p>代码与完整失败记录：genie2-rsi 分支 research/rsi/g2_sim/iterations/20261119_liftall。训练入口 ground_lift_training_config.json 仅列本轮复核通过物体。完整采样及原始视频服务器路径、SHA256 见 trial_index.json。</p>')
(out/'report.xml').write_text('\n'.join(x),encoding='utf-8');shutil.copy2(R/'liftall_verified_results.json',out/'liftall_verified_results.json')
print('REVIEW_READY',len(latest),len(passed),flush=True)
