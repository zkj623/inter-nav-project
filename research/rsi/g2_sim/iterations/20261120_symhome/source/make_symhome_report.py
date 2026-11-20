from pathlib import Path
import json
from xml.sax.saxutils import escape
R=Path(__file__).resolve().parent;rows=json.loads((R/'symhome_verified_results.json').read_text());out=R/'symhome_review';chosen_rows=[]
for a in sorted({r['asset'] for r in rows}):
 candidates=[r for r in rows if r['asset']==a]
 chosen_rows.append(next((r for r in candidates if r['verified_success']),next((r for r in candidates if r['result'].startswith('symhome_v4_trials/')),candidates[0])))
ok=[r for r in chosen_rows if r['verified_success']];motion=[r for r in chosen_rows if r['smooth_transition_verified']]
best=next((r for r in rows if r['asset']=='box_white' and r['smooth_transition_verified']),motion[0]);span=best['elbow_lateral_span_m']*100;err=max(r['prepared_error_rad'] for r in motion)*180/3.141592653589793;lag=max(r['preparation_max_tracking_error_rad'] for r in motion)*180/3.141592653589793
names={'basket_rectangular':'方篮','basket_woven':'编织篮','bin_lidded':'带盖桶','bin_pedal':'脚踏桶','bin_round_lidded':'圆带盖桶','bin_round_pedal':'圆脚踏桶','box_reinforced':'加固箱','box_white':'白箱','case_handheld':'手提硬箱','stool_rectangular':'长方凳','stool_wood_four_leg':'四脚木凳','table_side':'边桌'}
xml=f'''<h1>2026-11-20｜对称内收初姿与默认姿态平滑过渡</h1>
<p><b>已将工作初姿改成左右镜像、手肘内收；新增从仓库默认站姿连续运动到工作初姿的过程。新初姿下 {len(ok)}/12 种物体通过完整仿真复核。</b></p>
<p>两肘关节中心横向间距由105.3厘米收至约{span:.1f}厘米；两肩仍约48.7厘米。左右手腕前后位置、高度对齐。腰部工作角度保持[-40°,142°,-87°,0°,0°]，工作阶段仍只控制双臂和夹爪。</p>
<img path="@./initial_topview.png" caption="新版统一初姿俯视图：上方为机器人前方；左右双臂镜像，肘部内收"/>
<p>默认起点取自仓库 scene_flat_g2_op.yaml：双臂第2、4、6关节分别为-45°、-75°、-55°，其余未指定关节为0；底盘使用现有固定根、z=0.04米的仿真设置。采用12秒七阶轨迹，并发送对应的解析速度目标，起止速度、加速度及jerk归零；最大指令速度约37.3% URDF限速。到位稳定后再锁定腰部，随后执行各物体的接近和抓取。初始化以后没有机器人或物体位姿重置。</p>
<p>最终选用的配置中，运动时最大关节跟踪误差{lag:.2f}°，到位最大误差{err:.3f}°。运动仍有滞后，额外尝试的0.1 rad精跟踪目标尚未全部达到；结果保留原运行时0.25 rad偏差上限，并单独披露精跟踪指标。平滑限速过渡不代表每时刻精准跟随指令。同时复核原始关节角范围、实际速度、非装配部位自碰撞及镜像位置误差。12种物体场景的准备和接近路径均通过几何筛查；表中完整抬升结果来自新运行，不沿用旧初姿的通过记录。</p>
<h2>默认站姿到新版初姿</h2><figure view-type="Preview"><source path="@./preparation.mp4" name="G2_默认站姿到对称内收初姿_连续过渡.mp4"/></figure>
<h2>连续过渡后抓取示例</h2><figure view-type="Preview"><source path="@./full_sequence.mp4" name="G2_默认站姿_新版初姿_白箱完整抓取.mp4"/></figure>
<h2>新初姿回归验证</h2><table><thead><tr><th><p>物体</p></th><th><p>平滑过渡</p></th><th><p>完整抬升</p></th><th><p>停止原因</p></th></tr></thead><tbody>'''
for r in rows:
 suffix='（握力反馈）' if 'v4force_' in r['result'] else '（高求解精度）' if 'v4precise_' in r['result'] else '（全程平衡反馈）' if 'v4balanced_' in r['result'] else '（预紧后平衡反馈）' if 'v4staged_' in r['result'] else ''
 xml+=f"<tr><td><p>{names.get(r['asset'],r['asset'])}{suffix}</p></td><td><p>{'通过' if r['smooth_transition_verified'] else '未通过'}</p></td><td><p>{'通过' if r['verified_success'] else '未通过'}</p></td><td><p>{escape(r.get('failure') or '—')}</p></td></tr>"
xml+='''</tbody></table><p>抬升验收仍要求至少15厘米、保持3秒、倾斜≤20°、漂移≤2厘米、四指承力覆盖≥90%，以及可视接触、非手部碰撞、整体离地和关节余量检查。每种物体此次为相同固定初始条件的验证，不代表随机扰动成功率。准备动作在16维双臂/夹爪policy之外；尚未收数据或训练。</p>
<p><b>执行范围：现有固定底盘、理想腰部驱动的仿真。</b>新增静态力矩审查表明，过渡中腰部最大需求约226 Nm，高于当前URDF写入的50 Nm。因此本结果不能作为原始50 Nm驱动或真机可执行的保证。真机执行需要核实驱动参数并单独验证；该限制已记录到配置。</p>
<p>代码分支genie2-rsi，归档research/rsi/g2_sim/iterations/20261120_symhome。此前不对称初姿的12种物体通过记录仍作为历史保留。</p>'''
(out/'report.xml').write_text(xml,encoding='utf-8');print('REPORT_READY',len(ok),len(motion),lag,err)
