from pathlib import Path
import json
R=Path(__file__).resolve().parent;r=json.loads((R/'narrowrest_verified_results.json').read_text())[0];out=R/'narrowrest_review'
xml=f'''<h1>2026-11-20｜收肩待机：双臂外轮廓缩进底盘</h1>
<p><b>已进一步减小待机肩部外展，并同步微调前臂和腕部。按真实可视网格及仿真关节状态测量，双臂与夹爪整体横向宽度约{r['visual_arm_width_m']*100:.2f}厘米，底盘整体约{r['visual_base_width_m']*100:.2f}厘米，左右两侧均已缩进底盘横向包络。</b></p>
<table><thead><tr><th><p>项目</p></th><th><p>宽度</p></th></tr></thead><tbody><tr><td><p>上一版待机双臂/夹爪</p></td><td><p>约88.9厘米</p></td></tr><tr><td><p>本版待机双臂/夹爪（实测）</p></td><td><p>{r['visual_arm_width_m']*100:.2f}厘米</p></td></tr><tr><td><p>底盘及轮组外轮廓</p></td><td><p>{r['visual_base_width_m']*100:.2f}厘米</p></td></tr></tbody></table>
<h2>相同相机下的正面对比</h2><p>上一版：</p><img path="@./previous_front.png" caption="上一版：肩部外展和腕部壳体使双臂轮廓超出底盘"/><p>收肩后：</p><img path="@./standby_front.png" caption="收肩后：双臂保持下放，整体外轮廓已缩进底盘"/>
<img path="@./standby_oblique.png" caption="侧前方检查：手臂与躯干仍有间隙"/>
<p>左右缩进余量分别约{r['left_inset_m']*1000:.1f}和{r['right_inset_m']*1000:.1f}毫米；按原装配排除规则，非连接部位手臂与躯干的最近可视间隙约{r['nonadjacent_arm_body_visual_gap_m']*100:.2f}厘米。待机姿态未检测到自碰撞，原始关节角范围、镜像位置/朝向、实际速度和固定底盘检查均通过。宽度约束针对待机状态，工作与抓取过程仍可展开手臂。</p>
<h2>收肩待机到低位工作初姿</h2><figure view-type="Preview"><source path="@./preparation.mp4" name="G2_收肩待机_20秒连续进入低位工作初姿.mp4"/></figure>
<p>原低位工作初姿及16维动作空间保持不变，20秒位置/速度连续过渡通过复核。12种物体场景的待机、准备和接近路径均通过几何筛查。</p>
<h2>衔接白箱抓取的完整示例</h2><figure view-type="Preview"><source path="@./full_sequence.mp4" name="G2_收肩待机_低位初姿_白箱完整抓取_通过.mp4"/></figure>
<p>本次白箱完整流程通过：保持最低抬高{r['min_hold_lift_m']*100:.2f}厘米，最大倾斜{r['max_hold_tilt_deg']:.2f}°，四指承力覆盖{r['four_contact_fraction']*100:.0f}%。其它物体本轮未全量重跑，旧起点的验证结果保留原始归档，不自动视为新起点已通过。</p>
<p>仍为自定义仿真待机姿态，不是官方出厂姿态。固定底盘、理想腰部驱动及原低位姿态力矩限制问题沿用前述范围，未验证真机执行；尚未收数据或训练。代码分支genie2-rsi，归档research/rsi/g2_sim/iterations/20261120_narrowrest。</p>'''
(out/'report.xml').write_text(xml,encoding='utf-8');print('NARROWREST_REPORT_READY')
