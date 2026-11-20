from pathlib import Path
import json
R=Path(__file__).resolve().parent;r=json.loads((R/'directprep_verified_results.json').read_text())[0];m=json.loads((R/'directprep_observed_comparison.json').read_text());d=json.loads((R/'directprep_audit.json').read_text());old=m['old'];new=m['new'];pct=(1-new['actual_total_arm_travel_deg']/old['actual_total_arm_travel_deg'])*100
xml=f'''<h1>2026-11-20｜简化待机到夹取初姿：消除肩部绕转</h1>
<p><b>检查确认有多余的抬手和外展。原因是待机与工作初姿采用了不同的肩部逆运动学解；原来直接插值关节角，虽然每个关节平滑且单调，手的空间路径仍然绕行。</b></p>
<p>本次使用肩部另一组近似等效关节角，保留收肩、双臂下垂的外观，原低位工作初姿和后续抓取目标完全不变。肩部第一关节的目标转动由约176.5°减至3.5°，第三关节由约142.6°减至37.4°。这次不是压缩时长：仍使用20秒连续平滑过渡。</p>
<table><thead><tr><th><p>实际物理仿真轨迹（10Hz采样）</p></th><th><p>之前</p></th><th><p>本次</p></th></tr></thead><tbody>
<tr><td><p>双臂全部关节累计转角</p></td><td><p>{old['actual_total_arm_travel_deg']:.1f}°</p></td><td><p>{new['actual_total_arm_travel_deg']:.1f}°（减少{pct:.1f}%）</p></td></tr>
<tr><td><p>左手路径长度</p></td><td><p>{old['l']['wrist_path_m']*100:.1f}厘米</p></td><td><p>{new['l']['wrist_path_m']*100:.1f}厘米</p></td></tr>
<tr><td><p>左手相对起点最大上抬</p></td><td><p>{old['l']['wrist_upward_excursion_m']*100:.1f}厘米</p></td><td><p>{new['l']['wrist_upward_excursion_m']*100:.1f}厘米</p></td></tr>
<tr><td><p>左肘中心距身体中线最远</p></td><td><p>{old['l']['elbow_abs_y_peak_m']*100:.1f}厘米</p></td><td><p>{new['l']['elbow_abs_y_peak_m']*100:.1f}厘米</p></td></tr>
</tbody></table>
<h2>同速、同相机的过渡对比</h2><p>左：原来；右：简化后。两段视频均保留原20秒过渡时长。</p><figure view-type="Preview"><source path="@./comparison.mp4" name="G2_过渡对比_左原来_右简化后_同速.mp4"/></figure>
<h2>本次待机正面</h2><img path="@./standby_front.png" caption="等效肩部解：继续收肩、双臂自然下垂"/><p>实际臂爪宽度{r['visual_arm_width_m']*100:.2f}厘米，底盘宽度{r['visual_base_width_m']*100:.2f}厘米，左右仍均在底盘正面横向包络内。</p>
<h2>简化过渡衔接完整白箱抓取</h2><figure view-type="Preview"><source path="@./full_sequence.mp4" name="G2_简化过渡_白箱完整抓取.mp4"/></figure>
<p>12种物体场景的准备路径均通过几何筛查；本轮白箱完整物理仿真抓取经独立复核通过。原始关节范围、速度、镜像、固定底盘、自碰撞、工作初姿一致性及原抓取判据均复核。其它物体未全量重跑完整抓取。屈肘和降低身体等到位所需动作仍保留。</p>
<p>自定义仿真姿态，沿用固定底盘和理想腰部驱动；不代表真机验证。未开始收数据或训练。代码分支genie2-rsi，归档research/rsi/g2_sim/iterations/20261120_directprep。</p>'''
(R/'directprep_report.xml').write_text(xml);print('REPORT_READY',pct)
