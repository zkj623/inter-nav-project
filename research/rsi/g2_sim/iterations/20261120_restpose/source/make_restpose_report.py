from pathlib import Path
import json
R=Path(__file__).resolve().parent;rows=json.loads((R/'restpose_verified_results.json').read_text());pose=next(r for r in rows if r['workflow']=='standby_hold_and_preparation');full=[r for r in rows if r['workflow']=='standby_preparation_and_grasp'];ok=[r['asset'] for r in full if r['verified_success']];out=R/'restpose_review'
xml=f'''<h1>2026-11-20｜双臂下放待机起点：自定义仿真姿态</h1>
<p><b>已把首个待机起点换成身体直立、双臂自然下放的左右镜像姿态。它是自定义仿真待机姿态，不是官方出厂默认姿态。现有低位工作初姿保持不变。</b></p>
<img path="@./standby_oblique.png" caption="新的自定义待机姿态：双手在身体两侧下放"/>
<img path="@./standby_front.png" caption="正面检查：双臂和手腕左右镜像；肘部留出弯曲和身体间隙"/>
<p>独立复核通过：待机保持、20秒连续进入低位工作初姿并保持。待机臂段/手腕位置镜像误差约{pose['standby_arm_mirror_error_m']*1000:.2f}毫米，手腕姿态镜像误差约{pose['standby_end_rotation_mirror_error_rad']*180/3.141592653589793:.3f}°；待机最大关节误差约{pose['standby_max_tracking_error_rad']*180/3.141592653589793:.3f}°，过渡到位误差约{pose['prepared_max_tracking_error_rad']*180/3.141592653589793:.3f}°。检查包含原始关节角范围、实际运动速度、自碰撞及工作姿态一致性。</p>
<p>镜像校验覆盖手腕最终坐标系的位置和朝向。修正了初版只检查臂段原点所漏掉的第6关节符号映射；原初版约3厘米的手腕偏差样例不计为通过结果。</p>
<h2>新待机姿态到低位工作初姿</h2><figure view-type="Preview"><source path="@./standby_transition.mp4" name="G2_双臂下放待机_20秒连续过渡_低位保持.mp4"/></figure>
<p>七阶位置曲线配合解析速度目标，起止指令速度/加速度/jerk为0；初始化以后没有机器人或物体位姿重置。12种物体场景的待机、准备及接近路径均通过几何筛查。本次物理全流程仅复测白箱和硬箱，{len(ok)}/2通过；其它物体没有自动沿用旧起点结果。</p>
<h2>新起点衔接抓取示例</h2>'''
for r in full:
 name='白箱' if r['asset']=='box_white' else '手提硬箱';state='通过' if r['verified_success'] else '未通过'
 xml+=f'<h3>{name}：{state}</h3><figure view-type="Preview"><source path="@./{r["asset"]}.mp4" name="G2_双臂下放待机_{name}_完整过程_{state}.mp4"/></figure>'
xml+='''<p>新的待机配置位于research/rsi/g2_sim/standby_preparation_config.json。原低位工作姿态、16维双臂/夹爪动作空间和历史11/12抬升验证保留其原始归档；新起点的全流程复测范围单独列出。没有开始收数据或训练。</p>
<p>范围仍是固定底盘、理想腰部驱动的仿真。进入原低位工作姿态的腰部原始50 Nm力矩上限问题仍未解决；本轮不作真机可执行保证。</p>'''
(out/'report.xml').write_text(xml,encoding='utf-8');print('RESTPOSE_REPORT_READY',ok)
