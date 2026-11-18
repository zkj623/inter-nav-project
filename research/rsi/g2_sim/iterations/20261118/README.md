# 2026-11-18：G2 接触与抬升迭代

本轮新增 4 种同初态 3/3 通过的物体；加上旧快照 7 种，共 11/15 种有重复通过记录。旧 7 种并未全部按新时间步重测，不能称为同一修订版控制器的 11/15 成功率。提手编织篮在新步进下通过一次回归。其余 4 种仍未完整通过。

## 新增可复现基线

| 物体 | 控制版本 / 候选 | 同初态重复 | 最低保持抬升 cm | 最大倾斜 ° | 相对漂移 mm | 四指接触 |
| --- | --- | --- | --- | --- | --- | --- |
| `box_white` | `deadbandclock` / 0 | 3/3 | 18.21 | 4.86 | 0.34 | 100.0% |
| `box_reinforced` | `fixedclock` / 0 | 3/3 | 17.52 | 12.28 | 1.81 | 93.3% |
| `bin_round_pedal` | `loadclock` / 3 | 3/3 | 18.84 | 12.20 | 2.35 | 100.0% |
| `case_handheld` | `deadbandclock` / 0 | 3/3 | 16.12 | 11.68 | 0.21 | 100.0% |

四项均包含下降到低位、接近物体、建立四指支撑、原地抬升及保持。硬箱采用双手侧夹，不是夹提手。重复初态相同、结果确定性较强；不是随机扰动评估。

另用每个保持采样的物体位姿变换全部可视网格顶点，复核最低点离地间隙：白箱 ≥16.95cm、加固箱 ≥14.36cm、圆形脚踏桶 ≥16.71cm、硬箱 ≥12.33cm。`ground_clearance_check.json` 保存逐次结果；不是只看物体原点升高就认定全部离地。

## 时间步错误及接触控制修正

旧 World 配置为 physics_dt=1/120、rendering_dt=1/30，但循环混用 step(render=True/False)。实测每 4 次控制循环推进 7 个物理步，渲染调用推进 4 步。旧时间标签少计实际物理时间。`timebase_check.json` 保存最小实测结果。

修订版每次显式 step(render=False)，每 12 次另行 render()，运行时核对步数和时间差。物理 120Hz、视频 10fps。运动时长乘 1.75 以保留原物理速度；保持阶段为真实 3 秒。历史记录和旧源码保留，不事后改分。

四指策略增加连续接触预载、逐侧有界压紧、弱指接触时微调腕部、连续目标插值、接触稳定后冻结保持目标。白箱/硬箱采用 deadbandclock，加固箱采用 fixedclock，圆形脚踏桶采用 loadclock 的有界载荷预紧。不存在一个已证明适用于所有物体的统一控制器。

验收未降低：保持最低抬升 0.15m、3 秒、倾斜 ≤20°、相对 TCP 漂移 ≤0.02m、双侧接触 ≥90%；四指策略要求四个末端指面各 >0.5N 的采样比例 ≥90%。可视间隙 ≤1.5mm、接触见证误差 ≤2.5mm；无物体焊接、无物体位姿跟随、无底盘固定。

## 未通过与数据使用限制

- `stool_small`：增加夹持预载后能保持约 20cm、四指接触 100%，但倾斜约 39°。降低腕部倾角的搜索经过 1534 次 IK 调用仍无可达且自碰撞通过的路径。此外，该坐墩侧面软垫下是否有刚性承力结构未核实，不能将其侧夹数据直接标为真实刚体接触策略。
- `stool_rectangular`：薄曲面木凳的夹指载荷不平衡。forcefeature4 延长闭合后四指均建立约 8～15N 接触力，但抬升时仍滑脱。自适应闭合与不同接触方案的失败均保留。
- `chair_wheeled`：forcefeature3 低位双手候选能接触，但物体原点最多升高约 3.23cm，抬升时末端指面接触丢失；不代表所有脚轮离地。预置低位试验不包含从站立下降。中轴方案被实际物体碰撞检查拒绝。
- `chair_dining`：增加躯干 roll/yaw 搜索并分散候选高度后，找到一条通过几何审计的单腿抓取路径；物体原点升高约 7.82cm 时触发接触表面误差检查，不能视为完整离地。站立下降路径尚未通过。

追加试验：forcefeature5 在抬升/保持期间也执行有界夹持力反馈，木凳仍滑脱；办公椅最高原点抬升改善到约 6.70cm，但随后滑脱，保持最低仅约 2.51cm。levelclock 在圆凳离地后缓慢调整双腕，最高约 18.84cm，但最终仍倾斜约 34.6°，未建立稳定四指保持。餐椅五躯干自由度的双腿规划，正面 514 次、另外三个朝向 586 次 IK 调用，均未找到通过路径。这些是已搜索配置内的失败，不是对机器人不可抓取的数学证明。

餐椅 forcefeature5 单腿试验闭合至设定上限后，较弱指面约 7.8～7.9N，未持续达到该实验设置的 8N 预载目标，因此在抬升前停止。8N 是控制预载设置，不是最终 15cm/3 秒验收阈值；此结果不证明降低预载后必然不能抬升，也不能计作抓取成功。

接触点复查显示木凳接触落在曲面与边缘，办公椅落在斜向辐条；因此下一步需要同时优化真实指面、接触法向、抗重力/抗倾覆力矩和进入路径，不能只增加法向夹持力。`diagnose_pinch.py` 可从采样重建接触点；`gripper_center_check.json` 是随开合角变化的末端网格中心校准，不等同于真实承力点。

以上状态以 `result_index.json` 逐条结果为准，失败样本不能作为成功模仿数据。程序异常/取消不计有效物理试验。未生成神经网络权重或训练模型。

统一为 1kg 刚体、理想躯干驱动和原手臂/夹爪力矩限制；尚未验证真实质量、载物起身、行走、扰动鲁棒性及实机。资产许可与模型来源沿用上级 README。

## 证据与复现

本轮收录 53 次已完成记录，其中 51 次有效物理试验（36 次使用固定时间步），2 次程序异常/取消。全部结果保留源码 SHA256，成功记录校验通过。大型视频、逐帧采样和 USD 保留在外部实验目录；`result_index.json` 给出代表视频服务器路径。

`source/` 为本轮原样源码，`plans/` 为新增规划；旧规划仍位于上级 fixtures。`baselines.json` 只列三次通过的新基线。失败控制版本为研究证据，不是推荐入口。

本轮 balance 试验曾与旧快照脚本重名，归档时仅将文件名加上 `_20261118`，内容字节和执行 SHA256 不变；`source_names.json` 及每条结果的 `source_file` 给出映射。旧快照保持原样，复现准备不覆盖旧脚本。原始诊断字段的少量遗留说明见 `metadata_notes.json`。

```bash
ITER=research/rsi/g2_sim/iterations/20261118
WORK=/data20t/embodied/zkj/inter-nav-results/g2-replay-20261118-new
ASSETS=/data20t/embodied/zkj/inter-nav-results/g2-migration-20261111
PY=/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python
python3 "$ITER/replay.py" verify
python3 "$ITER/replay.py" prepare --workspace "$WORK" --assets-from "$ASSETS" --collision-deps /data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps
python3 research/rsi/g2_sim/workspace.py check --workspace "$WORK" --python "$PY" --verify-assets
python3 "$ITER/replay.py" run --workspace "$WORK" --python "$PY" --asset box_white --gpu 4 --repeat 1
```

WORK 必须全新；重复运行使用未占用 repeat。资产链接只读使用，不在该工作区运行模型生成。运行仍依赖实验室的 Isaac Sim/外部资产与部分历史绝对路径。

代码只发布至 `genie2-rsi`。R1 分支保留 `bb15d6a`。

[飞书审阅：本轮结论与视频](https://ocnprlp82ms4.feishu.cn/docx/LLihduA7VoWO8ExKS8dckhw9ncb#doxcnrWpDWkqwo9rjQmsQ5s5Bbb)
