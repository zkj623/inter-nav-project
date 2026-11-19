# 固定姿态全物体抬升验证（2026-11-19）

躯干仍为 [-40,142,-87,0,0] 度，肩高约 0.823 m；全部躯干关节保留至少约 10 度原始 URDF 限位余量。仅双臂和夹爪运动，根部固定。不是统一双臂初始姿态：每个物体从各自预抓取姿态开始。

本轮 12 个刚体抬升候选，11 个完成物理试验；脚踏桶 bin_pedal 在 81 个已有抓法/站位组合中没有合格路径（固定躯干逆解或底盘碰撞），未运行物理抬升。此有限搜索不证明所有抓法都不可能。两把大椅子保留推移任务，软包圆凳 stool_small 排除。

| 物体 | 复核结果 | 失败原因 | 保持最低抬高 m | 保持最大倾斜 deg |
| --- | --- | --- | --- | --- |
| basket_rectangular | FAIL | NONHAND_OBJECT_CONTACT | None | None |
| basket_woven | FAIL | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| bin_lidded | PASS |  | 0.17969746887683868 | 2.061663958392739 |
| bin_round_lidded | PASS |  | 0.19130723178386688 | 9.509916304322653 |
| bin_round_pedal | FAIL | NONHAND_OBJECT_CONTACT | None | None |
| box_reinforced | FAIL | NONHAND_OBJECT_CONTACT | None | None |
| box_white | PASS |  | 0.18168890476226807 | 5.5965657506983435 |
| case_handheld | FAIL | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| stool_rectangular | PASS |  | 0.19832982122898102 | 0.5348927669351157 |
| stool_wood_four_leg | PASS |  | 0.2028426229953766 | 0.5737574126233527 |
| table_side | PASS |  | 0.20094236731529236 | 1.5157116337933287 |
| bin_pedal | PLAN_FAIL | no admissible path in 81 attempts | — | — |

## 检查范围

至少抬高 15 cm，保持 3 秒，倾斜不超过 20 度，相对夹爪漂移不超过 2 cm，四接触承力覆盖率至少 90%。侧面张爪支撑要求四个末端指节；把手/边缘夹持及下托要求左右内外四分支。把手夹持的四分支要求比历史仅左右双侧合力检查更严格，不直接混用历史成功标签。

沿用可视表面接触检查及自碰撞检查；本轮增加运行时非手部接触冲量检查，并独立复核成功试验的采样几何穿透、原始关节余量、固定关节误差、根部漂移以及保持期间整个可视模型离地间隙。采样 10 Hz，物理步长 1/120 秒，采样峰值不等同于全物理步峰值。

四腿木凳、长方木凳、边桌、方篮统一使用已有修正表面碰撞资产；其他使用原始物体碰撞资产。本次没有修改共享资产。圆盖桶首轮因旧候选缺少 contact_centers_local 报 HARNESS_ERROR；根据 q_close 时真实指尖朝内表面补齐元数据后重跑，关节目标未改变，首轮错误保留但不计为物理失败。

统一物体质量假设 1 kg，理想躯干驱动、原手臂/夹爪力矩上限。没有物体绑定或运行中物体位姿写入。每个通过物体仅有一次确定初始条件试验，不等同于随机鲁棒性、真实质量或实机验证。尚未训练 policy；失败轨迹不作为成功示范。

## 复现

按 ../20261119_marginpose/README.md 准备工作区，再覆盖本目录 source/ 和 fixtures/。共享资产只读。运行前设置 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1，使用 Isaac Sim 4.2 对应 Python。

具体通过入口见 ../../ground_lift_training_config.json 的 validated_simulation_baselines。例如：

```bash
python liftall_general_trial.py --asset bin_lidded --plan-dir liftall_validated_plans --gpu 0 --repeat 2
```

长方木凳使用 liftall_under_trial.py；圆盖桶使用 liftall_metadata_plans。trial_index.json 记录完整采样和视频服务器路径及 SHA256，结果/协议/接触事件已归档。历史 envelope 字段沿用候选来源，当前姿态以重求后的 q 值和实际采样为准。
