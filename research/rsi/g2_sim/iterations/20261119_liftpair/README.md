# 加固箱与手提硬箱：固定姿态继续验证

本轮针对上一轮两个四接触承力覆盖仅 60% 的物体，完成 12 次物理试验，新增 2 种通过独立复核；当前总计 12/12 种拥有通过基线。

## 观察与调整

上一轮保持阶段高度、倾斜变化较小，但内外指负载反复切换，尤其左内指瞬时卸载。原控制器只在平均弱指载荷低于阈值时改变腕部平衡，可能漏掉这种偏载。本轮做控制与几何两类对照：

- roll 版本连续按内外指归一化载荷差调整腕部，最大角速度系数 0.6 度/秒，角度范围仍为 ±8 度；按真实仿真时间缩放更新量。
- soft 版本在上述基础上把每只夹爪的总接触载荷目标设为 20 N，以最大 0.8 mm/s 的挤压位移变化率缓慢调节，挤压范围仍为 0～12 mm，保留强指超 35 N 时释放规则。这里的载荷是接触力向量模之和，不是单独测量的法向力。
- range 版本在观察到 12 mm 位移补偿饱和后，将有界挤压范围扩到 20 mm；仍保留缓慢调节、35 N 释放、原驱动力矩上限和全部碰撞检查。该变化是手臂位置补偿范围，不是提高电机力矩上限。
- 重新按几何生成候选，尝试 35 度腕部倾角与更高侧面接触区，并验证前 10 cm 抬升同时前移 6 cm 的完整手臂路径。物体位姿没有被驱动。

躯干仍为 [-40,142,-87,0,0] 度，约 10 度最小原始限位余量。物体质量 1 kg、摩擦系数、驱动力矩上限、碰撞资产和最终通过标准均未改变。

## 全部物理结果

| 物体 | 试验 | 独立复核通过 | 失败原因 | 保持最低抬高 m | 最大倾斜 deg | 四接触覆盖 |
| --- | --- | --- | --- | --- | --- | --- |
| case_handheld | liftpair_range_trials/liftfix_arc_plans/case_handheld/candidate1_repeat1/result.json | True | None | 0.18129873275756836 | 17.12474585676882 | 0.9666666666666667 |
| case_handheld | liftpair_range_trials/liftfix_arc_plans/case_handheld/candidate1_repeat2/result.json | True | None | 0.18129873275756836 | 17.12474585676882 | 0.9666666666666667 |
| box_reinforced | liftpair_range_trials/liftfix_box_plans/box_reinforced/candidate0_repeat1/result.json | False | CONTACT_POINT_OFF_VISIBLE_SURFACE | None | None | None |
| case_handheld | liftpair_range_trials/liftpair_geometry_plans/case_handheld/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.17915888130664825 | 13.26387075805885 | 0.8333333333333334 |
| case_handheld | liftpair_range_trials/liftpair_higharc_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None | None |
| case_handheld | liftpair_roll_trials/liftfix_arc_plans/case_handheld/candidate1_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| box_reinforced | liftpair_roll_trials/liftfix_box_plans/box_reinforced/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| case_handheld | liftpair_soft_trials/liftfix_arc_plans/case_handheld/candidate1_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| box_reinforced | liftpair_soft_trials/liftfix_box_plans/box_reinforced/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| box_reinforced | liftpair_soft_trials/liftpair_geometry_plans/box_reinforced/candidate2_repeat1/result.json | True | None | 0.18652324378490448 | 12.603425587224587 | 1.0 |
| box_reinforced | liftpair_soft_trials/liftpair_geometry_plans/box_reinforced/candidate2_repeat2/result.json | True | None | 0.18652324378490448 | 12.603425587224587 | 1.0 |
| case_handheld | liftpair_soft_trials/liftpair_geometry_plans/case_handheld/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.17915888130664825 | 13.26387075805885 | 0.8333333333333334 |

通过仍要求至少抬升 15 cm、保持至少 3 秒、倾斜不超过 20 度、相对漂移不超过 2 cm、四个末端指节承力覆盖至少 90%。运行时非手部接触冲量、可视表面、碰撞、关节锁定与物理时钟检查保留；成功轨迹再独立复核整体离地、非手部几何穿透、根部漂移、原始关节余量。

当前配置优先选择同一控制器/计划/候选重复通过的基线，再比较保持倾斜；同初始条件重复不等同于随机鲁棒性测试。

## 复现和范围

按 ../20261119_liftfix/README.md 准备工作区，覆盖本目录 source/ 与 fixtures/。共享资产只读。OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1；使用 Isaac Sim 4.2 对应 Python。每个通过物体的 iteration、harness、plan_dir、candidate、repeat 见 ../../ground_lift_training_config.json。此前十种物体沿用各自已归档基线，不声称在本轮控制器上全部回归。

根部和非手臂关节固定，双臂按物体预抓取姿态初始化。确定初始条件验证不等同于随机鲁棒性或实机验证，尚未开始 policy 训练。失败记录不进入成功示范集。trial_index.json 保存全部成功与失败记录，以及完整视频、采样的服务器路径与 SHA256。
