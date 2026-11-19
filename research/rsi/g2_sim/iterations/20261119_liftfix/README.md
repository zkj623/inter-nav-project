# 固定 G2 姿态：六种失败物体的抓取调整

继续使用 [-40,142,-87,0,0] 度躯干姿态及约 10 度最小关节限位余量。本轮针对此前六种未通过候选调整，共保留 27 次物理试验。新增通过 4 种；加上上一轮六种，当前共 10/12 种通过各自归档基线。

## 调整方法

- 方篮减浅夹爪插入深度，避免腕部接触；编织篮调整右夹爪沿闭合轴的对中位置。
- 圆形脚踏桶与加固箱调整物体前后站位和侧面接触高度；加固箱在升高接触区时双手各外移 4 mm，避免预抓取时指面穿透。
- 手提硬箱提高侧面接触区；额外搜索上部特征夹持，但开放夹爪几何检查没有合格候选，保留失败原因。
- 脚踏桶重新基于几何射线生成倾斜腕部四指支撑策略，固定原躯干重新求解完整路径，首次得到可物理执行候选。
- 独立 balance 控制器：弱指承力不足且强指低于 30 N 时继续有限挤压，保持原有 12 mm 上限与腕部平衡约束；保持阶段继续手臂承力反馈。夹持类可在 +/-8 mm 内对中，并要求四分支预载后才抬升。执行过的源码不覆盖，结果绑定源码 SHA256。
- preload 变体进一步把侧面支撑控制目标提高至每指 5 N，开始保持前要求各指 >4 N、物体速度 <5 mm/s 连续八次更新；最终成功判据不变。
- hold 变体针对高预载门控反复等待，保留 5 N 承力目标及主动保持反馈，将进入保持的门控恢复为原基线各指 >1 N、速度 <15 mm/s 连续四次更新。最终 15 cm / 3 秒 / 20 度 / 2 cm / 四接触覆盖 90% 验收未改变，必须实际执行保持才能判成功。
- 箱桶避让路径还测试了前 10 cm 抬升同步向前移出、随后继续竖直抬升的方案。前移量仅用于手臂目标和名义碰撞审计，未控制物体位姿；是否通过仍以运行时接触与保持结果为准。

判据没有放宽：抬高至少 15 cm，保持至少 3 秒，倾斜 <=20 度，相对夹爪漂移 <=2 cm，四接触承力覆盖率 >=90%；非手部接触冲量、可视表面、碰撞和固定关节检查均保留。独立复核要求整体可视模型离地 >5 mm、采样非手部穿透不超过 1 mm、锁定误差 <0.005 rad、根部漂移 <1 mm。

| 物体 | 试验路径 | 独立复核通过 | 失败原因 | 保持最低抬高 m | 四接触覆盖 |
| --- | --- | --- | --- | --- | --- |
| box_reinforced | liftfix_balance_trials/liftfix_box_plans/box_reinforced/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.19199315458536148 | 0.6 |
| box_reinforced | liftfix_balance_trials/liftfix_box_plans/box_reinforced/candidate1_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| bin_pedal | liftfix_balance_trials/liftfix_pedal_plans/bin_pedal/candidate1_repeat1/result.json | True | None | 0.18602734804153442 | 0.9666666666666667 |
| case_handheld | liftfix_balance_trials/liftfix_pedal_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | liftfix_balance_trials/liftfix_pedal_plans/case_handheld/candidate1_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| bin_round_pedal | liftfix_balance_trials/liftfix_plans/bin_round_pedal/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | -1.4901161193847656e-08 | 0.36666666666666664 |
| bin_round_pedal | liftfix_balance_trials/liftfix_plans/bin_round_pedal/candidate1_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | liftfix_balance_trials/liftfix_plans/case_handheld/candidate1_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | liftfix_hold_trials/liftfix_arc_plans/case_handheld/candidate1_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.17926499247550964 | 0.6 |
| bin_round_pedal | liftfix_preload_trials/liftfix_arc_plans/bin_round_pedal/candidate1_repeat1/result.json | True | None | 0.19231650233268738 | 1.0 |
| case_handheld | liftfix_preload_trials/liftfix_arc_plans/case_handheld/candidate1_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| bin_round_pedal | liftfix_preload_trials/liftfix_plans/bin_round_pedal/candidate1_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| box_reinforced | liftfix_trials/liftfix_box_plans/box_reinforced/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.16989728063344955 | 0.7 |
| box_reinforced | liftfix_trials/liftfix_box_plans/box_reinforced/candidate1_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| box_reinforced | liftfix_trials/liftfix_box_plans/box_reinforced/candidate2_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| bin_pedal | liftfix_trials/liftfix_pedal_plans/bin_pedal/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| bin_pedal | liftfix_trials/liftfix_pedal_plans/bin_pedal/candidate1_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| bin_pedal | liftfix_trials/liftfix_pedal_plans/bin_pedal/candidate2_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| bin_pedal | liftfix_trials/liftfix_pedal_plans/bin_pedal/candidate3_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| basket_rectangular | liftfix_trials/liftfix_plans/basket_rectangular/candidate0_repeat1/result.json | True | None | 0.1953253671526909 | 1.0 |
| basket_woven | liftfix_trials/liftfix_plans/basket_woven/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| basket_woven | liftfix_trials/liftfix_plans/basket_woven/candidate1_repeat1/result.json | True | None | 0.1978112757205963 | 1.0 |
| bin_round_pedal | liftfix_trials/liftfix_plans/bin_round_pedal/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 7.450580596923828e-08 | 0.3333333333333333 |
| bin_round_pedal | liftfix_trials/liftfix_plans/bin_round_pedal/candidate1_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.1645953208208084 | 1.0 |
| bin_round_pedal | liftfix_trials/liftfix_plans/bin_round_pedal/candidate2_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.16754883527755737 | 1.0 |
| case_handheld | liftfix_trials/liftfix_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | liftfix_trials/liftfix_plans/case_handheld/candidate1_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |

## 复现和训练边界

先按 ../20261119_liftall/README.md 准备工作区，再覆盖本目录 source/ 和 fixtures/；共享资产保持只读。运行前设置 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1，使用 Isaac Sim 4.2 对应 Python。每个通过物体的 harness、plan_dir、candidate、iteration 见 ../../ground_lift_training_config.json 的 validated_simulation_baselines。旧六种沿用上一轮冻结基线，不声称已在新控制器上重新回归。

全部物体统一假设 1 kg，根部与非手臂关节固定，理想躯干驱动。双臂仍按物体预抓取姿态初始化，未验证统一 reset、随机初始误差或实机能力。没有物体绑定或运行中位姿写入。仅通过独立复核的轨迹可成为后续训练成功示范，尚未开始 policy 训练。

trial_index.json 保存所有成功与失败记录，以及完整采样和原始视频服务器路径、SHA256。关节姿态以当前 q 与物理采样为准，历史候选 envelope 字段保留用于追溯。
