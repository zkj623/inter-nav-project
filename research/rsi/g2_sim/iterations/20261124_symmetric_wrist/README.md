# 对称腕部低位初姿：2026-11-24

右腕 J6 从 +0.8923 改为 −0.8923 rad，与左腕一致。身体 −59°/149°/−75°，头部27°，物体原距离不变。末端镜像姿态误差约0.000614°，位置误差约0.00183 mm。

12/12 几何路径检查通过；本轮各物体一次名义初态物理验证，8/12通过独立验收。不是多次扰动成功率。失败不得用于成功数据集。

| 物体 | 结果 | 原因 |
|---|---|---|
| basket_rectangular | FAIL | CONTACT_POINT_OFF_VISIBLE_SURFACE |
| basket_woven | PASS | - |
| bin_lidded | PASS | - |
| bin_pedal | PASS | - |
| bin_round_lidded | PASS | - |
| bin_round_pedal | PASS | - |
| box_reinforced | FAIL | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED |
| box_white | FAIL | CONTACT_POINT_OFF_VISIBLE_SURFACE |
| case_handheld | PASS | - |
| stool_rectangular | PASS | - |
| stool_wood_four_leg | PASS | - |
| table_side | FAIL | ACTUAL_VELOCITY_LIMIT |

手提箱 repeat2 完整站姿→低位→抓取→抬升通过。准备阶段20秒七阶平滑插值；修复了头部提前锁在0°、却命令27°的问题：准备期间保留原头部关节限制，到达目标后与身体一起锁定。repeat1失败保留在first_attempts.json。其余11类从统一低位开始，完整站姿过渡仅做几何检查。控制器验收阈值未放宽。

使用 `python iterations/20261124_symmetric_wrist/source/symmetric_workspace.py --workspace /path/to/new_workspace --package /path/to/g2_sim` 创建独立工作区。之后运行 `python realpose_<asset>_trial.py --asset <asset> --plan-dir realpose_plans --gpu <id>`，再 `python verify_realpose.py --trial-dir realpose_trials/realpose_plans/<asset>/candidate0_repeat1`。完整站姿流程使用手提箱的 `--plan-dir symmetric_preparation_plans`。具体结果、精确源码与hash见results及source/source_versions。

仍为固定底盘、理想身体驱动、1kg刚体的仿真验证。旧240条数据未改动；本轮未进行12×20扩量，官方腕部图像仍是静态预览，未宣称三路同步训练采集完成。

远端原始轨迹/视频：`/data20t/embodied/zkj/inter-nav-results/g2-wrist-symmetric-20261124/realpose_trials`。腕部左右图见review。
