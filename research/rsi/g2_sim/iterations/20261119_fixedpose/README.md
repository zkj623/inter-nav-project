# G2 固定低姿态、仅双臂抬升验证

训练范围收敛为原地抬升。固定底盘世界位姿，并通过关节上下限相等锁定腰、头和轮组；只有左右 7 轴手臂和两个夹爪主关节可动，policy 动作为 16 维绝对关节位置目标。尚未开始训练。

选择的共同躯干姿态为 body_joint1..5 = [-50, 150, -85, 0, 0] 度，对应肩高约 0.791 m、整体躯干前倾 15 度。150 度是折叠机构的单关节角度，不是整体前倾角。完整名称、弧度、动作顺序见 fixtures/fixed_ground_lift_config.json。

初始化共用躯干、头与底盘姿态；双臂目前仍从各物体经过审计的预抓取姿态开始。尚未验证全物体共用的双臂 reset 姿态及其接近路径。

筛选 6 个躯干候选，复用既有成功抓取几何，手臂独立逆解，前后站位最多调整 8 cm；按抓取与 20 cm 抬升全路径检查。首轮仅取第一个可达站位，然后追加碰撞感知的站位搜索。最终 11 个物体通过可视、自碰撞、原生凸几何和全身/物体审计，见 selected3 审计摘要。几何可行不等于物理抓取成功。

两把大椅子只保留推移结果，不进入抬升集；软包圆凳的刚性接触面未确认；边桌在本轮姿态与既有接触区域的有限搜索中未找到可行路径。未证明这些物体永远不能在固定姿态抬升。

## 实际物理测试

| 物体 | harness 通过 | 失败原因 | 最低保持抬升 m | 最大锁定关节误差 rad |
| --- | --- | --- | --- | --- |
| box_reinforced | True | None | 0.17022709548473358 | 1.1106758392998017e-05 |
| case_handheld | True | None | 0.16712552309036255 | 1.3511813449440524e-05 |
| box_reinforced | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.1698043793439865 | 1.1694298336806241e-05 |
| stool_rectangular | False | HARNESS_ERROR | None | 0.0 |
| stool_rectangular | True | None | 0.19826222956180573 | 1.152260483650025e-05 |
| stool_rectangular | False | UNDER_SUPPORT_FOUR_CONTACTS_NOT_ESTABLISHED | None | 1.132861234509619e-05 |

继承 1 kg 统一质量、理想躯干驱动和原手臂/夹爪力矩上限。保留接触表面误差、真实接触、物体倾斜、保持与碰撞检查；不向物体添加附着或位置跟随。固定根部是本轮用户指定的训练简化，不作为移动底盘稳定性或实机证据。每个代表物体只有本轮实际完成的次数，未做扰动成功率统计。

初版新增关节记录包含 numpy.float32，导致 JSON 写出错误；记录格式在 v2 改为 Python float。两个旧运行取消，保留日志；其中一个在停止时记录时钟错误。它们不计为能力失败或通过。

固定姿态后箱子初版四指保持覆盖率只有 63.3%，尽管抬高约 17 cm 仍判失败；追加使用原 deadband 控制器重测。木凳初版四分支承力未稳定建立；追加 supportstable 版本，使用三个采样点的平均力、每步 0.05 度调节，双分支均超过 0.75 N 后暂停均衡调节，最多 96 次预加载。四接触大于 0.5 N、保持覆盖率至少 90% 的验收标准不变。

fixtures/fixedpose_physical_geometry_review.json 对成功试验的实际关节与物体位姿逐采样复核非手部几何穿透，阈值 1 mm；这不是非手部接触冲量测量。已执行旧计划保留原始来源姿态的历史 envelope/descent 元数据，执行使用的是重解后的 q_pre/q_grasp/q_close/q_lift；selected3 计划已显式区分历史与当前 envelope 并移除未执行下降元数据。

## 复现

先用 ../20261118_push/replay.py prepare 准备新的独立工作区（原资产与 collision-deps 参数不变），然后覆盖复制本目录 source/*.py 和 fixtures/ 内的文件/目录到工作区根目录。不得修改共享资产。

运行固定姿态测试，例如：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/isaaclab/bin/python fixedpose_sidev2_trial.py --asset box_reinforced --plan-dir fixedpose_3_validated_plans --gpu 0 --repeat 2
```

木凳原版用 fixedpose_supportv2_trial.py，修正版用 fixedpose_supportstable_trial.py，硬箱及箱子 deadband 重测用 fixedpose_deadbandv2_trial.py。已执行物理测试使用 fixedpose_3_validated_plans；后续统一候选集是 fixedpose_selected3_validated_plans。trial_index.json 保存已执行源码 SHA、协议、结果以及大视频/采样文件服务器路径和 SHA。manifest.json 校验归档字节。
