# Go2失稳：策略与物理后端对照

当前完整任务状态见[最新结果](go2_final_push_release.md)：v146 T04部分侧推后失败，v147验证最后一次接触撤离。下方记录此前CPU/GPU物理和保持逻辑对照，不能将各历史成功案例合并为当前版本任务成功。

## v136候选：原rough，GPU物理，不启用关节目标保持

v134相同GPU物理下，开启保持的旧rough只通过1/3，flat通过2/3。v135关闭额外保持后，旧rough的三个就位/转向/启停案例全部通过，全物理步最大倾角分别为21.386234°、19.150566°、20.586730°，最终均满足实测停稳；两次保持相关失败不再出现。没有更换权重、修改PD/DC电机参数或扩大30°界限。

flat在v135关闭保持后仅通过1/3：T04虽然健康并停稳，但恢复运动只移动0.092m，未达到0.25m要求；另一个T05方向达到30.438°。因此未替换原rough权重。旧rough对照说明物理后端与保持逻辑的组合值得优先验证，不代表当前policy在所有场景都稳定。原始T04/T05仍需完整推箱复跑；短测的保存几何不作为在线导航先验地图。证据：[GPU与无保持对照](data/go2_gpu_raw_candidate_v136.json)。

已加入显式`--physics-backend gpu_dynamics`。默认`default`保留原World配置；GPU选项在`env.reset()`之前启用GPU dynamics/GPU broadphase并关闭CCD，保留原CPU/NumPy数据读取。复位后检查实际场景开关，未生效则报错。每次完整任务保存`physics_contract.json`，并写入机器人/录像配置；源码快照增加物理配置模块，共55份。691项回归检查通过，1项既有G1相机检查未运行。v136使用生产配置接口的三个真实物理集成案例全部通过，逐项复现v135旧rough的倾角、完成和停稳结果，8GiB资源上限及实际GPU场景开关均已核验。原始场景v137按固定源码`5db0af14da9abdfe4e5f77b171b280b6b95c1465`排队，先T04完整验收，再T05；仍未宣称新版原始场景通过。

在README完整命令后追加以下候选参数，先选择单一任务`--task-ids T04`，其后再回归T05；候选运行不添加`--stationary-joint-hold`。该开关仍保留给历史T05复现。

```bash
--physics-backend gpu_dynamics \
--policy-path /data20t/embodied/share/inter-nav/assets/robots/go2/policy/move_by_speed/rough_model_7850.pt \
--robot-usd /data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027/official-policy-review-20261030/nvidia_isaaclab_4_5/go2_training_flattened.usdc \
--robot-physics-profile isaaclab_training --actuator-mode explicit_dc
```

上面是追加参数，不是独立shell命令。完整场景采用24GiB硬上限、零swap、56GiB主机准入和共享串行锁。


更新：2026-10-30。**尚未解决全部任务：T05 v127通过，T04 v127失败。**

## 历史v127完整任务状态

正式运行源码仍为`648ecf46fd703557bf7d720ab773f18e76c9229e`。T05已将B07侧推0.836079m，在线确认并穿越新通路，停稳误差0.173951m，全物理步最大倾角21.712124°，五路完整25fps视频已校验。见[T05证据](go2_supported_idle_hold.md)。

T04运行5905主步，在B06推箱前的就位转向阶段失稳，最大倾角37.904221°、终点误差2.749809m，未推开B06。失败窗口没有非地面接触；指令包含后退和正向转弯，18°恢复触发后倾角仍继续增加。全程失败视频已保存，不能因为进程正常退出、最后姿态恢复或`push_count`已计入一次尝试，就声称完成推箱。

Windows失败总览：`E:/inter-nav-results/go2-online-smooth-20261027/recognition-skill-v127-t04-ground-truth-try1/T04/video/combined.mp4`。五路各1236帧、25fps、49.44秒。

## 已排除的直接替代

- v128先直行建立步态、再加入转向的实验仍失稳，另一个T05方向超时；改动已撤回，没有混进正式控制源码。
- v129第三方flat6800在三个就位/启停案例中只通过一个；T04及另一个T05方向超过30°。不能直接替换后称作修复。
- 另一个本地model_9700有236维输入，与当前接口不同，没有用补零方式强行替换。
- 宇树官方RL Gym/RL Lab有Go2训练/部署代码，所查固定公开版本未附带Go2权重。来源、接口差异及范围见[官方核查](go2_official_policy_review.md)。第三方flat/rough和NVIDIA检查点均不是宇树出厂策略。

## 物理配置差异

实际Isaac Sim为4.2.0.2。安装的IsaacLab2.0.2不适合直接作为该版本原生参考，因此在独立路径使用官方IsaacLab v1.4.0（`dd044a4977722d85e966c5b63949093c9b14667f`）；未改共享环境。

v131原生探针遗漏默认复位并错误创建第二个Core articulation，结果无效，不能拿来比较。修正后的v132使用原生环境、执行器和复位，flat6800完成17秒序列，50Hz采样最大倾角15.040°。其本体观测与当前Core tensor读取/NumPy观测公式最大差值1.19e-7，关节顺序一致。这是观测数值核对，不能证明整个控制集成完全一致，也不是全物理步导航验收。

v133在当前框架中固定相同flat6800、19刚体/15.019kg资产、1cm地面、50Hz推理、200Hz物理及指令序列，只对照GPU dynamics + broadphase：

| 配置 | 完成17秒序列 | 全物理步最大倾角 | 最低机身高度 |
|---|---|---|---|
| 默认CPU物理，MBP broadphase | 否，12.46秒触发健康保护 | 31.150646° | 0.235962m |
| GPU dynamics，GPU broadphase | 是 | 18.029496° | 0.259612m |

序列是零速3秒、后退−0.2m/s并转弯+0.6rad/s 3秒、零速4秒、前进0.5m/s 3秒、零速4秒，保留同一速度渐变器。两组均未启用支撑关节保持。GPU组保留CPU/NumPy数据读取，`PhysicsContext.device`及缓存`use_gpu_sim`仍显示CPU/false；实际USD场景属性为`enableGPUDynamics=True`、`broadphaseType=GPU`，两类信息均完整保存。未声称切成GPU tensor pipeline。

这一对照说明后端配置会影响这组物理轨迹，尚不能证明全任务稳定或把CPU/GPU以外的差异全部排除。原生v132还使用不同全局稳定化/重力/反弹阈值配置，不能把15.040°与18.029°之差全归因于某一参数。完整证据见[JSON记录](data/go2_policy_physics_diagnosis_v133.json)。

## 当时的下一步（v134，已完成）

v134在此前相同的三个T04/T05保存几何约束就位案例中，分别运行旧rough7850和flat6800，启用GPU dynamics/GPU broadphase，其余源码、保持和恢复逻辑不变。诊断地图仅用于限制短测指令，不作为真实导航的先验地图。通过后再决定是否集成并复跑原始T04/T05；短测通过不代表推箱任务通过。

保持原场景尺寸、1cm地面、在线空地图初始化、全局/局部1/10Hz、Voronoi引导、可见性门控箱子真值、无rollout瞬移、0.25m到点与全物理步30°要求。诊断8GiB/零swap、40GiB准入；完整场景24GiB/零swap、56GiB准入，沿用共享串行锁。
