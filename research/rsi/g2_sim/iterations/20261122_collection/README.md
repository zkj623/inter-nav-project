[**本轮采集已完成：240条正式数据与实际覆盖范围**](FINAL_DATASET.md)。训练入口为冻结清单 `dataset/frozen_manifest.jsonl`。

# G2 物体初态扰动与 ACT 数据采集

快照时间 UTC：2026-11-23T06:35:39.740426+00:00。本快照完整目标是否完成：**True**。目标为12类各20条成功轨迹；当前独立验收计数为 `{'basket_rectangular': 20, 'basket_woven': 20, 'bin_lidded': 20, 'bin_pedal': 20, 'bin_round_pedal': 21, 'box_reinforced': 20, 'box_white': 21, 'case_handheld': 20, 'stool_rectangular': 21, 'stool_wood_four_leg': 21, 'table_side': 21, 'bin_round_lidded': 20}`。数据文件保存在服务器，不在Git中。`collection_status.json` 是本提交时快照，运行中最新状态请读服务器 `collection_progress.json`。

## 任务与物理范围

沿用自然公共低位初姿 `[-40,142,-87,0,0]` 度、固定底座、1kg刚体、理想腰腿驱动1e7，工作时锁定腰腿。手臂/夹爪原力矩限制、物体材质与可视/碰撞几何保留。不是50Nm配重姿态，也不是1200Nm硬件验证。每条任务从相同低位公共全关节初态开始，接近、抓取/托举、抬升>=15cm、保持>=3s；本批大多数轨迹一次初始化到低位，不包含站姿过渡，第一条白箱pilot另有完整过渡。原默认配置没有修改。

物体初态改变后，将既有接触策略的TCP目标按物体SE2位姿变换，并重新求双臂IK及整段路径碰撞检查。机器人公共初姿不随物体平移。探索过XY±5/10/20mm及偏航±3/5度；这些是尝试范围，不能当作全部可行范围。矩形凳额外测试固定偏航、向机器人方向5~20mm平移；难例另有更小扰动采样。准确的每条位姿和实际覆盖范围见dataset manifest与metadata。嵌套 `placement.success` 仅是规划通过；只按 `verification.verified_success` 统计成功数据。

## 已实现的录制与验证

物理120Hz、policy目标20Hz零阶保持6tick。每个控制步在动作执行前渲染固定640x480 RGB并读取16维实际qpos/qvel，再保存真正执行的位置目标(rad)。不录制准备阶段，不给训练图像叠加文字。图像/状态使用同一仿真时钟，camera timestamp偏差需<=1/120s。常规阶段和手提箱test_lift、凳子support_preload/support_settle等特殊阶段连续录制；早期漏阶段产生的数据完整性失败已剔除。只改repeat的同初态重跑和动作回放不计入新数据组。

`source/collection_recorder.py` 与 `test_collection_recorder.py` 检查动作前状态、6tick保持和特殊阶段覆盖。独立验收保留原抬升/倾斜/漂移/四点承力/可视间隙/接触面/非手部穿透判据，并检查可视自相交、根漂移、冻结关节、原关节角与速度、非轮部地面支撑、数据形状/有限值/时间连续性。验收程序版本2增加实际qvel上限和非轮部AABB离地检查。旧10Hz的post-step审计样本未必与20Hz pre-action观测存在同一时刻，`same_time_qpos_max_error_rad=null` 表示没有重合采样点，不是零误差；同步合约另由录制器测试和物理动作回放验证。

白箱pilot的保存动作已在新物理仿真中直接回放，并独立通过。回放不调用expert的retarget、位姿追踪或力平衡；结果见 `evidence/pilot_replay_verify.log`。图像回放不是MP4播放，而是将HDF5动作重新下发给机器人。

## 运行与复现

先读根目录 [交接文档](../../HANDOFF_ACT_COLLECTION.md)。Isaac Python为 `/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python`，当前数据工作区 `/data20t/embodied/zkj/inter-nav-results/g2-act-collect-20261122`，现有模型依赖来自相邻 `g2-roundretry-20261120` 和 `r1lite-collision-deps`。需要同一服务器资产权限；仓库没有完整二进制资产包。所有共享资产链接只读，禁止在链接工作区重建或覆盖模型。

创建新复现目录：

```bash
python3 research/rsi/g2_sim/iterations/20261122_collection/source/collection_workspace.py \
  --workspace /data20t/embodied/zkj/inter-nav-results/g2-collection-replay-new
```

在新目录用Isaac Python运行：

```bash
python plan_collection.py --asset box_white --index 1 --dx .003 --dy -.002 --yaw 1
python collect_box_white_trial.py --asset box_white \
  --plan-dir collection_plans/placement_0001 --gpu GPU
python verify_collection.py \
  --trial-dir collection_trials/collection_plans/placement_0001/box_white/candidate0_repeat1
```

此处 `python` 指Isaac Python，`GPU` 要替换为空闲设备编号。工作区应保持全新；已有trial目录会复用旧结果，不能因此计为新采集。其他物体入口为 `collect_<asset>_trial.py`。`plan_collection_batch.py` 生成初始24组候选，`plan_collection_narrow.py` 和 `plan_stool_translation.py` 提供已明确记录范围的补充候选；`run_collection_batch.py` 批量运行并独立验收，目标每类20、最多每类60次已完成尝试。本次先用5个GPUworker，随后主调度器使用G2_COLLECTION_GPUS=0,2,3，额外调度器在GPU6最多3个、GPU4最多2个并行场景，启动前检查显存。复用前核对设备与资源，两个调度器的GPU必须分开管理；只运行主调度器时默认2,6。

回放：

```bash
python replay_box_white_trial.py --asset box_white \
  --plan-dir collection_plans/placement_0001 --gpu GPU \
  --replay-episode /absolute/path/to/episode.hdf5
```

精确复现历史轨迹应将 `source/source_versions/<source_sha256>.py` 复制成独立脚本名，并选择对应 `dependency_versions` 中的录制器实现及原计划。运行目录/结果隔离，禁止覆盖原始episode。当前源版本缩短静止等待，接近和接触/抬升运动保持原速度；试验过的更快接近导致JOINT_TRACKING_ERROR，已经恢复原速度，失败未计数。为多进程设置16线程并取消墙钟限速；物理仍严格每tick1/120s。向量化重力矩与原实现102姿态比较最大误差2.85e-13Nm，保留所有质量/轴线/驱动上限；日志见 `evidence/fast_gravity_test.log`，原实现也随source保留。

## 数据位置与使用

服务器 `/data20t/embodied/zkj/inter-nav-results/g2-act-collect-20261122/dataset/accepted/<asset>/<group_id>.hdf5` 为通过验收的数据硬链接；原始trial保留episode、review.mp4、samples.json、接触记录、result、metadata与verification。`dataset/manifest.jsonl` 包含类别、分组ID、split、帧数、位姿扰动、文件SHA和原始trial路径。运行中该manifest是暂存版本，split可更新；等本批结束后冻结清单再训练。最终按每类16/2/2划分train/validation/test，按placement group隔离，不按帧切分；归一化只用train。

HDF5为 `/observations/qpos[T,16]`、`qvel[T,16]`、`images/workspace[T,480,640,3]`、`action[T,16]`、`timestamps[T]`、`camera_timestamps[T]`、`phase[T]`。16维顺序以根配置 `policy_action_joint_names` 为准：左臂7、右臂7、左右夹爪master各1。没有夹爪从动关节动作、腰腿动作或真值物体位姿policy输入。物体真值/接触力只用于expert与验收。

当前交付是演示采集与回放，未训练ACT；ACT16维适配、loader、训练与Isaac闭环评估仍按根交接文档推进。20条是首轮数据规模，不是随机成功率或训练效果保证。开发期相机/录制错误、加速诊断和自适应缩小采样范围均保留在尝试记录里，不能把全程通过/尝试数解释为某一冻结expert的无偏成功率。
