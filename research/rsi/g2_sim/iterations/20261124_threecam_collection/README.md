# G2 三相机 ACT 演示采集

当前状态：两条三相机试采及纯动作回放已通过，12 类批量采集正在运行，尚未冻结 240 条训练集。以服务器 `collection_progress.json` 和各 episode 的 `acceptance.json` 为准。归档中的 progress 是快照，不能视为实时状态。

代码基线为 `genie2-rsi` 的 `f649f70a0713700bbc58397c948be157aaf15e16`。独立工作目录：`/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124`。旧 240 条数据不混入本批。

HDF5 和 metadata 内历史字段 `source_commit` 指上述基线，并不表示这次新增采集代码已存在于该基线提交。实际执行代码由每条数据的 `source_sha256`、recorder 哈希、`source_versions/` 精确存档及本迭代 `results/current_source_manifest.json` 追溯。首批部分任务启动后仅添加了驱动参数导出字段，没有改变控制行为；原版本也保留在源文件存档中。

目标为 12 类刚体各 20 个不同 XY 初态的成功演示。保持各物体原朝向、1 kg 质量、固定底座和身体、公共低位初姿，以及现有物理验收阈值。先尝试 ±5 mm 到 ±1 cm 平移；失败后允许缩小采样范围，但必须报告实际范围，相邻计数样本至少相差 1 mm。所有接近、夹持、接触接近、抬升及稳定保持目标重新求解，并保留失败索引。

策略输入为头部 RGB、左右官方腕部 RGB 和实际 `qpos[16]`；动作是 20 Hz 的 16 维绝对关节位置目标（rad），每步保持 6 个 120 Hz 物理 tick。相机外参随实际关节状态更新；所有渲染刷新均断言不推进物理时钟。头部原始图像 640×400，腕部 640×528，保留官方畸变与安装参数。外部相机只用于审阅和历史物理审计，不作为训练输入。

动态三相机试采暴露了腕部图像刷新滞后及碰撞调试网格遮挡。现已按实际 FK 更新腕部相机位姿、在同一物理时刻刷新图像，并仅关闭 `/collisions/` 网格的渲染可见性；碰撞几何、碰撞配置和真实可视模型未删除。最初两个黑帧试采不进入训练集，白箱 placement 1/2 的修复复测使用 repeat 2。

执行顺序：

1. `plan_threecam.py` 生成位置扰动和有碰撞检查的路径。
2. `realpose_<asset>_trial.py` 执行原物理控制器，并通过 `threecam_recorder.py` 同步记录。
3. `verify_realpose.py` 独立检查物理稳定性和原始数据。
4. `verify_threecam.py` 检查三相机、实际镜像姿态、源文件哈希及实际执行指令与记录动作的时序对应。
5. 两条 pilot 必须原样回放，仅使用记录动作，不调用 expert 的物体跟踪或夹持微调；生成 `threecam_pilot_verified.json` 后才能扩量。
6. `run_threecam_collection.py` 可恢复地采集每类 20 条合格数据，失败和近重复样本不计数。
7. `finalize_threecam_dataset.py` 在 240 条全部齐备后冻结索引、哈希、按 episode 分组的 192/24/24 划分，以及仅用 train 计算的归一化统计。

`act_threecam_dataset.py` 是 ACT 数据适配器，返回 RGB `[3,3,528,640]`、状态 `[16]`、动作 chunk `[K,16]` 和 padding mask `[K]`。头部画面上下各补 64 行，不裁剪或旋转；腕部保持原始比例。默认 K=20。该文件不是 ACT 训练器或已训练模型。

现有抓法使用每类固定的夹爪增益及力矩上限，各类之间并不相同。最终数据集在 `policy_execution_profiles.json` 中保存这些参数，manifest 为每条轨迹标记 `drive_profile_id`；同一类不得混用不同配置。ACT 仿真评估必须为该物体加载对应底层驱动参数。纯动作回放不调用专家物体跟踪或力平衡反馈，但仍需要相同的底层关节控制及重力补偿。不要将这批数据描述为统一夹爪驱动配置或真机数据。

复现工作区：在仓库根目录执行 `python research/rsi/g2_sim/iterations/20261124_threecam_collection/source/threecam_workspace.py --workspace /absolute/new/workspace`。依赖和共享资产路径沿用前序 natural_four 工作区；运行仿真需要本项目 Isaac Sim 环境。新工作区先完成两条纯动作回放的 pilot 验证，再运行批量采集。现有服务器任务用 `collection.lock` 防止重复启动；`watch_threecam_freeze.py` 只有在 12 类都达到 20 条后才生成最终 manifest 并检查真实 HDF5 的 ACT loader。

进入新工作区后，使用 Isaac Sim 的 Python 顺序运行 `validate_threecam_pilots.py --gpu <空闲GPU>`、`run_threecam_collection.py --gpus <可用GPU编号列表>`、`finalize_threecam_dataset.py`。GPU 列表允许重复编号以并发，但需事先检查显存余量。不要在当前正在采集的工作区重新运行 pilot 或改动 recorder。默认批量最多尝试每类 120 个位置，若仍不足 20 条则标记 `needs_iteration`，不会伪造完成或自动降低验收标准。

冻结前不要训练。完成后以 `dataset/manifest.jsonl`、`dataset/normalization.json`、`dataset/dataset_summary.json` 为准；试采日志、回放验证、逐次失败原因和视频均保留在工作目录。只验证本仿真设定，不代表真机腰腿负载能力或任意位置泛化。
