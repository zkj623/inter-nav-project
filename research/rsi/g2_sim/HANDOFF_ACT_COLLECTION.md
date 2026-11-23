<!-- frozen-collection-20261122 -->
本轮12类各20条已完成并冻结，共240条，192/24/24划分。请先读 [首轮实际数据结果](iterations/20261122_collection/FINAL_DATASET.md) 与冻结清单。ACT训练仍待实现。
<!-- /frozen-collection-20261122 -->

# G2 交互数据采集与 ACT 训练交接

更新时间：2026-11-22。交接对象：负责继续采集仿真演示并训练策略的同事。代码基线为 `genie2-rsi` 的 `73c70e184ab32a0564a7f031bbb47a6d93768418`；本交接文件之后的文档提交不改变该实验基线。

## 目标与当前决定

第一阶段针对现有 **12 类刚体，每类采集至少 20 条独立成功演示，共至少 240 条**，然后训练一个联合覆盖这 12 类物体的 ACT 双臂抬升策略。20 条指通过验收的轨迹，不是尝试次数；完全相同的初态和控制轨迹重复运行，不计为新的有效演示。20 条是首轮基线规模，不保证足够收敛或泛化。

任务固定为：自然低位公共初姿 → 双手接近 → 抓取/托举 → 抬高至少 15 cm → 稳定保持至少 3 s。暂不训练搬起后行走、起身、放置或椅子推移。用户已决定先继续沿用现有仿真设定，实际腰腿硬件能力后续另行核实。

**保留现有自然低姿态，不使用 `20261121_torquelow` 的侧倾单臂配重方案。** 当前 12 个成功基线采用固定底座、1 kg 刚体、理想腰腿驱动，进入工作阶段后固定腰腿；现有 expert 中 body cap 为 `1e7`，并非已确认的真机规格。不要在这次数据中悄悄切换成 50 或 1200 N·m，也不要混入解锁腰腿的扭矩实验。记录实际生效设置。训练结果首先只对这一仿真任务负责。

## 现有代码与未完成工作

代码目录为本文件所在的 `research/rsi/g2_sim/`。

| 内容 | 当前入口 | 如何使用 |
| --- | --- | --- |
| 当前低位初姿、动作顺序、成功基线索引 | [ground_lift_training_config.json](ground_lift_training_config.json) | `common_initial_joint_positions_rad` 是公共低位初姿；`validated_simulation_baselines` 是历史通过入口 |
| 最新站姿、20 秒平滑过渡及逐物体运行入口 | [standby_preparation_config.json](standby_preparation_config.json) | 使用 `execution`；不要把旧归档入口误当作最新站姿入口 |
| 最新过渡代码 | [20261120_directprep](iterations/20261120_directprep/README.md) | 消除肩部绕转，低位公共初姿不变 |
| 圆形带盖桶修正 | [20261120_roundretry](iterations/20261120_roundretry/README.md) | 该物体必须使用 roundretry 入口 |
| 12 类公共低姿态的历史结果 | [20261120_symhome](iterations/20261120_symhome/README.md) 与上述索引 | 其中早期失败已由后续控制器修正；不能只读旧表判断最终状态 |
| 视频审阅 | [飞书文档](https://ocnprlp82ms4.feishu.cn/docx/LLihduA7VoWO8ExKS8dckhw9ncb) | 默认站姿到低位过渡、抓取视频与各次迭代说明 |

现有 12 类各有同一低位初姿下的成功基线；**最新站姿 → 简化过渡 → 完整抓取流程只重新验证了 `box_white`、`bin_round_lidded`**。其他 10 类的新入口已有代码和准备路径几何筛查，但需先完整物理复测，再开始扩量。

原始 expert 输出 `result.json`、`protocol.json`、`samples.json`、接触记录、视频和 prepared USD，主要用于验收。`collect_strategy15.py` 是历史结果汇总。后续已实现[初态扰动、同步HDF5录制、批量验收与物理动作回放](iterations/20261122_collection/README.md)，当前完成度见该目录 `collection_status.json`，服务器实时状态见 `g2-act-collect-20261122/collection_progress.json`。下文保留采集与训练合约；采集器已有对应实现，ACT的16维适配、训练和Isaac策略评估仍需完成。运行中的manifest不是冻结训练集，等本批采集结束并冻结分组后再训练。

## 物体清单与最新 expert 入口

所有入口均有 `--asset`、`--plan-dir`、`--candidate`、`--repeat`、`--gpu` 参数，候选均从 0 开始。以根配置 `execution` 为最终来源。

| Asset ID | 物体 | 脚本 | Plan directory | 目标成功条数 |
| --- | --- | --- | --- | ---: |
| basket_rectangular | 矩形篮 | directprep_basket_rectangular_trial.py | directprep_plans | 20 |
| basket_woven | 编织篮 | directprep_basket_woven_trial.py | directprep_plans | 20 |
| bin_lidded | 带盖桶 | directprep_bin_lidded_trial.py | directprep_plans | 20 |
| bin_pedal | 脚踏桶 | directprep_bin_pedal_trial.py | directprep_plans | 20 |
| bin_round_lidded | 圆形带盖桶 | roundretry_trial.py | roundretry_diagonal_validation_plans | 20 |
| bin_round_pedal | 圆形脚踏桶 | directprep_bin_round_pedal_trial.py | directprep_plans | 20 |
| box_reinforced | 加固箱 | directprep_box_reinforced_trial.py | directprep_plans | 20 |
| box_white | 白箱 | directprep_white_validation_trial.py | directprep_white_validation_plans | 20 |
| case_handheld | 硬质手提箱 | directprep_case_handheld_trial.py | directprep_plans | 20 |
| stool_rectangular | 矩形凳 | directprep_stool_rectangular_trial.py | directprep_plans | 20 |
| stool_wood_four_leg | 四脚木凳 | directprep_stool_wood_four_leg_trial.py | directprep_plans | 20 |
| table_side | 边桌 | directprep_table_side_trial.py | directprep_plans | 20 |

本轮不包含软包、软袋、软篮、软提手，以及未确认接触材质的 `stool_small`；两把大椅子已有推移研究，不能计入本次抬升成功数。若后续新增刚体类别，单独建 ID、验证可视/碰撞表面和 expert，再按每类 20 条扩充，不能复用旧类别标签。

## 服务器环境与独立工作区

当前可用服务器为 `zkj@162.105.195.166`。本地 Windows 的 `E:\G2_interaction` 是审阅副本，物理仿真在 Linux 服务器执行。

- Isaac Sim 4.2 对应 Python：`/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python`。
- 已准备的最新完整依赖工作区：`/data20t/embodied/zkj/inter-nav-results/g2-roundretry-20261120`。
- 几何依赖：`/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps`。
- 大型模型、USD 和 vendor 没有全部放入 Git。服务器工作区和资产包仍是必要依赖；没有权限或目录不存在时，应先取得资产，不能用其他模型静默替代。
- 分配空闲 GPU 后设置 `G2_GPU`；不要直接占用文档示例中的任意固定 GPU。

以下命令从自己的 `genie2-rsi` checkout 根目录运行。工作区必须是新目录；不要在既有结果目录里反复覆盖。

```bash
export G2_REPO="$PWD"
export G2_WORK=/data20t/embodied/zkj/inter-nav-results/g2-act-collect-20261122
export G2_BASE=/data20t/embodied/zkj/inter-nav-results/g2-roundretry-20261120
export G2_PY=/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 - <<'PY'
import os, shutil, json, ast
from pathlib import Path
repo=Path(os.environ['G2_REPO']).resolve()
base=Path(os.environ['G2_BASE']).resolve(strict=True)
work=Path(os.environ['G2_WORK'])
pkg=repo/'research/rsi/g2_sim'
work.mkdir(parents=True, exist_ok=False)
# 继承已准备的依赖，不复制旧 trials，避免旧成功结果被当作新采集。
for p in base.iterdir():
    if p.is_symlink():
        (work/p.name).symlink_to(p.resolve(strict=True), target_is_directory=p.is_dir())
    elif p.is_file() and p.suffix in ('.py','.json'):
        shutil.copy2(p, work/p.name)
    elif p.is_dir() and p.name.endswith('_plans'):
        shutil.copytree(p, work/p.name)
# 从分支覆盖当前的 expert 和 fixture，顺序不可颠倒。
for name in ('20261120_symhome','20261120_restpose','20261120_narrowrest',
             '20261120_directprep','20261120_roundretry'):
    iteration=pkg/'iterations'/name
    for p in (iteration/'source').glob('*.py'):
        shutil.copy2(p, work/p.name)
    for p in (iteration/'fixtures').iterdir():
        if p.is_dir(): shutil.copytree(p, work/p.name, dirs_exist_ok=True)
        else: shutil.copy2(p, work/p.name)
for name in ('ground_lift_training_config.json','standby_preparation_config.json'):
    shutil.copy2(pkg/name, work/name)
for p in work.glob('*.py'): ast.parse(p.read_text(encoding='utf-8-sig'))
cfg=json.loads((work/'standby_preparation_config.json').read_text())
for aid, entry in cfg['execution'].items():
    assert (work/entry['harness']).is_file(), aid
    plans=json.loads((work/entry['plan_dir']/(aid+'.json')).read_text())['plans']
    assert len(plans)>entry['candidate'], aid
print('Workspace ready:',work,'entries:',len(cfg['execution']))
PY
```

`generated/`、`robot_assets/`、`object_assets/`、`strategy15_surface_assets/`、`strategy15_surface_proxies/`、`vendor/` 可能是指向共享资产的符号链接。不要在此运行资产生成、惯性重写、碰撞重建脚本。需要改资产时另建实体副本并记录哈希。以上 `prepare_*.py` 历史脚本带固定路径，不要直接运行它们。根目录 `workspace.py run` 只面向早期快照基线，不是当前 12 类最新入口。

先复现已通过的两个完整流程（在另一处选择并导出 `G2_GPU`）：

```bash
cd "$G2_WORK"
"$G2_PY" directprep_white_validation_trial.py \
  --asset box_white --plan-dir directprep_white_validation_plans \
  --candidate 0 --repeat 1001 --gpu "$G2_GPU"
"$G2_PY" roundretry_trial.py \
  --asset bin_round_lidded --plan-dir roundretry_diagonal_validation_plans \
  --candidate 0 --repeat 1001 --gpu "$G2_GPU"
```

白箱结果在 `directprep_trials/<plan_dir>/<asset>/candidate0_repeat1001/`，圆桶在 `roundretry_trials/...`。进程退出码 0 不一定表示物理成功；必须读新生成的 `result.json`、确认 `success=true`、`failure=null`，并完成独立几何/轨迹检查。已有输出目录会被脚本直接复用，**每次尝试都要新 repeat 或新工作区**。

独立检查参考 `verify_liftpair.py`、`verify_restpose.py`、`verify_narrowrest.py`。`verify_roundretry.py` 可检查 roundretry 目录；`verify_directprep.py` 还读取相邻历史工作区用于旧新版对比，移植时应将该对比与核心验收拆开，不能把缺少旧文件当成新轨迹通过。给新采集器实现统一逐 episode 验收，不依赖这两个脚本当前的固定 glob 和历史路径。

## 分阶段采集任务

1. **先录 1 类的 2 条 pilot。** 完成下节同步 RGB/state/action 录制、HDF5 写入、离线回放和小数据过拟合检查，确认数据对齐，再大规模跑。先用 `box_white`。
2. **完整复测 12 类。** 从相同窄站姿平滑降到公共低姿态，执行各自最新 expert。未通过的类别先依据接触/碰撞/IK/夹持力原因修正，不下调验收阈值。修正后的脚本和计划需单独提交并记录版本。
3. **按每类 20 条通过轨迹扩量。** 所有尝试保留索引和失败原因，只把通过轨迹加入训练集。达到 240 条后仍逐类检查，不能用其他物体的富余数量补缺。
4. **新增物体作为第二批扩展。** 先完成现有 12 类数据闭环，再按同一流程为新刚体采集与验收。

每类初步覆盖 20 个不同的物理初态：4 个接近原位置的微偏移、8 个约 ±1 cm 范围的 XY 偏移、8 个可行范围内更宽的 XY/朝向组合。可从平移 ±2 cm、偏航 ±5°尝试；**这些范围是采集起点，不是已验证可行域**。窄把手或边缘夹持应缩小范围，逐项记录。朝向样本应围绕各物体计划本身的朝向变化，例如 `case_handheld` 的既有放置并非统一零度。

位置/朝向变化后重新求该物体的接近、抓取、抬升 IK 与碰撞安全路径；公共机器人低位初姿保持不变。不能只移动 USD 物体而仍播放旧 world-frame 抓取轨迹，也不能将整段机器人关节角任意加噪声。视觉灯光/纹理扰动可作为额外增强，但单纯改变画面不能替代物理初态覆盖。同一个 placement 的重复及视觉变体归同一组，不能跨数据集划分。

质量第一批保持 1 kg，不同时扩大质量、摩擦和几何随机化。将物体姿态扰动、expert 计划/控制器版本、随机种子、相机参数写入每条元数据。

## ACT 数据录制约定

### 策略边界与动作维度

默认站姿到低位的 20 秒准备动作在 policy 外执行。训练 episode 从公共低位初姿 settle 后、`common_approach` 开始前记录，到成功 hold 结束。包含接近、夹持、调节、抬升和保持，不能从每个物体各不相同的 pregrasp 开始，否则模型不会学习从公共初姿接近物体。完整站姿过渡仍保存为审阅视频，不喂给双臂 policy。

输入采用固定工作区 RGB 与实际关节位置 `qpos[16]`。第一版用 1 个固定外部相机，必须看清双手、物体和接触区域；若遮挡严重，在采集前就固定增加手腕相机，并在所有 episode 中一致使用。现有 `/World/GraspReview` 会临时切换俯视截图，不能未经处理就作为训练相机：独立建不移动的训练相机。RGB 建议 `640×480`、无文字覆盖，记录外参、内参、色彩顺序和 resize 方法。

输出是未来 K 步 **16 维绝对关节位置目标，单位 rad**。顺序必须严格复用配置的 `policy_action_joint_names`：

```text
0..6:   idx21_arm_l_joint1 ... idx27_arm_l_joint7
7..13:  idx61_arm_r_joint1 ... idx67_arm_r_joint7
14:     idx31_gripper_l_inner_joint1
15:     idx71_gripper_r_inner_joint1
```

夹爪使用 master 关节角，不直接把弧度转换成未经标定的开口米数；实际读数可能因接触而不同于目标值。mimic followers 不增加动作维数。头、腰腿、底盘不输入动作向量。保留全关节状态用于审计。

Expert 目前使用真实模拟物体姿态、接触力等 privileged feedback；这些用于演示生成和验收，第一版不作为 policy 输入。推理时不能继续调用 expert 的 retarget/力平衡来替模型完成任务；若以后加入接触反馈，应明确成为训练和推理都具备的观测。

### 时间对齐与控制周期

现有物理步长为 **1/120 s**，审阅 samples/video 约 **10 Hz**，expert 目标在物理步更新。它们不是现成的 ACT 同步序列；现有 samples 在执行动作并 step 后记录，不能将同一行状态直接当作该动作执行前的观测。

**需实现**明确的训练控制步。首轮建议 policy/演示录制 **20 Hz**，每个动作保持 6 个物理 tick；低层每 tick 更新既有 PD/重力补偿，动作只含对应的 16 个位置目标。选择该节拍后，expert 也必须以相同保持规则执行并重新通过验收。不要仅每 6/12 帧抽取连续 expert 的瞬时目标，就假定回放会相同。

每步先在物理时刻 t 渲染并读取 `RGB_t、qpos_t`，再确定并记录真正下发的 `action_t`，执行 6 个物理步得到下一观测。渲染不得额外推进物理时钟。相机帧和状态用 simulation timestamp 对齐，避免重复旧图像；图像转 RGB 后保存。标签是限幅后真正执行的位置指令，不是未来实际 `qpos`，也不是手工重建的规划路径。持有阶段仍记录实际执行目标。

先用新节拍录制并原样回放 2 条 pilot，验证图像/状态/动作无一帧错位，机器人和物体行为仍通过验收。若更换为 10/30 Hz，必须同步修改采集、训练 chunk 时域和评估控制器，重新验证并单独版本化数据。

### 每条 episode 的保存格式

**HDF5 合约**如下；新录制器已实现核心观测/动作/图像/时间序列，具体输出与验收以20261122_collection为准。模型训练可采用作者实现的 HDF5 loader，但需适配 G2 维数、数据划分及时间语义。

```text
accepted/<asset>/episode_<id>.hdf5
  attrs: sim=True, schema_version, asset_id, success=True,
         control_hz=20, physics_hz=120, joint_names, camera_names
  /observations/qpos              float32 [T,16]       # 执行动作前实际位置，rad
  /observations/qvel              float32 [T,16]       # rad/s，审计/可选输入
  /observations/images/workspace  uint8   [T,480,640,3] # RGB，无水印
  /action                        float32 [T,16]       # 真正执行的绝对目标，rad
  /timestamps                    float64 [T]          # 仿真秒
  /phase                         每步阶段标签
```

每条同时保存 `metadata.json`、验收 `result.json`、原始接触/全关节审计轨迹和 `review.mp4`。metadata 至少含：source commit/source SHA、脚本与 plan SHA、asset/collision SHA、seed、物体初始位姿/质量/摩擦、公共初姿、固定底座/腰腿锁定设置、生效驱动上限、相机标定、控制节拍、episode 起止时间、失败/成功原因、placement group ID。图像和 action 同长、无 NaN、时间戳严格递增。

失败尝试放 `rejected/` 并写原因，不得混入成功模仿学习数据。先写临时文件，完成物理验收和数据完整性检查后再原子移动到 accepted。总索引 `manifest.jsonl` 记录所有尝试、相对路径、哈希、类别、group、split 和验收结果。大文件存服务器数据目录或对象存储，不把全部视频/HDF5 提交到 Git。

## 成功验收与数据划分

保留已有 expert 的 preload、settle 和 hold 全部判据，**不能用 result.success 一个字段代替独立验收**。主要阈值如下，策略的接触定义以原脚本为准：

- 保持至少 3 秒，整个 hold 最低抬升 ≥0.15 m；物体最大倾斜 ≤20°，相对夹爪漂移 ≤0.02 m。
- 双臂共四个有效支撑接触满足原力阈值和覆盖率 ≥90%；spread 使用末端指节，pinch/托举按原有效分支判断。不能改成“任意一个手指有接触”。
- 承力手指的可视间隙 ≤1.5 mm，接触点到原可视表面误差 ≤2.5 mm；非手部穿透 ≤1 mm，非手部/不应发生的自碰撞冲量 ≤0.001 N·s；保持时原物体网格离地 >5 mm。
- 检查原关节/速度限制、全轨迹可视自相交、公共工作姿态、固定非手臂关节误差 <0.005 rad、底座漂移 <1 mm。物体不能绑定、跟随或在交互中写位姿；不能借身体或环境支撑。

增加采集后的图像、动作时序和回放验收；每个 episode 至少审阅一次视频。对 `bin_round_pedal`、`bin_pedal`、`box_reinforced`、`box_white` 特别检查倾斜和接触余量，不只是压线成功。

每类 20 条默认分为 **16 train / 2 validation / 2 held-out test**，合计 192/24/24。划分按 episode/placement group，在采集前规划保留的姿态组；不能按帧随机切分。同一物理初态的重跑、视觉增强和相同计划派生版本应在同一 split。归一化均值/方差只用 train 计算，保存 joint order 和训练集 ID。两条测试演示仅用于离线诊断，不能代表闭环成功率。

## ACT 训练与闭环评估

ACT 原始论文为 [Zhao 等，RSS 2023](https://www.roboticsproceedings.org/rss19/p016.html)，作者代码为 [tonyzhaozh/act](https://github.com/tonyzhaozh/act)。本次查阅的仓库 HEAD 为 `742c753c0d4a5d87076c8f69e5628c79a8cc5488`；使用时固定 commit，并保存自己的 G2 适配补丁。

作者代码面向 ALOHA 的 14 维状态/动作和 MuJoCo 任务，**不能直接运行它的示例命令就训练/评估 G2**。需实现以下适配：

1. 将 `imitate_episodes.py`、`detr/models/detr_vae.py` 等的 14 维状态投影、模型构造、action head、rollout 缓冲区改为统一配置的 16 维；不能只改一个常量。移除 ALOHA 专用夹爪变换。
2. 用上述 HDF5 合约与 manifest 提供 variable-length loader、显式分组 split、train-only normalization、chunk 尾部 padding mask。原作者 loader 有 sim/real 的不同对齐逻辑，本数据应按明确的 pre-action 定义加载，不再套 real 的历史时移。
3. 注册 G2 数据路径、camera_names 和控制节拍；用 Isaac Sim 环境实现 reset/step/观测/终止/成功判据。作者自带 MuJoCo `sim_env.py` 的任务不是 G2 环境，`--eval` 也不能自动评估本机器人。
4. 先用 1–2 条演示验证模型能过拟合并在同条件闭环复现，再训练 12 类混合数据。按物体均衡采样，避免长 hold 或长 episode 主导。第一版是同一“抬升”任务、多个物体，不必先训练 12 个独立模型。

以下为 **G2 首轮起始超参数建议**，不是已经验证有效的设置，也不是作者原默认值的照搬：

| 参数 | 起始值 |
| --- | --- |
| state/action dimension | 16 / 16 |
| 控制频率 | 20 Hz，与数据一致 |
| chunk size K | 20，即未来 1 秒 |
| 视觉 backbone | ResNet18 |
| hidden_dim / dim_feedforward | 512 / 3200 |
| batch size | 8，按显存调整 |
| optimizer / learning rate | AdamW / 1e-5；backbone 1e-5 |
| KL weight | 10；loss 为 masked action L1 + KL |
| 初始训练预算 | 约 30000 optimizer steps，按验证与闭环结果延长 |
| 随机种子 | 至少 0、1、2，分别保存结果 |

推理时每个控制步重新观察、预测动作 chunk，第一版只执行第一个动作，然后重新预测；模型稳定后再对比 temporal ensembling。输出反归一化后经原关节角/速度约束与同一低层控制器执行，不应一次无反馈地执行全部 1 秒动作。推理期 CVAE 使用先验均值/零 latent，不能读演示的未来 action。相机、动作限幅、20 Hz 保持与训练数据必须一致；若限幅频繁触发，记录并作为失败分析。

模型选择不能只看 validation loss。每类另做 **20 次未参与训练/调参的新 placement 闭环评估**，固定并保存 seed 列表；目标评估 rollout 共 240 次，与 240 条演示分开计数。报告逐类成功/尝试、总体宏平均、倾斜/漂移/接触分布、失败分类和代表视频。评估不允许 expert 接管、物体位姿反馈校正或失败重试后只统计成功。需要调参时用 validation placement；最终测试集只在配置冻结后使用。

## 需要交付的内容与完成标准

- **代码**：统一采集入口、物体扰动及重新规划、同步录制器、episode 验收器、回放器、ACT G2 训练及 Isaac 闭环评估入口。提交时附真实可运行命令和环境锁定文件。
- **数据**：12 类各至少 20 条通过的独立演示；manifest、分组 split、元数据、审计轨迹、数据/资产哈希及服务器路径。不得只交视频或 samples.json。
- **验收报告**：逐类成功条数、尝试数/失败原因、采用的扰动范围；12 类最新完整流程和新录制节拍均通过复测；回放及同步检查通过。不得只把 repeat 1 改到 20 当作完成。
- **训练与评估**：至少一个 ACT checkpoint、归一化统计、16 维 joint order、相机/控制配置、训练日志、commit 与 seed、逐类闭环结果和可审阅视频。20 条不足的类别给出增量数据方案，不承诺固定成功率。

接手优先顺序：**白箱 2 条 pilot → 同步数据/回放 → 12 类完整复测 → 每类 20 条 → 单条过拟合 → 联合 ACT → 新 placement 闭环评估**。维持当前配置与几何验收，先把数据和策略闭环做通，不扩展到真机执行。
