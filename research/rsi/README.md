# RSI：policy–skill 协同迭代研究

本模块属于现有 `inter-nav-project`，但不改变默认导航任务。RSI 是研究/优化层，清障场景是 benchmark，机器人 policy 是底层物理能力；三者通过任务、`ctx.execute(...)`、Outcome 和 episode 数据连接，而不是将算法写死在某个仿真场景里。

## 目录与边界

```text
grutopia_extension/
  manipulation/              双臂 IK、接触搬运、碰撞检查
  interaction_clearance/     清障任务、policy 注册与 skill 执行
grutopia/demo/               原有运行入口；skill-evolution 入口保留兼容
research/rsi/
  evolution/                 skill 候选生成、反馈与选择
  evaluation/                冻结版本、多布局评测、启动保护
  benchmarks/clearance/      六类场景与有界初态配置
  tools/                     数据导出、综合视频与文件核对
  paths.py                   显式结果目录配置
```

上游已经决定“必须交互”并指定物品、通路及允许放置区。本模块只负责交互清障，不负责选择绕行或交互，也不负责完成整个导航任务。现有研究任务以无把手箱子的双手侧夹、抬起、运输、放稳和释放为主，不声称支持任意物体、动态障碍或实机部署。

## 已实现与未实现

| 部分 | 状态 |
| --- | --- |
| 物理 policy 初始化 | 专家/数值规划与反馈控制；不是神经网络策略。 |
| Python skill 执行 | 已实现，受限 `run(ctx)`、调用预算、真实失败返回。 |
| Skill 候选生成 | 保留早期 Astra/Codex 生成与冻结 policy 评测入口。 |
| 六类复杂布局、随机初态评测 | 已实现；冻结源码/资产、保留失败、独立重审。 |
| 轨迹和可视化 | 53 维 observation、22 维 action、物理采样、完整成功/失败回放。 |
| 自动双臂 policy 学习更新 | 尚未实现。旧单臂训练器不能直接训练当前组合接口数据。 |
| 连续自动 policy–skill 双向学习闭环 | 尚未接通；不能将工程迭代和测试称为已经完成的自动协同进化。 |

当前 v4/round-06 没有调用 Astra、训练神经网络权重或使用物体附着。此前生成器的存在不等于本轮实际调用过模型。未来目标是 `policy 初始化 → skill 迭代 → 多布局数据 → policy 更新 → 新冻结轮次`。

## 当前同一个 skill 的流程

`grutopia_extension/interaction_clearance/skills/composable_clear_blockage.py`：

```text
base_move(target='grasp')
 → arm_plan(mode='grasp')
 → grasp_lift()
 → base_move(target='placement')
 → arm_plan(mode='place')
 → place_release()
 → task.return_to_start 为 true 时才 base_move(target='home')
 → verify_clearance()
```

每步失败立即返回真实 Outcome。底盘移动使旧手臂计划失效，执行前核对测量漂移。默认由障碍感知 A* 选择路径，不由 skill 选择固定 x-first/y-first。当前可在原允许放置区内搜索位置/朝向，检查持物运输、降低、释放和回撤全过程，并支持受限携物转向。

箱子是自由动态刚体，靠双手接触力搬运。禁止附着、禁用重力、运动学搬运或 rollout 中瞬移。外部底盘规划间距为 40 mm，持物/手臂预检 15 mm，运行身体安全门为 5 mm，手指代理允许穿入量为 2 mm。失败不会触发降低安全标准或自动重试不安全执行。

## 环境、资产与快速检查

实验室服务器：

```bash
cd /path/to/inter-nav-project
source /data20t/embodied/share/inter-nav/env.sh
export PYTHONPATH="$PWD:/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps:$PYTHONPATH"
$ISAAC_PYTHON -m unittest discover -s tests -p 'test_*clearance*.py'
$ISAAC_PYTHON -m unittest discover -s research/rsi -p 'test_*.py'
$ISAAC_PYTHON grutopia/demo/r1lite_interaction_clearance.py \
  --validate-only \
  --skill grutopia_extension/interaction_clearance/skills/composable_clear_blockage.py \
  --output /path/to/new-empty-preflight
```

`--validate-only` 不启动物理仿真，也不是任务成功样本。其他机器需要 Isaac Sim 4.2、基础运行依赖与 `requirements/rsi.txt`；`python-fcl` 还需要兼容的本机 FCL/二进制环境。旧单臂训练测试额外需要 PyTorch。不要修改共享环境来绕过依赖错误。

整理后的 103 项执行层测试和 9 项 RSI/导出/目录测试全部通过，工具语法及新旧 CLI 入口已核对。全仓测试并非全部通过：共享 SDK 未安装 pytest，主分支还存在导航演示/配置缺失及 `/tmp/qwen3_topology_worker.err` 权限问题；feature 的 8 个全仓 unittest 错误都在原始主分支上出现，未混入对这些问题的修复。该结果不能被解释为所有导航/仿真测试均通过。

机器人 USD、配套 URDF 和完整 mesh 需要通过 `R1LITE_USD_PATH` 指定，URDF 与 USD 采用对应同名文件路径；外部资产不打包到普通 Git。机器人来源为 Galaxea R1 Lite 相关资产，实验室路径与约束见 [资产说明](benchmarks/clearance/ASSETS.md)。箱子和墙体由任务代码程序化创建，质量/尺寸由各场景配置明确给出。

不要设置 `CUDA_VISIBLE_DEVICES`。单次仿真用 `--gpu`，批次用 `--gpus`；完整 campaign 当前固定 GPU 0 和 7，启动前必须确认空闲。每张选用 GPU 同时最多一例仿真。

## 两种评测/生成入口不能混用

- `evolution/skill_evolution.py` 是旧七布局生成器，suite 结构为 `layouts` 和 `train/validation/heldout`，只将训练反馈提供给生成器。原 `grutopia/demo/evolve_r1lite_clearance_skill.py` 现在是兼容入口。
- `evaluation/run_stability_round_v3.py` 和 `run_frozen_campaign_v2.py` 使用六类布局的 `cases` suite。完整阶段是问题回归 2 → 原始回归 12 → 开发 30 → 独立验证 60 → 重启 18，任一阶段未全部通过就停止。
- 旧生成提示包含历史接口描述，不能直接把当前 60 个验证初态送给旧生成器。接入 v4 前需更新 policy 契约和评测适配。模型生成需要执行端已配置的 Codex CLI，本仓库没有模型凭据，本次整理没有调用模型。

单批次重新执行已有冻结版本的例子：

```bash
REPO="$PWD"
RESULTS=/data20t/embodied/zkj/inter-nav-results/r1lite-stability-v4
export R1LITE_USD_PATH="$RESULTS/round-06/robot_snapshot/r1lite_navigation.usd"
export PYTHONPATH="$RESULTS/round-06/runtime:$REPO:$PYTHONPATH"
$ISAAC_PYTHON research/rsi/evaluation/run_stability_round_v3.py \
  --runtime "$RESULTS/round-06/runtime" \
  --suite "$REPO/research/rsi/benchmarks/clearance/baseline_suite.json" \
  --output "$RESULTS/new-regression-UNUSED" --gpus 0 7 \
  --launch-wrapper "$REPO/research/rsi/evaluation/sdk_launch_guard.py"
```

该命令需要已有、哈希匹配的冻结 runtime 与机器人快照，不能把随时修改的 checkout 当作旧版本。新的 skill/policy 轮次应创建独立 runtime、资产快照、manifest 和 `evaluation_sources/`，然后调用 `run_frozen_campaign_v2.py --root RESULTS --round NEW_ROUND`。`install_round.py` 会对照上一冻结源码拒绝覆盖共享改动。禁止覆盖已计分的旧 episode。

## 配置与统计有效性

`benchmarks/clearance/baseline_suite.json` 是六类、每类两个原始配置；`development_suite.json` 和 `validation_suite.json` 是 round-06 实际使用的 30/60 初态。机器人 XY 每轴 ±0.08 m、yaw ±15°；箱子 XY 每轴 ±0.035 m、yaw ±8°。只过滤非法/初始化碰撞状态，不按任务可解性预筛。

**这批验证初态已经被测试和查看，现作为公开复现配置，不能再次用于调参后的“从未见过的独立验证”。** 如果验证失败指导了修改，就将其转为开发数据，下一轮换新的 seed/初态。当前生成脚本采用固定采样配置，原样重跑不会产生新的独立验证集。

## 数据与可视化

通过 `--results-root` 或环境变量 `R1LITE_RSI_RESULTS_ROOT` 显式指定外部结果根；默认是被 Git 忽略的 `research/rsi/outputs/`。结果目录与脚本目录不再必须相同。

Windows 客户端例子：

```powershell
$env:PYTHONPATH='C:\path\to\inter-nav-project'
$results='E:\R1Lite_interaction\R1Lite_稳定搬运迭代_v4'
python -m research.rsi.tools.export_verified_cases --batch round-06/validation --results-root $results
python -m research.rsi.tools.create_campaign_montage --batch round-06/validation --results-root $results
python -m research.rsi.tools.create_validation_overview --round round-06 --results-root $results
```

这些视频工具保留当前 Windows FFmpeg/中文字体默认路径，其他主机需配置兼容环境或调整路径常量；不能声称它们已经支持任意 Linux 渲染环境。导出保留全部失败，核对源文件哈希、帧数和标签。零动作规划拒绝只能显示明确标注的原始场景静帧，不能编造动作录像或独立审计通过。

轨迹记录为 53/22 组合接口，并包含 200 Hz 物理采样、task/skill/source 哈希和重审。未来训练必须联合 runtime eligibility、完整任务评分及独立 audit 筛选；失败不能当成功模仿样本。完整成功要求支撑、静止、在允许区、释放双手、清路且机器人不挡路，不是只把箱子举起来。

原始轨迹、视频、权重、USD/mesh 和完整实验快照留在外部结果目录，不进入普通 Git。仓库只保存源码、配置和 [结果摘要](benchmarks/clearance/round06_results.json)。

## 已验证的 round-06 与剩余工作

冻结执行源码 SHA256：`1976d9d465aad95356cbd55d2b5fd5f35aa304e0c611deb1102b6df7d484b0c0`。本次目录整理未改变这组已执行的物理代码哈希；新的研究工具与兼容入口不是对旧快照的覆盖。

| 阶段 | 成功 / 总数 |
| --- | --- |
| 两个问题配置回归 | 2 / 2 |
| 原始回归 | 12 / 12 |
| 开发初态 | 30 / 30 |
| 独立验证初态 | 56 / 60 |
| 重启复测 | 未启动，因验证未全部通过 |

四例失败为两个零动作的底盘规划拒绝、一个双臂 IK 朝向误差超限、一个身体/箱子正间距降至约 4.93 mm 后触发 5 mm 停止门。后者不是记录到物体穿透，但仍不符合安全成功条件。

完整视频包含全部 60 例，56 成功/4 失败，12 窗口分页、2 倍速、226.44 秒；58 例使用实际动作录像，2 例用明确标注的真实静帧。实验数据位于 `/data20t/embodied/zkj/inter-nav-results/r1lite-stability-v4/round-06`，本地在 `E:\R1Lite_interaction\R1Lite_稳定搬运迭代_v4`。视频相对路径为 `round-06/validation/多窗口汇总/独立验证集_全部60例_12窗口分页_成功56失败4_2倍速.mp4`。

后续先诊断四例失败、改进新冻结版本、换独立验证集并完整验收，再接双臂 policy 学习器及当前 skill 生成器的双向闭环。现在不能宣称所有场景稳定、任意障碍可解、连续时间无碰撞或实机成功。
