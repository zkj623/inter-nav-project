# Go2 模型匹配与候选策略（v117，2026-10-30）

原始 T04/T05 尚未完成。T05 v108 已把 B07 推开 0.778568 m，并获得在线通路确认，但后续导航失稳，未穿越到点；T04 v109 在接触箱子前失稳。两次最大倾角分别为 39.879° / 42.480°，最终目标误差为 1.529 m / 4.822 m。进程正常退出和局部推箱成功均不代表任务成功。

## 来源和部署差异

旧 `rough_model_7850.pt` 来自第三方 isaac-go2-ros2。宇树公开 RL Gym / RL Lab 仓库的核查见 [官方策略核查](go2_official_policy_review.md)；未在所查公开来源找到可直接使用的 Go2 权重，也没有启动宇树策略训练。

本候选使用 NVIDIA Isaac Lab 发布的 `Isaac-Velocity-Rough-Unitree-Go2-v0` 检查点（235 维输入、12 维动作），不是宇树出厂固件。权重、资产的固定下载地址和 SHA256 见 [来源清单](data/go2_reference_manifest_v117.json)。匹配的训练代码为 Isaac Lab v2.0.0；policy 50 Hz、物理 200 Hz、动作尺度 0.25、显式 PD 25/0.5、DC 饱和力矩 23.5 Nm。

旧部署资产有 39 个刚体，零质量 IMU/radar 在 PhysX 中各变成约 1 kg，实测总质量 17.019 kg；训练资产为 19 个刚体、15.019 kg。训练资产的 27 个碰撞形状、默认世界位姿与原资产逐项匹配，未缩小机器人或删除碰撞体，见 [碰撞几何对比](data/go2_collider_equivalence_v117.json)。旧部署小腿的原生限速为 15.7 rad/s，训练代码则将显式执行器的 30 rad/s 限速写入 PhysX。

显式选择 `isaaclab_training` 配置后，加载检查过的训练资产，设置全部关节原生限速 30 rad/s、刚体线速度上限 1000 m/s / 角速度上限 1000 度/s，并匹配小腿和足部接触材料 0.8/0.6、multiply。地面、箱子和头部材料保持原设置。solver 4/0、关闭自碰撞、原生驱动和关节摩擦为零，本来已正确，不根据初始化前警告改动它们。

资产加载器以前会因文件较小而替换一个合法的引用式 USD；现在先检查实际组合的 Go2 刚体、12 个关节和碰撞几何，合法引用文件优先采用。启动时输出 `robot_model_contract.json`，检查实际引用、哈希、质量、刚体数、碰撞体数、限速、驱动和材料；不匹配直接报错。默认 `legacy` 配置保留历史复现能力。

## 短测试结果与限制

每种配置采用相同的 10 组独立初始化指令，包括前后移动、双向侧移、转向、微调及模式切换。原始健康标准为倾角 <30°、机身高度 ≥0.21 m；最终停稳需连续满足测量条件 0.4 秒。独立案例的复位只用于此诊断，不用于导航 rollout。

| v116：同一训练资产、30 rad/s | 健康完成 | 最终停稳 | 最大倾角 |
|---|---:|---:|---:|
| 旧 rough，旧接触设置 | 9/10 | 6/10 | 30.28° |
| 旧 rough，训练接触设置 | 10/10 | 8/10 | 28.94° |
| NVIDIA rough，训练接触设置 | 10/10 | 7/10 | 20.10° |
| NVIDIA flat，训练接触设置 | 7/10 | 6/10 | 5.34° |

flat 的失败来自机身过低，不能因为倾角小就算通过。NVIDIA rough 候选的倾角余量更大，但最终停稳数量没有超过匹配后的旧 rough；因此仅作为完整场景候选，不宣称全面优于旧策略。旧资产/旧 rough 的 v111 为健康 10/10、最终停稳 4/10，也说明空场短测不等同于原始任务成功。完整数据见 [测试记录](data/go2_policy_benchmarks_v117.json)。v114 两组“training asset”被加载器静默替换，已排除；有效规范资产对照从 v115 开始。

## 获取资产和运行

在仓库根目录、Isaac Python 环境中运行；本工具只下载固定哈希的公开资源，模型二进制不提交到 Git。

```bash
source /data20t/embodied/share/inter-nav/env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
go2_reference=/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027/official-policy-review-20261030/nvidia_isaaclab_4_5
"$ISAAC_PYTHON" tools/prepare_isaaclab_go2_reference.py --output "$go2_reference"
```

在 [README 的完整在线导航命令](../README.md#go2-在线建图规划与交互导航) 中追加以下参数（任务选择 `--task-ids T04` 或 `T05`），其余参数保持一致：

```bash
--policy-path "$go2_reference/rough_checkpoint.pt" \
--robot-usd "$go2_reference/go2_training_flattened.usdc" \
--robot-physics-profile isaaclab_training --actuator-mode explicit_dc
```

上述三行是追加参数，不是独立 shell 命令。短测试的可复用入口如下；`--case-ids forward_stop,mode_switches` 可缩短为集成烟测，省略则运行十组。

```bash
"$ISAAC_PYTHON" tools/benchmark_go2_locomotion.py \
  --gpu 1 --output /path/to/new-diagnostic-output --label nvidia-rough-training \
  --policy "$go2_reference/rough_checkpoint.pt" \
  --robot-asset "$go2_reference/go2_training_flattened.usdc" \
  --physics-profile isaaclab_training --command-profile slew
```

完整场景以固定提交的独立 worktree 串行运行：24 GiB 上限、零 swap、主机可用内存 ≥56 GiB 持续 60 秒后准入。诊断为 8 GiB 上限和 40 GiB 准入。原始 Go2/MV7 尺寸、1 cm 地面、原始目标、空地图初始化、1/10 Hz 地图、Voronoi、可见性约束的箱子真值和五路 25 fps 全程视频不变。验收必须同时满足实际推箱清路、重新观测通路、实际穿越、目标误差 ≤0.25 m、全物理步倾角 <30°、最终停稳及视频/源码审计。

## 集成检查

v117 正式代码回归 654 项通过，1 项既有 G1 相机测试未运行。真实 SDK 烟测检查通过：原生模型合同有效，前进启停与动作切换两组均健康完成且最终停稳；数值重现此前同配置的基准测试。证据：[初始化参数](data/go2_integration_v117_robot_model_contract.json)、[两组短测](data/go2_integration_v117_summary.json)。此记录不是 MV7 全任务验收。
