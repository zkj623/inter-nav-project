# 基于宇树官方配置训练 Go2 替代策略

最新：v160 T05在低速探索点停滞，已保存失败前缀；执行反馈修复通过765项回归，原始T05 v162正在重跑，同版T04未启动。见[当前问题、修复与录像](go2_execution_stall.md)。下方为本页对应版本的历史实现和验证。

更新2026-10-30。**1024环境、2000轮第一阶段训练已完成，45维接口及新策略的有界输入候选通过诊断，原始T05 v160正在运行，同版T04尚未启动。** 当前完整证据与限制见[部署及原任务复测](go2_unitree_deployment.md)。训练源码仍为`8ab8daaaf925d4180a455343b46cd402b4b0e861`；训练奖励不代表任务成功。

## 来源与适配

官方[Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3)所查版本没有附带Go2权重，本次从随机初始化训练。保留其Go2 45维Actor、60维Critic、512/256/128网络、PPO参数、速度指令课程、奖励和域随机化。每个移植文件保留原始地址和SHA256，详见[来源及差异清单](../tools/go2_official_training/upstream_manifest.json)。这是基于官方代码的自行训练策略，不是宇树出厂或官方发布的预训练权重。

服务器为Isaac Sim4.2.0.2和550.107.02驱动，采用已验证的Isaac Lab1.4源码和现有RSL-RL2.3.1运行，不修改共享环境安装。适配内容包括旧命名空间、射线配置字段、通过applied_torque提供Critic关节力矩；动作按官方±100限幅。关闭未被Actor/Critic使用的高度扫描、地形变化和纹理，平地高度保持0.01 m。

物理使用与导航相同的19刚体、15.019 kg标称Go2资产，SHA256为`75958c8286d2dfdf7617108a685110ac831dfc47f2546e71a8cc2ccaecbd83a5`，不缩放机器人。采用导航匹配的DC电机25/0.5、23.5 Nm和30 rad/s，未采用官方Go2HV的不同转矩曲线。保留官方训练时的质量/摩擦随机化，Actor 50 Hz、物理200 Hz；训练倾角终止收紧至30°。训练中的跌倒复位只属于独立RL采样，不能用于导航rollout。导航启动15°、全程30°、最终实测误差0.25 m等标准不变。

## 已完成的训练管线检查

第一次32环境初始化时，过大的render_interval造成5000秒渲染时间步，未进入梯度更新，已终止并保留原因记录。随后改回decimation=4；headless模式自行省去不必要渲染。

第二次32环境检查完成5轮、3840条状态转移，用时28.25秒，进程组内存峰值2.896 GiB。Actor最大参数变化0.01451；CPU导出与训练模型推理的最大差值3.73e-8。GPU物理、45/60维输入、12个关节顺序、19刚体和DC参数均经运行时检查。这个权重仅证明训练和导出管线可运行，没有进行步态或任务验收。

第一阶段在1024环境从随机初始化训练2000轮，不从5轮短测权重继续。每100轮保存检查点，逐轮记录损失、回报、存活时长、速度跟踪等episode指标。最终导出`actor.pt`和供导航加载器使用的`actor_checkpoint.pt`，同时保存观测顺序、缩放、限幅、关节顺序和模型哈希。45维观测适配和原生逐步比对已完成；原始任务验证仍未完成，见[部署验证](go2_unitree_deployment.md)。

## 运行命令

在实验分支仓库根目录执行。每次使用新的run-id：

```bash
python3 -u tools/supervise_go2_official_training.py \
  --run-id go2-official-training-example \
  --num-envs 1024 --iterations 2000 --cap-gib 24
```

监督器采用24 GiB硬上限、零swap，主机可用内存至少56 GiB持续60秒，GPU空闲至少14 GiB，并持有与原场景相同的串行锁。32环境管线检查使用`--num-envs 32 --iterations 5 --cap-gib 8`，主机准入40 GiB。内存低于16 GiB持续10秒暂停，4 GiB紧急终止，恢复和退出沿用现有资源保护。不要并行启动本任务的训练与仿真。

结果根目录：`/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027/<run-id>`。实时状态在相邻`<run-id>.status.json`；本次启动助手额外生成的`<run-id>.launch.json`记录启动版本；直接运行上述监督命令时，以状态文件和源码清单为准。实际工作进程PID见`<run-id>/scope_identity.json`。实际本轮run-id是`official-go2-v155-train1`。状态文件必须结合对应PID核查，不能从旧文件声称仍在运行。

`data/live.json`是最新训练指标，`data/training_metrics.jsonl`是完整曲线，`data/checkpoints/model_*.pt`是可续训检查点。监督命令支持`--resume /绝对路径/model_*.pt`，但新进程仍需新run-id及资源准入。训练使用可控仿真场景，不读取MV7预建占据图。

## 后续验收

先检查直行—停稳、双向侧移、双向小角度转向后的完整停稳与受力接触，对照现有policy，记录实际速度、倾角、足滑、制动和航向回摆。确保训练与导航观测和动作契约一致后，才切入原始T05，再跑同版原始T04；保持在线空地图、原始坐标、真实侧推、新观测确认、实际穿越、目标停稳及五路25 fps视频。当前没有任何新权重被接受为原始任务成功版本。

详细[运行证据](data/go2_official_training_v155.json)、[官方来源复核](go2_official_policy_review.md)、[此前回摆诊断](go2_yaw_step_diagnostics.md)。
