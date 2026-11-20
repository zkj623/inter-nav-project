<!-- effortpose -->
[Experimental original50Nm posture investigation](iterations/20261120_effortpose/README.md): upright raised-work candidate only; accepted standby/low-home and12 lift baselines remain unchanged.
<!-- /effortpose -->

<!-- roundretry -->
All12 assets now have nominal verified lifts from the same low work home. [Round-lidded-bin fix and validation scope](iterations/20261120_roundretry/README.md). Latest preferred standby complete workflows verified for two assets only; default/work postures unchanged.
<!-- /roundretry -->

<!-- directprep -->
Preferred startup uses a work-compatible shoulder IK branch to remove the arm swing detour. [Verification](iterations/20261120_directprep/README.md). Low work home unchanged; one white-box full workflow verified.
<!-- /directprep -->

<!-- narrowrest -->
Preferred standby now narrows the full arm/gripper frontal envelope inside the chassis. [Configuration](standby_preparation_config.json) · [Width and transition verification](iterations/20261120_narrowrest/README.md). Work home unchanged; one new-start full grasp example verified.
<!-- /narrowrest -->

<!-- restpose -->
Preferred startup now uses a [custom arms-down standby](standby_preparation_config.json), then the unchanged low work home. [Preparation and example validation](iterations/20261120_restpose/README.md). Old11 lift baselines retain their archived startup; new-start complete examples are separately indexed. Simulation only, no collection.
<!-- /restpose -->

<!-- symhome -->
Latest symmetric home: 11/12 full simulations verified, continuous preparation from shipped default. [Results and scope](iterations/20261120_symhome/README.md). Ideal torso only; no hardware or collection.
<!-- /symhome -->

<!-- caseyaw -->
Latest hard-case iteration: 12/12 common-start baselines verified. [Contact-layout and leveling tests](iterations/20261119_caseyaw/README.md). Data collection awaits review.
<!-- /caseyaw -->

<!-- commonfix -->
Latest common-start validation: 11/12 passed. [Remaining-object iteration](iterations/20261119_commonfix/README.md); collection awaits review.
<!-- /commonfix -->

<!-- commonstart -->
Common full initial joint pose and executed approach: 10/12 passed [validation](iterations/20261119_commonstart/README.md). Earlier pregrasp results remain historical. Training collection awaits video review.
<!-- /commonstart -->

<!-- liftpair-evaluation -->
两种箱体继续验证后，当前 12/12 种抬升候选拥有通过基线：[本轮对照结果](iterations/20261119_liftpair/README.md)。
<!-- /liftpair-evaluation -->

<!-- liftfix-evaluation -->
固定姿态迭代后 10/12 种抬升候选通过各自归档基线：[六种物体的调整结果](iterations/20261119_liftfix/README.md)。
<!-- /liftfix-evaluation -->

<!-- liftall-evaluation -->
固定姿态全物体抬升结果：6/12 候选通过当前检查，详见 [逐物体验证](iterations/20261119_liftall/README.md) 和 [当前训练配置](ground_lift_training_config.json)。
<!-- /liftall-evaluation -->

# G2 rigid-object interaction experiments

智元 G2 CRS + OmniPicker 的 Isaac Sim 4.2 仿真实验归档。机器人模型来自固定版本的 Genie Sim robot model；这里不是完整 Genie Sim 框架的迁移。入口独立于现有 R1 Lite 流程。

## 2026-11-19 更新

<!-- margin-pose -->
当前训练姿态已更新为 [-40,142,-87,0,0] 度，所有躯干关节至少保留约 10 度限位余量，肩高约 0.823 m。[筛选与物理验证](iterations/20261119_marginpose/README.md)；[当前训练配置](ground_lift_training_config.json)。下方旧姿态仅保留为历史对照。
<!-- /margin-pose -->

[固定低姿态、只训练双臂抬升的姿态筛选与验证](iterations/20261119_fixedpose/README.md)：腰部五关节 [-50, 150, -85, 0, 0] 度，肩高约 0.791 m、整体前倾 15 度，固定底盘/腰/头/轮组，仅 16 维手臂与夹爪动作。11 种物体通过几何路径审计，3 种代表物体各完成一次固定支撑物理抬升通过；其余未在该姿态完成物理重测。统一的是非手臂姿态，双臂仍按物体预抓取初始化。尚未开始训练。

## 2026-11-18 更新

<!-- push-iteration -->
[两把大椅子的推移验证与复现入口](iterations/20261118_push/README.md)：办公椅同初态3次推移通过；餐椅同初态3次推移通过。从预设低姿态开始，沿用1千克与理想躯干驱动；办公椅按刚体滑动测试，未建模脚轮滚动。推移与抬升分别统计。
<!-- /push-iteration -->

[后续交互方式迭代：镜像侧夹、曲面托底和接触力平衡](iterations/20261118_support/README.md)。本轮结果、完整下降与低姿态范围、毫米级位置偏移结果分别列出；两把椅子未计为成功，圆凳的真实软包接触仍待确认。

[时间步修正、4 种新增通过物体及完整试验记录](iterations/20261118/README.md)。历史累计 11/15 种有同初态三次通过记录；旧 7 种尚未全部按新时间步重测，剩余 4 种未完整通过。以下统计属于冻结旧快照。

## 原快照已验证范围

15 种刚体物体，63 次有效物理试验，另有 1 次取消/无效试验。7 种物体在相同初始条件下连续 3 次满足完整判据：`bin_lidded`、`bin_pedal`、`bin_round_lidded`、`basket_woven`、`basket_rectangular`、`stool_wood_four_leg`、`table_side`。最低保持抬升 15 cm，保持阶段倾斜不超过 20°，并通过对应策略的接触判据。带盖桶接近倾斜阈值，仍需提升裕量。

另外 6 种已测试但未通过完整判据，2 种椅子未找到可接受的规划。两个箱子虽能保持抬升，但未通过四点接触判据，不能计为完整成功。详见 `fixtures/strategy15_summary.json`。

试验统一使用 1 kg 刚体，躯干采用理想驱动，手臂和夹爪保留原驱动限制；没有物体焊接、物体位姿跟随或机器人基座锁定。这不是原始质量、真实躯干负载、搬起后起身/行走或实机鲁棒性验证。三次重复采用相同初态，不代表随机扰动成功率。

## 内容

- `snapshot/`：按测试时文件原样保留的 Python 源码，包括模型准备、几何分析、规划与物理抓取实验。保留历史路径和失败方案，不应将所有脚本当作推荐入口。
- `fixtures/`：15 物体目录、各策略规划、结果汇总与外部资产哈希。
- `results/`：有效和取消试验的结果、协议与取消原因；不包含大型轨迹和视频。
- `baselines.json`：7 种通过物体对应的脚本、规划目录和候选编号。
- `snapshot_manifest.json`：冻结文件的 SHA256；`upstream.json`：上游来源与固定提交。
- `workspace.py`：在新目录中准备实验、校验依赖与资产、运行已通过的基线。

早期 6 次试验的执行源码哈希与最终快照不同，汇总中的 `current_source_matches_executed_hash` 显式记录该差异，不能声称全部历史试验均由最终快照执行。

## 服务器复现入口

需要 Linux、Isaac Sim 4.2 对应 Python、CUDA GPU，以及 numpy/scipy/trimesh/python-fcl/OpenCV/USD。大型机器人模型、物体 USD、修复后的表面代理和 vendor 目录未提交；当前使用服务器上已准备好的资产包。源码和部分 USD 保留实验室绝对路径，因此尚不是任意机器上的独立安装包。

在仓库根目录运行（`WORK` 必须是全新的目录）：

```bash
WORK=/data20t/embodied/zkj/inter-nav-results/g2-replay-new
ASSETS=/data20t/embodied/zkj/inter-nav-results/g2-migration-20261111
PY=/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python
python3 research/rsi/g2_sim/verify_snapshot.py
python3 research/rsi/g2_sim/workspace.py prepare --workspace "$WORK" --assets-from "$ASSETS" --collision-deps /data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps
python3 research/rsi/g2_sim/workspace.py check --workspace "$WORK" --python "$PY" --verify-assets
python3 research/rsi/g2_sim/workspace.py run --workspace "$WORK" --python "$PY" --asset basket_woven --gpu 4 --repeat 1
```

准备步骤复制源码和规划，但将大型资产目录链接到资产包；不要在这个链接工作区运行资产生成/修改脚本。需要修改资产时先制作独立副本。重复试验使用新的 `--repeat` 编号；物理失败时入口返回非零退出码。运行结果写入工作区的对应 `*_trials` 目录。

## 来源与审阅

Genie Sim robot model 固定提交 `6d5b2159806ae0e64b432b638e9ee288d8bddc8f`，上游 Apache-2.0；驱动审计参考 Genie Sim 提交 `795b3fc77e4dd9f3ef972302ab90420c35450f62`。原始家庭物体来自 OpenRobotLab/GRScenes（CC-BY-NC-SA-4.0），获取和使用外部资产须遵守其许可证。具体 URL 见 `upstream.json`；不在本目录重新分发这些上游资产。

[飞书审阅文档：15 物体结论、13 个代表视频及审计附件](https://ocnprlp82ms4.feishu.cn/docx/LLihduA7VoWO8ExKS8dckhw9ncb#doxcn9ZQNONez54Wcm2cEDRpF1c)。文档需要相应访问权限。
