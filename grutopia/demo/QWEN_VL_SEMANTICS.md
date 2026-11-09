# Qwen-VL 物体分类接入（2026-09-27）

来源：[Kitty051201 / why-tvtest-20260927](https://github.com/Kitty051201/inter-nav-project/tree/why-tvtest-20260927)，参考提交 `0514c06`。本次按当前项目结构移植视觉分类链路，保留已有终点节点与实验记录工作。

## 原来有什么问题，如何改进

| 原来有什么问题 | 如何改进 |
| --- | --- |
| DINO 提示词及其输出短语承担物体类别判断；矩形屏幕候选可能对应电视、显示器、画框或镜子。 | DINO 只提供候选框，MobileSAM 提供 mask，Qwen3-VL-8B-Instruct 对框内物体分类。类别为 unknown 时记录判断，禁止写成目标证据。 |
| 环境词与目标词混在一个长提示词中；目标框可能被大量背景候选挤占。 | 对同一 RGB 分别做环境和目标两次 DINO 查询。目标 television 使用 `monitor . screen`；这两个词只生成候选，最终类别由 Qwen 判断。 |
| 对分割后去掉背景的裁剪分类，会损失上下文。 | Qwen 读取保留真实背景、增加 15% 边距并画红框的 RGB 裁剪；mask 与 RGB-D 仍负责物体世界坐标。裁剪最长边限制为 384。 |
| 重叠候选增加推理开销，同一帧可能反复增加节点证据。 | 目标候选优先，IoU ≥ 0.8 去重，最多分类 12 个候选。Qwen 来源同一节点同一帧最多计一次。 |
| 不同类别可能因图像特征相似而被合并。 | Qwen 类别仅按明确类别与精确别名关联节点；monitor 不等于 television。保留旧模式的标签融合逻辑。 |
| 视频只显示 DINO 短语，无法看见 Qwen 实际判断。 | 同步显示 `DINO 标签 -> Qwen 类别`，当前查询目标标红。现有 `groundingdino_detections.jsonl` 增加 classified：包含类别、模型、输出文本及是否作为地图证据。unknown 也记录。 |
| Qwen 模式取消了原有 CLIP 临时引导；远处目标被 Qwen 暂时分为 door、monitor 或 unknown 时，导航丢失了 DINO 已找到的方向。 | 增加独立的 DINO 目标观察候选。目标查询分支经 SAM/RGB-D 定位的框可以引导走近观察，Qwen 的类别与目标标签票数仍独立保存。已有 Qwen 目标节点优先接管导航。 |

## 目标候选引导（2026-09-27 追加）

- 只使用当前任务的 DINO 目标查询分支，不写死 refrigerator。television 的 monitor/screen 框也只是搜索候选，不能算电视的类别证据。
- 使用既有 DINO 分数门槛（默认 0.25）及有效 mask/depth；按 0.75m XY 距离关联观察位置，默认累计 2 个不同捕获步后引导。重复使用同一帧不会增加支持数，限频/失败帧不会提供新候选。
- 候选与语义场景图分开。即使 Qwen 给出 unknown，也可以保留搜索方向；该框仍不写入语义地图。候选自身不能设置 `target_confirmed`，到达候选点不能结束实验。
- 采用当前地图中已观察到且未被障碍膨胀覆盖的接近位置；保持当前候选，避免每帧换目标。走近后停下并朝向候选，在正常相机更新中继续收集证据，不追加即时模型调用。
- 候选 480 步未再次看到、导航尝试超过 1200 步，或近处观察 96 步仍没有语义目标接管，就释放并冷却 800 步，恢复前沿探索。冷却期间的框不能预先累计下一次锁定支持数。这些界限只约束临时候选，不增加最终目标的确认门槛。
- 默认开启。`--no-target-cues` 可关闭，用于与上一版 Qwen 导航对照；CLIP 和 Isaac 路径不使用这套候选引导。
- 地图以紫色菱形 `CUE: <查询物体>` 标记；trace 的 `goal_kind=target_cue` 与 `target_cue_id`、稀疏 events 的 selected/observing/released 事件记录候选过程。控制台仍只输出原有简短进度。

## 与当前终点节点工作的关系

本次沿用当前终点标签累计证据要求和到达距离阈值。**没有加入接近目标后再收集两帧的复核门槛，也没有移植新的不可达切换机制。** Qwen 分类决定每帧的物体类别，既有终点逻辑负责判断证据是否足够、机器人是否到达。

原有文本 Qwen3-8B 仍负责探索前沿评分；新增 Qwen3-VL-8B-Instruct 服务专门看图分类。`--no-qwen` 关闭前者，不会关闭视觉分类。

## 启动

DINO 与 MobileSAM 继续使用现有 12181、12183 服务。Qwen-VL 使用独立 Python 环境，避免升级旧感知环境影响 DINO。该环境必须能够导入 `transformers.Qwen3VLForConditionalGeneration`，并已安装本地 Qwen3-VL-8B-Instruct 权重；服务不会自动下载模型。模型用法参考[官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

```bash
# 将变量设为实际的独立环境与本地权重目录
$QWEN_VL_PYTHON grutopia/demo/serve_semantic_perception.py qwen-vl \
  --qwen-vl-model "$QWEN_VL_MODEL" --device cuda:3 --port 12185

curl -fsS http://127.0.0.1:12185/health

$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --detection-mode open_vocab --semantic-classifier qwen-vl \
  --target television --gpu 0 --qwen-device cuda:2
```

默认演示使用 `open_vocab` 与 `qwen-vl`。启动前检查三个模型服务。显式选择 `--semantic-classifier clip` 可使用原有 DINO 标签与 CLIP 特征链路。`--detection-mode isaac` 保留纯真值模式。

## 记录与验证边界

继续使用既有独立运行目录、run manifest、summary、events、trace、视频帧索引及目标地图标记；分类结果写入现有检测 JSONL，控制台不逐框输出。

基础 Qwen 版本 `c3e2539` 已在服务器运行一次：`qwen_vl_refrigerator_20260927_131201` 在第 2728 步到达，Qwen 目标节点有 65 次 refrigerator 观察。但与 9 月 25 日成功实验相比用步数更多，不能宣称 Qwen 已提升导航效率。

本次候选引导测试覆盖独立候选几何、unknown 不入图、重复帧不累计、候选不能触发成功、超时/冷却退出、语义目标接管和独立记录。本地和服务器的 42 项相关检查通过。回放基础 Qwen 版本检测记录，在第 72/96 步已有同一区域的 DINO 目标候选（Qwen 分别判为 door/cabinet），而语义目标到第 792 步才达到选择要求。回放只证明搜索方向证据能更早保留：旧记录的 unknown 框没有三维坐标，未纳入；未回放历史占据地图或新的运动轨迹，不能把第 96 步当作实际导航切换时刻。

## 完整导航实验（2026-09-27）

候选引导版本 `f7eab59` 已完成运行 `qwen_vl_target_cues_20260927_140312`。目标为 refrigerator，使用同一 GRScene 配置、open_vocab、Qwen-VL 和 1.10m 接近点到达阈值，12000 步上限。正常退出，视频及记录完整。

| 指标 | 基础 Qwen 版本 | Qwen + 候选引导 |
| --- | ---: | ---: |
| 到达步数 | 2728 | 2458 |
| 首次候选引导 | 无 | 72 |
| 首次选择 Qwen 目标节点 | 792 | 696 |
| 首次满足目标确认要求 | 1344 | 840 |
| 最终 refrigerator 标签支持 | 65 | 70 |
| XY 轨迹长度（含步态抖动） | 8.035m | 7.782m |
| 重规划次数 | 6 | 3 |
| 感知 / 规划失败 | 0 / 0 | 0 / 0 |

本轮第 72 步的候选由 Qwen 判为 door，但保留了搜索方向；第 696 步交给 Qwen 目标节点，第 840 步达到原有 8 次标签证据要求。候选自身没有触发成功。本轮没有触发候选超时或冷却，因此这些退出分支的验证仍来自接口测试。

与基础 Qwen 版本相比，本轮步数减少 9.9%，轨迹长度减少 3.1%。两轮接近点分别为 [10.05, -1.85] 和 [10.05, -1.65]；最终距目标中心分别为 1.295m 和 1.467m。因此，较少步数不代表最终物理位置更靠近目标，也不能把全部步数变化归因于候选引导。与 9 月 25 日旧版成功运行（2365 步）相比，本轮仍多 93 步。以上均为单次实验，未固定仿真随机种子，尚不能证明普遍成功率提升或整体导航超过旧版。
