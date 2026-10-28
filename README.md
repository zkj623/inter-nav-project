# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, and semantic
Voronoi exploration.

The default demo places Go2 in a GRScenes home. It explores until it matches
a text target such as `refrigerator` and walks up to the scanned object.

## Go2 在线建图、规划与交互导航

未知地图版本的入口是 `grutopia/demo/go2_online_interactive_navigation.py`。
当前开发测试默认使用 `--interaction-source ground_truth`，从布局提前获知哪些箱子支持推；空间占据地图仍在线建立。需要恢复视觉识别时显式使用 `--interaction-source vision`。
它保留原尺寸 Go2 / MV7、1 cm 地面、原始起终点和物理侧推；每个任务从空占据地图开始。
墙体地图由在线 LiDAR 建立，目标尚不可达时沿 Voronoi 路线探索可达边界。
全局层 1 Hz 更新，局部层 10 Hz 更新，箱子移开后的空间需要新射线确认才会通行。
在线模式默认使用固定 200 Hz 物理步长的 DC 电机控制、50 Hz 原 Go2 policy 和 25 fps 录像。
当前五组统一物理验收尚未完成；各版本成功及失败结果见下方在线导航说明。
当前候选修复了 Go2 全关节物理设置的作用范围、辅助地面接触参数的初始化时机，并采用可启动步态的接触速度和基于实测速度的提前制动。原始 T04/T05 仍须整场复跑验收；局部推箱对照不计为任务完成。

在实验分支 `codex/go2-online-smooth-20261027` 的仓库根目录执行：

```bash
source /data20t/embodied/share/inter-nav/env.sh
export MALLOC_ARENA_MAX=2
export PYTHONPATH="$PWD:$PYTHONPATH"
go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-online-$(date +%Y%m%d-%H%M%S)"
"$ISAAC_PYTHON" -u grutopia/demo/go2_online_interactive_navigation.py \
  --benchmark-dir "$go2_inputs" --output "$go2_out" \
  --task-ids T01,T02,T03,T04,T05 --gpu 1 --max-steps 40000 \
  --floor-top .01 --map-source online --interaction-source ground_truth --path-controller continuous --navigation-attitude-recovery \
  --route-planner voronoi --navigation-heading path \
  --roadmap-shape circle --route-clearance .30 --navigation-half-width .22 \
  --voronoi-clearance-weight .30 \
  --global-map-hz 1 --local-map-hz 10 --local-map-radius 3 \
  --push-start-speed .60 --push-max-speed .95 --turn-speed 1.0 --contact-centering-speed .25 --contact-lateral-speed .35 \
  --contact-yaw-tolerance .10 --contact-front-gap .40 \
  --record-video --material-mode preview \
  --third-person-height 1.35 --third-person-forward .25 --third-person-fov 100
```

`--task-ids T04` 可单独运行一组。完整视频为 `<输出目录>/T04/video/combined.mp4`，
地图灰色为未观测区域，绿色起点和红色目标始终标在请求坐标。
不录像的物理诊断可去掉 `--record-video` 并使用 `--material-mode simple`。
完整材质场景加载时间较长，建议保留至少 78 GiB 可用内存和 14 GiB 显存。

上方命令是当前交互控制的复测配置。已验收的普通导航中，T01 采用 DC 控制，T02/T03 保留
实测更平稳的原 PD 控制版本；对应源码、视频及尚未通过的试验见下方在线导航说明。
这不是同一套配置五组统一通过的声明。侧移与前后移动的请求分开配置：独立物理试验中
0.25 m/s 纯侧移未形成有效迈步，当前侧移采用已测可起步的 0.35 m/s，并结合提前制动。
持续侧移片段可跟踪该速度，但短距离纠偏和动作切换仍需改进；当前已加入受碰撞检查约束的平滑减速、必要时的航向精调和实测停稳窗口，T04/T05 物理验收尚未通过。

在线模式仅从输入目录读取仿真场景配置、箱子布局和任务坐标，不把 `navigation_grid.npz`、
`original_metadata.json` 或 `passages.json` 交给规划器。箱子身份/尺寸/位姿仍是仿真检测接口，
只有当前 LiDAR 命中后才加入几何规划。当前交互资格使用布局真值；视觉模式则要求实际 RGB-D 关联及 Qwen3-VL 两帧确认。定位及几何跟踪仍沿用仿真接口。
视觉识别、白名单、技能接口与运行命令见 [视觉推箱 skill](docs/go2_visual_push_skill.md)。
设计、证据和验证状态见 [在线导航说明](docs/go2_online_navigation.md)。

## 历史结果：使用已知全局地图的箱子导航

以下命令和视频来自先前的**预生成全局碰撞地图**版本，不能作为上方未知地图版本的验证结果。

原尺寸 MV7 中的点到点导航、交互决策和侧推清路入口为 `grutopia/demo/go2_side_clearance_navigation.py`。
以下命令在本实验室服务器上按已验证的 v19 源码运行 T05 并保存完整视频；在仓库根目录执行，GPU 编号按空闲情况修改。实验分支 `codex/go2-dynamic-interaction-20261026` 保存相关提交；复现请使用下方列出的确切源码版本。

```bash
git switch --detach e028847
source /data20t/embodied/share/inter-nav/env.sh
export MALLOC_ARENA_MAX=2

go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-interactive-nav-$(date +%Y%m%d-%H%M%S)"

set -o pipefail
"$ISAAC_PYTHON" -u grutopia/demo/go2_side_clearance_navigation.py \
  --benchmark-dir "$go2_inputs" \
  --task-ids T05 \
  --gpu 1 \
  --floor-top 0.01 \
  --max-steps 30000 \
  --record-video --material-mode preview \
  --route-planner voronoi --navigation-heading path \
  --path-controller pure --roadmap-shape circle --route-clearance 0.30 \
  --voronoi-clearance-weight 0.30 --navigation-half-width 0.22 \
  --global-map-hz 1 --local-map-hz 10 --local-map-radius 3 \
  --third-person-height 1.35 --third-person-forward 0.25 --third-person-fov 100 \
  --push-start-speed 0.60 --push-max-speed 0.95 --turn-speed 1.0 \
  --contact-centering-speed 0.25 --contact-yaw-tolerance 0.10 --contact-front-gap 0.40 \
  --output "$go2_out" 2>&1 | tee "${go2_out}.log"
```

`--task-ids` 支持逗号分隔的列表；但本轮五组成功录像采用下述两套源码/配置，不能将某一套参数直接标作五组统一验证通过。
输出目录的 `T05/video/combined.mp4` 是完整四画面录像，`third_person.mp4` 为随狗朝向转动、狗头朝上的正上方视角；各组任务独立保存 25 fps 视频。

地图使用 1 Hz 全局规划层和 10 Hz 局部动态层。箱子位移后清除其旧占据，保留墙体和其他物体；清路完成会立即刷新全局路线。录像地图同时显示推箱原因、侧停方向、预测路径收益和局部碰撞图。设计与限制见 [分层交互导航说明](docs/go2_layered_navigation.md)。

已整理五组**个体审计通过**的原始任务完整录像：T01/T03/T04/T05 使用 `e028847` + `pure`，T02 使用 `f7e33cf` + `guarded --navigation-attitude-recovery`。T04/T05 完整机身穿门并到达原目标，停稳误差分别为 0.124 m、0.086 m；T02 正式完整材质录像为 5400 步、误差 0.167428 m，五路视频各 1132 帧、25 fps。全部25段选定视频均已逐帧解码并保存于 E 盘。具体路径、版本及限制见 [五组已验证录像](docs/go2_verified_runs.md) 和 [HANDOFF 结果](HANDOFF.md#go2-layered-results)。

这不是同一配置五组统一通过：v23 的 T01 在第 3867 步失稳；T02 完成后已停止该统一试验，T03/T04/T05 的该批次记录为 `stop_requested`，不选作交付结果。T02 正式运行全程健康，但最高倾角 29.929° 接近 30° 阈值；新配置尚未证明普遍稳定，四组继续采用已验证的 v19 录像与配置。箱子身份/尺寸/位姿由仿真提供。

复现 T02 时，将上方命令的源码版本 `e028847` 改为 `f7e33cf`，`--task-ids T05` 改为 `--task-ids T02`，并将 `--path-controller pure` 改为 `--path-controller guarded --navigation-attitude-recovery`。源码快照逐文件哈希已核对到这两个提交。当前代码自动测试为 326 passed、1 deselected；未运行的一项 G1 测试依赖本分支不含的人形配置。

外部场景、布局/碰撞数据及模型需要服务器共享环境，详情、五组起终点和已知失败阶段见 [HANDOFF：Go2 可交互箱子导航](HANDOFF.md#go2-interactive-navigation)。

## Environment setup

On this lab server, `embodied` members should clone the repo and use the
shared conda / weights / scenes:

[docs/en/get_started/lab-shared-environment.md](docs/en/get_started/lab-shared-environment.md)

On any other machine, install Isaac Sim 4.2, the two conda environments,
scene/robot assets, and perception services by following
[docs/en/get_started/environment-setup.md](docs/en/get_started/environment-setup.md).

## Run Go2 navigation

On this lab server, after `source /data20t/embodied/share/inter-nav/env.sh`:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator
```

Do not set `CUDA_VISIBLE_DEVICES`. Use `$ISAAC_PYTHON`, not `python`.

On any other machine, from the repo root in the Isaac env:

```bash
export PYTHONPATH="$PWD"
export QWEN3_PYTHON=/path/to/miniconda3/envs/semexp/bin/python

python grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --target refrigerator
```

`--gpu` is Isaac Sim, `--perception-gpu` selects CLIP in the legacy classifier mode, and `--qwen-device` is the
Qwen3-8B worker. Pick free GPUs on your machine. Change `--target` to
`chair` or `plant` if you want a different object.

## Detection modes

`--detection-mode` selects which perception source feeds the semantic scene
graph. All three modes run the same mapping, Voronoi exploration, planning, and
locomotion stack; only the origin of the object detections changes.

| Command | Detections | Use it for |
| --- | --- | --- |
| `--detection-mode isaac` | Isaac Sim semantic boxes only (ground truth) | Geometry and planner debugging, an oracle upper bound on target recall, runs without any VLM service |
| `--detection-mode open_vocab` | GroundingDINO + MobileSAM + Qwen-VL (default) | Measuring what the robot actually perceives, with no ground-truth leakage |
| `--detection-mode hybrid` | Both, fused per object | The full open-vocabulary exploration method |

**1. Isaac ground truth only** — one GPU, no GroundingDINO, no MobileSAM, no Qwen:

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --detection-mode isaac \
  --no-qwen \
  --target refrigerator
```

**2. Open vocabulary only** — requires DINO, SAM, and the Qwen-VL HTTP service:

Start the services in two separate terminals (replace `cuda:1` with the
perception GPU you selected):

```bash
$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py grounding-dino \
  --host 127.0.0.1 --port 12181 --device cuda:1
```

```bash
$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py mobile-sam \
  --host 127.0.0.1 --port 12183 --device cuda:1 \
  --mobile-sam-checkpoint grutopia/assets/models/mobile_sam.pt
```

Start Qwen-VL in an independent environment with local weights (see
[Qwen-VL setup and integration notes](grutopia/demo/QWEN_VL_SEMANTICS.md)):

```bash
$QWEN_VL_PYTHON grutopia/demo/serve_semantic_perception.py qwen-vl \
  --qwen-vl-model "$QWEN_VL_MODEL" --device cuda:3 --port 12185
```

Verify all three endpoints before starting Isaac:

```bash
curl -fsS http://127.0.0.1:12181/health
curl -fsS http://127.0.0.1:12183/health
curl -fsS http://127.0.0.1:12185/health
```

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --detection-mode open_vocab \
  --target refrigerator
```

**3. Hybrid** — ground-truth labels fused with open-vocabulary detections
(choose this mode explicitly):

```bash
$ISAAC_PYTHON grutopia/demo/go2_semantic_exploration.py \
  --gpu 0 \
  --perception-gpu 1 \
  --qwen-device cuda:2 \
  --detection-mode hybrid \
  --target refrigerator
```

How the three modes differ at runtime:

- `isaac` ignores the open-vocabulary stack entirely; GroundingDINO is never
  queried, so `open_vocabulary_frames` stays `0`.
- `open_vocab` never falls back to simulator labels. It checks GroundingDINO,
  MobileSAM, and Qwen-VL (or CLIP when explicitly selected) before Isaac starts, and rejects an unavailable
  stack with an explicit error. If a ready service later fails three
  consecutive RGB queries, the run stops instead of exploring with an empty
  semantic graph.
- `hybrid` merges both sources per object: a scene-graph node keeps every
  source that agreed on it and geometry comes from the denser observation.
  It reports a failed preflight and degrades to Isaac-only when the services
  are unreachable.

Recorded runs identify their source, so results stay comparable: every
scene-graph node carries `sources` (`isaac`, `open_vocabulary`, `qwen_vl`, or their combinations) and
every statistics block reports `semantic_detection_mode`.
For degraded hybrid runs, `semantic_detection_effective_mode`,
`open_vocabulary_available`, and `open_vocabulary_startup_error` preserve what
actually ran in the saved JSON.
Saved statistics also expose attempts, completed queries, detections,
partial candidate failures, the last detected labels, and the last exception.
Periodic console progress stays compact.

An empty programmatic room is available with `--scene programmatic`.

A finished run prints `semantic_exploration_result` with `"success": true`
and writes to a new timestamped directory under `grutopia/results/`:

- `combined.mp4`, `robot_rgb.mp4`, `third_person.mp4`, `map_topdown.mp4`
- `groundingdino.mp4` and `groundingdino_detections.jsonl` when the detector is queried
- matching `*_preview.png`
- `run_summary.json` (terminal status, target confirmation, arrival distances)
- `manifest.json`, `events.jsonl`, `trace.csv`, `progress.json` (comparison and diagnosis)
- `video_frames.jsonl` (regular video frame-to-step index)
- `final_map.json` (Voronoi graph, decisions, trajectory)
- `final_map.npz` (occupancy layers and skeleton)

If `--map-output` is omitted, it automatically resolves to
`<record-dir>/final_map`, so videos and map metadata cannot silently land in
different run directories.
An explicit `--record-dir` must be new; the runner refuses to overwrite an
earlier experiment. To build a CSV across runs:

```bash
python grutopia/demo/summarize_semantic_runs.py grutopia/results \
  --output grutopia/results/semantic_runs.csv
```

For staged offline, geometry-only, and fused semantic validation, follow the
[navigation and semantic regression test plan](docs/navigation-semantic-regression-test-plan.md).
The [September 19–25 navigation report](docs/reports/go2-open-vocabulary-weekly-2026-09-19-to-25.md)
compares the earlier failures with the label-fusion rerun.

## Architecture

1. `SemanticVoronoiGraph` builds a safe medial-axis skeleton on observed free
   space, plus room-like regions and doorways.
2. `OpenVocabularyPerception` sends RGB to GroundingDINO + MobileSAM, then
   Qwen3-VL-8B-Instruct classifies marked context crops. The legacy CLIP mode
   remains available through `--semantic-classifier clip`. `--detection-mode` picks between these detections,
   the Isaac simulator labels, or a fusion of both.
3. `AdaptiveExplorationPlanner` ranks frontiers. A local Qwen3-8B worker may
   rerank the bounded topology JSON. The model never sends locomotion
   commands; A* and the Go2 RSL policy still move the robot.

## License

Simulation code follows the original GRUtopia MIT license. GRScenes assets
remain CC BY-NC-SA 4.0 and must be downloaded separately.
