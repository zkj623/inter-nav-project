# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, and semantic
Voronoi exploration.

The default demo places Go2 in a GRScenes home. It explores until it matches
a text target such as `refrigerator` and walks up to the scanned object.

## Go2 可交互箱子场景导航

原尺寸 MV7 中的点到点导航、交互决策和侧推清路入口为 `grutopia/demo/go2_side_clearance_navigation.py`。
以下命令在本实验室服务器上运行 T05 并保存完整视频；在 `codex/go2-dynamic-interaction-20261026` 分支的仓库根目录执行，GPU 编号按空闲情况修改。

```bash
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
  --push-max-speed 0.95 --turn-speed 1.0 \
  --contact-centering-speed 0.25 --contact-yaw-tolerance 0.10 --contact-front-gap 0.40 \
  --output "$go2_out" 2>&1 | tee "${go2_out}.log"
```

只跑两组推箱任务时将 `--task-ids T05` 改为 `--task-ids T04,T05`；全部五组使用 `--task-ids T01,T02,T03,T04,T05`。
输出目录的 `T05/video/combined.mp4` 是完整四画面录像，`third_person.mp4` 为随狗朝向转动、狗头朝上的正上方视角；各组任务独立保存 25 fps 视频。

地图使用 1 Hz 全局规划层和 10 Hz 局部动态层。箱子位移后清除其旧占据，保留墙体和其他物体；清路完成会立即刷新全局路线。录像地图同时显示推箱原因、侧停方向、预测路径收益和局部碰撞图。设计与限制见 [分层交互导航说明](docs/go2_layered_navigation.md)。

本轮 **T04 和 T05 已使用同一版 v19 代码完成原始起终点任务**：停稳后终点误差分别为 0.124 m 和 0.086 m，完整机身穿门，全部导航、地图、健康及录像审计通过。每组五路完整录像为 25 fps，T04 各 3820 帧、T05 各 2215 帧。v19 普通回归中 T01、T03 已完成，T02 在急转弯时失稳；v20 停车转向未解决失稳且已回退，v21 guarded 复测保持稳定但重规划衔接失败，当前 v22 修复后正在完整复测；详见 [本轮正式结果](HANDOFF.md#go2-layered-results)。请查看每组 `result.json` 的 `success`/`reason`、终点误差与 `rollout_health`。启动时换过起点的 `calibration_only` 结果不能作为原任务成功记录。完整材质场景占用较多内存，同一时间只启动一个仿真进程。`--task-ids` 列表会在同一场景中逐组复位机器人、全部箱子和地图，并各自保存录像。
当前 v21 对比将命令中的 `--path-controller pure` 改为 `--path-controller guarded`，使用框架已有的朝向误差平滑减速和全机身运动检查。原 Go2 policy、Voronoi 图、场景尺寸、终点和健康阈值保持原设置。v21 在全程健康的情况下因全局膨胀区起点衔接失败而停止；v22 修复安全恢复连接点搜索和实际跟踪目标同步，完整实跑仍在验证，尚不能宣称五组统一通过。

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
