# inter-nav-project

Go2 open-vocabulary semantic exploration on an unknown occupancy map. This
repository is a self-contained slice of the InternUtopia / GRUtopia stack:
Isaac Sim runtime, Unitree Go2 locomotion, RGB-D/LiDAR mapping, and semantic
Voronoi exploration.

The default demo places Go2 in a GRScenes home. It explores until it matches
a text target such as `refrigerator` and walks up to the scanned object.

## R1 Lite policy–skill iteration research

The experimental RSI module is separate from the default navigation demos:

- `grutopia_extension/interaction_clearance/` and `manipulation/`: contact-only
  box-clearance tasks, composable physical policies, skill execution, and safety checks.
- `research/rsi/`: skill candidate evolution, frozen multi-layout evaluation,
  clearance benchmark configurations, data export, and success/failure video tools.
- The existing navigation demos and their default task behavior are unchanged.

The current expert-initialized version passes 12/12 original configurations and
30/30 development states, but only 56/60 independent validation states. Automatic
bimanual policy learning and the complete two-way policy–skill learning loop are
not implemented; this is not a claim of stable success across all scenes.

See [the RSI research README](research/rsi/README.md) for the architecture,
commands, dataset boundaries, verified results, and remaining work.

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
