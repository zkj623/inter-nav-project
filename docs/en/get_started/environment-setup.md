# Environment setup


> The default demo now uses Qwen3-VL-8B-Instruct for object classification. Start the additional port 12185 service in its own environment; see [Qwen-VL integration](../../../grutopia/demo/QWEN_VL_SEMANTICS.md). Use `--semantic-classifier clip` for the legacy perception setup below.

If you are on this lab server and your account is in the `embodied` group,
skip this page and use the shared conda, weights, and scenes:

[lab-shared-environment.md](lab-shared-environment.md)

Otherwise follow the sections below in order. After the last step you can run
the Go2 demo in the [README](../../../README.md).

## 1. Hardware and software

| Item | Requirement |
| --- | --- |
| OS | Ubuntu 20.04 or 22.04 |
| GPU | NVIDIA RTX 2070 or newer. Full semantic mode is comfortable on 2–3 GPUs (24 GB class). One GPU is enough for the geometry-only smoke test. |
| Driver | 535 or newer (535.129.03 is the Isaac Sim 4.2 recommendation) |
| RAM | 32 GB or more |
| Disk | about 25 GB for Qwen3-8B + GroundingDINO + CLIP + MobileSAM, plus Isaac Sim |
| Simulator | [Isaac Sim 4.2.0](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) only. Do not use 4.1 or 4.5. |
| Python | 3.10, provided by Isaac Sim |
| Conda | Miniconda or Anaconda |

Isaac Sim and Qwen3 need different PyTorch / Transformers versions. This repo
uses **two conda environments**:

- `grutopia`: Isaac Sim + this package (simulation, mapping, Go2 policy, CLIP)
- `semexp`: GroundingDINO, MobileSAM, Qwen3-8B worker

## 2. Install Isaac Sim 4.2.0

1. Install the NVIDIA driver and verify `nvidia-smi` works.
2. Install [Omniverse Isaac Sim 4.2.0 (workstation)](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).
3. Confirm the install contains `isaac-sim.sh` and `python.sh`. The default
   path looks like:

```text
~/.local/share/ov/pkg/isaac-sim-4.2.0
```

If Isaac already runs from an Isaac Lab conda env (`isaaclab`), you can skip
section 4 and point `ISAAC_PYTHON` at that env's `python`. It must still be
Isaac Sim **4.2**.

## 3. Clone this repository

```bash
git clone https://github.com/zkj623/inter-nav-project.git
cd inter-nav-project
```

All later commands are run from this directory.

## 4. Create the Isaac / simulation environment

This links a conda env to Isaac Sim and installs the Python package.

```bash
# Needs conda on PATH. The script asks for the Isaac Sim folder
# (the directory that contains isaac-sim.sh) and the env name.
./setup_conda.sh
```

Accept the default env name `grutopia`, or type another name. Then:

```bash
conda activate grutopia
python -m pip install -r requirements/runtime.txt

# Confirm Isaac Sim imports
python -c "import omni.isaac.kit; print('isaac ok')"
```

If you already have a working Isaac 4.2 Python (for example Isaac Lab):

```bash
export ISAAC_PYTHON=/path/to/isaaclab/bin/python
$ISAAC_PYTHON -m pip install --no-deps -e .
$ISAAC_PYTHON -m pip install -r requirements/runtime.txt
```

In later commands, `python` means the Isaac env. You can also set:

```bash
export ISAAC_PYTHON="$(which python)"   # after conda activate grutopia
```

## 5. Create the model environment

Do **not** install these packages into the Isaac env.

```bash
conda create -y -n semexp python=3.10
conda activate semexp
python -m pip install -r requirements/semantic-exploration.txt
```

`requirements/semantic-exploration.txt` installs recent `torch`,
`transformers`, MobileSAM, Flask, and Hugging Face Hub. If the default PyTorch
wheel does not match your CUDA driver, install a CUDA build from
[pytorch.org](https://pytorch.org/get-started/locally/) first, then rerun the
requirements file.

```bash
export QWEN3_PYTHON="$(which python)"   # after conda activate semexp
```

Keep this variable set in every terminal that starts a model service or the
Go2 demo.

## 6. Download assets and models

Nothing under `grutopia/assets/` is committed. From the repo root:

```bash
conda activate grutopia
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# GRScenes home used by the default profile (~ tens of MB after extract)
python grutopia/demo/download_mv7_scene.py

# Official Unitree Go2 USD (needs Isaac / Nucleus access)
python grutopia/demo/download_go2_asset.py

# isaac-go2-ros2 rough-terrain policy (SHA-256 verified)
python grutopia/demo/download_go2_policy.py
```

```bash
conda activate semexp
# CLIP, GroundingDINO, MobileSAM, Qwen3-8B (about 20 GB)
python grutopia/demo/download_semantic_exploration_models.py
```

If Hugging Face rate-limits you, run `huggingface-cli login` in `semexp`
first. If Nucleus cannot fetch the Go2 USD, place `isaaclab_go2.usd` at the
path below yourself.

Expected layout:

```text
grutopia/assets/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd
grutopia/assets/benchmark/meta/MV7J6NIKTKJZ2AABAAAAADA8_usd/object_dict.json
grutopia/assets/robots/go2/isaaclab_go2.usd
grutopia/assets/robots/go2/policy/move_by_speed/rough_model_7850.pt
grutopia/assets/models/mobile_sam.pt
```

The scene license is
[CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenRobotLab/GRScenes).

## 7. Start perception services

Open two terminals in the **model** env. Pick a free GPU (example: `cuda:1`).

```bash
cd /path/to/inter-nav-project
conda activate semexp
export QWEN3_PYTHON="$(which python)"

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py grounding-dino \
  --host 127.0.0.1 --port 12181 --device cuda:1
```

```bash
cd /path/to/inter-nav-project
conda activate semexp
export QWEN3_PYTHON="$(which python)"

$QWEN3_PYTHON grutopia/demo/serve_semantic_perception.py mobile-sam \
  --host 127.0.0.1 --port 12183 --device cuda:1 \
  --mobile-sam-checkpoint grutopia/assets/models/mobile_sam.pt
```

Wait until each process prints that Flask is running. Optional check:

```bash
curl -s http://127.0.0.1:12181/health
curl -s http://127.0.0.1:12183/health
```

If these services are down, exploration still runs with Isaac semantic labels.
If Qwen3 fails to start, frontier ranking falls back to geometric scoring.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `omni.isaac` import fails | Isaac Sim 4.2 is not sourced. `conda activate grutopia` after `./setup_conda.sh`, or use Isaac's `python.sh`. |
| `GRScenes navigation USD not found` | `download_mv7_scene.py` did not finish. Confirm the `.usda` path in section 6. |
| Go2 USD missing | Run `download_go2_asset.py` from the Isaac env, or copy `isaaclab_go2.usd` into `grutopia/assets/robots/go2/`. |
| CUDA OOM | Give Isaac, the two HTTP services, and Qwen different GPUs. Or run `--no-qwen` / `--detection-mode isaac`. |
| Qwen worker never starts | `QWEN3_PYTHON` must be the `semexp` interpreter, not Isaac Python. |
| No detections | GroundingDINO / MobileSAM not listening on `12181` / `12183`. |
