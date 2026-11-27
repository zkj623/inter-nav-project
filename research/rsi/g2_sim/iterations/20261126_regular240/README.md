# Regular 240-episode pi0.5 experiment

Started 2026-11-26 at 21:13:23 server local time. Initial status: preparing
dataset, not yet optimizing the model. Supervisor PID at launch: 2765708.

Use only the accepted `collection_state` records: 12 objects, 20 episodes each.
Do not include the 31 wider-placement episodes or learned-policy rollouts.
Verify source hashes, existing physical/three-camera acceptance, shapes, joint
order, finite values and timestamps. Require all 240 episodes to pass; do not
silently train a smaller subset. Reuse only caches with matching source hashes.
Image conversion is serial to bound RAM and disk pressure.

Split at episode/placement level with deterministic seed 20261127 plus object
index: 18 training and 2 validation episodes per object. No commanded placement
may overlap the splits. This yields 216 training episodes / 238645 frames and
24 validation episodes / 26488 frames. The collection covers small translations
of approximately +/-1 cm with fixed object yaw; it does not test new objects or
new rooms. Validation is independent of this new training run, not a new dataset
that was never used during earlier development.

Initialize from official pi05_base, with LoRA on language/action expert and the
visual backbone frozen. Recompute training-only normalization for all 12 objects.
Inputs remain language, head/left-wrist/right-wrist RGB, and qpos16. Predict 14
arm deltas relative to current measured qpos plus 2 absolute gripper commands.
Action chunk 50, data 20 Hz. Object-balanced episode/time sampling. Limit open
HDF5 cache handles to 16. Four-device JAX discovery was verified before training.

Training GPUs: physical 0, 2, 4, 6; global batch 8 (2 per GPU), OpenPI default
data parallelism with fsdp_devices=1. GPU 1 and 3 were occupied at inspection.
Six cards were essentially idle. Require at least 22000 MiB free on each selected
training card before launching. Training and policy evaluation run sequentially.

Stages: ceil(train_frames * epochs / 8) for epochs 1, 3, 5: 29831, 89492,
149154 optimizer steps. Checkpoints every 5000 steps, synchronous writes and
retention of the newest live checkpoint, with verified stage backups. The fixed
cosine schedule spans the full five-epoch run, peak LR 5e-5, final LR 5e-6;
resume optimizer/model state between stages. Equivalent epoch counts are nominal
sample-count equivalents because episodes and object groups are sampled randomly.

After each stage, evaluate the first held-out placement for every object:
12 continuous-policy rollouts. No target freezing at 10 cm and no height-based
early success stop. The final 10.1 seconds are labeled for auditing only; the
policy keeps its existing action queue and receives no phase/object-state input.
Base contact alone is not failure. Report complete mesh floor clearance separately
from strict final-window height, relative-motion and orientation checks.
The remaining physical/contact safety stops remain active.

Time budgets come from all 240 raw HDF5 timestamp/phase arrays:
ceil to 5 seconds of max(longest demonstration duration for the object,
latest demonstration hold-start time + 10.1 seconds). This covers the original
demonstrated action duration and allows a longer observation window.
Whole-dataset duration: min 41.55, median 56.425, p95 69.6875, max 82.5 seconds.
Most demonstrations hold for 2.95 seconds; the side table holds for 11.95 seconds.

| Object | Fixed policy horizon (s) |
| --- | ---: |
| basket_rectangular | 65 |
| basket_woven | 70 |
| bin_lidded | 70 |
| bin_pedal | 55 |
| bin_round_lidded | 60 |
| bin_round_pedal | 75 |
| box_reinforced | 90 |
| box_white | 70 |
| case_handheld | 80 |
| stool_rectangular | 55 |
| stool_wood_four_leg | 60 |
| table_side | 65 |

Offline probes use held-out demonstration states. Only the first of the two
held-out episodes per object is automatically evaluated. Different checkpoints
share commanded placements, sampling seeds and object-specific horizons, but
not exact simulator-state forks. This does not test chassis transport.
Initialization/progress timeouts allow one infrastructure retry; physical
failure is not retried.

The already-running supervisors still pass their legacy lock10 argument.
evaluate_pi05_regular240.py explicitly overrides it to full; all newly generated
trial records and summaries use full_lift_v1. Archived previous experiments are
unchanged. Twelve harnesses passed syntax/no-lock/202-observation-action checks.
Metric checks cover ten-second acceptance, rejection of a five-second window,
and a late drop. A full simulation smoke test remains pending the first checkpoint.

Monitor resources every 10 seconds; stop owned children for sustained memory,
swap-I/O, disk, GPU-memory or temperature pressure. Existing 2 GiB swap occupancy
alone is not treated as active thrashing. No unrelated processes are stopped.

Remote workspace:
`/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124`

- Supervisor: `run_pi05_regular240.py`
- Status/logs: `pi05_regular240_20261126/`
- Frozen input manifest once ready: `datasets/regular240_v1/manifest.json`
- Run: `regular240_relative_v1`
- Scope: `pi05_regular240_relative_v1_scope.json`
- Live checkpoints: `/tmp/zkj_g2_20261125_checkpoints/g2_pi05_lora/regular240_relative_v1/`
- Stage reviews: `pi05_focus_20261125/reviews/pi05_regular240_<step>_heldout/`

This document records the launch configuration, not completed training results.

## Extension requested before training began

Total training extended to 20 equivalent epochs, with evaluations at 1, 3, 5,
10, 15 and 20 epochs. The original supervisor continues preparation and stages
1/3/5. A detached continuation supervisor waits for that exact process to exit
and verifies successful completion before resuming the same optimizer/model
state for stages 10/15/20. It does not interrupt conversion or run overlapping
trainers. If the initial supervisor fails, continuation stops for diagnosis.

The trainer reads extended_training_plan.json before configuring its schedule.
Cosine decay now spans 596613 steps (20 nominal epochs), including the initial
stages; no optimizer training had started when this was changed. The original
training_plan.json describes only the initial five-epoch stage; the extended
plan describes the full requested schedule.

## Recovery on 2026-11-27

All 240 episodes finished preprocessing. The first four-GPU run started at
02:56:43 but stopped at 03:01:16 after approximately 34 optimizer steps because
GPU 0/4 free memory fell below the 1536 MiB safety floor. Available host memory
was 191.9 GiB and swap I/O was zero. No completed checkpoint or policy evaluation
was produced. The original continuation correctly refused to proceed after failure,
but the initial supervisor lacked automatic resource-recovery handling.

recover_pi05_regular240.py replaces both supervisors, reuses the completed data
and covers stages 1/3/5/10/15/20 in one process. It selects four GPUs with at
least 22000 MiB free before each training attempt. After GPU-memory/resource
pressure it stops only its own child, cools down currently busy cards, selects
available cards again and resumes a completed checkpoint when one exists.
Resource retries are bounded to eight per training stage; unrelated jobs are
never terminated. Unexpected non-resource training errors still stop for diagnosis.
The initial recovery selection was physical GPUs 6/7/2/5, batch 8.
