# π0.5 closed-loop fine-tuning — 2026-11-25

The user requested prioritizing π0.5, first validating multi-object lift behavior and then training on a broader distribution. This iteration uses language, head/left-wrist/right-wrist RGB, and 16 joint coordinates. It does not use task IDs, elapsed-time conditioning, simulated forces, expert takeover, object attachments, or object pose feedback for policy control.

## Running experiment

- Remote workspace: `/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124`
- Supervisor: `pi05_focus_supervisor.py`; state: `pi05_focus_20261125/status.json`.
- Run: `pilot8_focus_v1`, initialized from the completed π0.5 `pilot4/3000` parameter checkpoint; optimizer starts fresh.
- Original 8 demonstrations, 4 objects, 8,595 synchronized frames; batch 2; absolute 16-dimensional joint targets at 20 Hz; 50-step chunks.
- Model-internal state/action dimension is padded to 32. Inputs and labels use the same training-set quantile normalization as the inference server.
- LoRA on the language model/action expert; frozen visual backbone; original OpenPI image augmentation.
- Stages: 6,000, 18,000, then 36,000 cumulative new updates, with checkpoint resume and closed-loop evaluation between stages. Inference executes 5 actions before replanning.
- Each evaluation first measures offline action error, then evaluates all four objects in simulation using one checkpoint.
- Expansion gate: at least three of four objects pass twice with the same checkpoint. If the gate is not met after 36,000 updates, stop for failure analysis rather than claiming success or automatically scaling data.
- A stage checkpoint is synchronously saved on local SSD, then copied and SHA256-verified to persistent storage after evaluation. Previous experiment files remain unchanged.

The supervisor status is authoritative. A launched process or low offline loss is not evidence of a successful policy. At the time of this initial note, training was starting; no new closed-loop results existed.

## User-requested acceptance change

Separate `stable_policy_*_trial.py` harnesses and `verify_stable_lift.py` preserve the old experiments. Lift height >=15 cm, final stable hold >=3 seconds, tilt <=20 degrees, relative drift <=2 cm, visual-surface checks, physical geometry, joint limits, and self-collision checks remain.

Four-finger and bilateral contact fractions are diagnostics, not pass/fail gates. Minor base contact is explicitly limited to `base_link`, <=0.02 N·s impulse per event, and <=3 mm penetration. These numeric tolerances are an implementation choice interpreting the user's allowance for slight base contact; wrist/arm collisions and deeper base penetration are not exempt. Contact logs and geometry hits are retained. Boundary tests cover allowed base contact versus wrist contact, excessive impulse, and excessive penetration.

## Broader-distribution stage

`prepare_pi05_focus_expanded.py` checks the existing accepted raw episodes, excludes unreadable files, validates three-camera/state/action alignment, deduplicates sources, and converts one episode at a time. Whole object-placement groups are held out; positions used by the initial eight demonstrations cannot become held-out positions. Normalization uses training episodes only.

Initial metadata scan: 63 registered episodes, 57 readable three-camera candidates, 6 unreadable HDF5 files. Do not call all 63 usable data. Conversion performs further checks. The reinforced box currently has only one candidate, so it cannot support a position holdout. The generated `scope.json` explicitly records objects without holdouts, actual episode counts, and position/yaw ranges.

After the repeated pilot gate, the pipeline trains the broader dataset for five equivalent sampling passes with batch 2, then evaluates training and held-out placements separately. This is existing position/orientation variation plus image augmentation; it does not establish unseen-object, unseen-background, or real-hardware generalization. Future new-scene claims require additional data and held-out scenes.

## Resource management

One training process, followed by one policy server and one serial simulation evaluator. Training and simulation never overlap. A shared supervisor file lock prevents concurrent copies of this pipeline. Monitor RAM, swap I/O, disk, GPU free memory/temperature, and CPU I/O wait every 10 seconds. Persistent pressure stops only this supervisor's child process groups. Sparse synchronous checkpoint writes avoid accumulating asynchronous save jobs. No other user's processes are modified.

Source files in `sources/` are snapshots intended to be installed into the corresponding remote workspace; `pi05_focus_setup.py` generates the separate harnesses and auditor from the preserved existing implementation. The archived source manifest and `setup.json` record hashes. Do not relaunch an active supervisor; read `status.json` and inspect its PID first.
