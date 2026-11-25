# π0.5 closed-loop fine-tuning — 2026-11-25

The user requested prioritizing π0.5, first validating multi-object lift behavior and then training on a broader distribution. This iteration uses language, head/left-wrist/right-wrist RGB, and 16 joint coordinates. It does not use task IDs, elapsed-time conditioning, simulated forces, expert takeover, object attachments, or object pose feedback for policy control.

## Running experiment

- Remote workspace: `/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124`
- Supervisor: `pi05_focus_relative_supervisor.py`; state: `pi05_focus_20261125/status.json`.
- Run: `pilot8_focus_v1`, initialized from the completed π0.5 `pilot4/3000` parameter checkpoint; optimizer starts fresh.
- Original 8 demonstrations, 4 objects, 8,595 synchronized frames; batch 2; 14 relative arm targets plus 2 absolute gripper targets at 20 Hz; 50-step chunks.
- Model-internal state/action dimension is padded to 32. Inputs and labels use the same training-set quantile normalization as the inference server.
- LoRA on the language model/action expert; frozen visual backbone; original OpenPI image augmentation.
- Stages: 6,000, 18,000, then 36,000 cumulative new updates, with checkpoint resume and closed-loop evaluation between stages. Inference executes 5 actions before replanning.
- Each evaluation first measures offline action error, then evaluates all four objects in simulation using one checkpoint. A deterministic seed derived from the run name is recorded; the repeated evaluation uses a different seed. The server's source hash is recorded alongside the checkpoint hash.
- Expansion gate: at least three of four objects pass twice with the same checkpoint. If the gate is not met after 36,000 updates, stop for failure analysis rather than claiming success or automatically scaling data.
- A stage checkpoint is synchronously saved on local SSD, then copied and SHA256-verified to persistent storage after evaluation. Previous experiment files remain unchanged.

The supervisor status is authoritative. A launched process or low offline loss is not evidence of a successful policy. At the time of this initial note, training was starting; no new closed-loop results existed.

## Official action representation correction

Checked official OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`: [`LeRobotAlohaDataConfig`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L229) defaults to `use_delta_joint_actions=True`; `pi05_aloha_pen_uncap` uses this configuration. [`DeltaActions` and `AbsoluteActions`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L204) subtract/add the same current observed joint state across the whole action chunk. This is not an increment between adjacent future timesteps. Absolute action prediction is supported by other configurations; it is not inherently invalid, but our initial G2 adapter omitted the default ALOHA conversion.

For G2, columns 0?13 are arm joints and columns 14?15 are grippers. Training labels are `future_absolute_arm_target - current_measured_arm_qpos`; grippers remain absolute. Action quantiles are recalculated in this relative representation from training episodes and all chunk offsets. Inference first denormalizes and then adds the observation's arm qpos once to every chunk element; the simulator receives absolute joint targets as before. Original demonstrations need no recollection for this correction.

`verify_pi05_relative_official.py` compares the adapted labels against the actual official transform and reconstructs the original targets at 24 start/middle/end samples across all 8 episodes, including terminal padding. Exact transform equality passed; maximum reconstruction error was 0 radians; grippers were unchanged. This validates representation consistency, not policy success. `records/relative_launch.json` records the new launch and source hashes. No closed-loop result exists yet for the corrected run.

Start only after verifying no supervisor holds the shared lock: `python pi05_focus_relative_supervisor.py /data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124/weights/pi05_base`. Training begins with the eight-demo pilot; expansion remains conditional on repeated closed-loop success.

## User-requested acceptance change

Separate `stable_policy_*_trial.py` harnesses and `verify_stable_lift.py` preserve the old experiments. Lift height >=15 cm, stable hold >=3 seconds, tilt <=20 degrees, relative drift <=2 cm, visual-surface checks, physical geometry, joint limits, and self-collision checks remain. The first measured lift >=15 cm with tilt <=20 degrees starts a 3.1-second acceptance window, during which the learned policy continues unchanged. The 80-second duration is a maximum timeout, not a required fixed endpoint. Object measurements select the evaluation window only and are not policy inputs.

Four-finger and bilateral contact fractions are diagnostics, not pass/fail gates. Minor base contact is explicitly limited to `base_link`, <=0.02 N·s impulse per event, and <=3 mm penetration. These numeric tolerances are an implementation choice interpreting the user's allowance for slight base contact; wrist/arm collisions and deeper base penetration are not exempt. Contact logs and geometry hits are retained. Boundary tests cover allowed base contact versus wrist contact, excessive impulse, and excessive penetration. Successful trials additionally pass `verify_pi05_threecam.py`, checking all-frame camera validity, synchronization, recorder/calibration provenance, and action alignment. Arm mirror asymmetry is reported for visual review, not used as an extra lift-success gate.

## Broader-distribution stage

`prepare_pi05_focus_expanded.py` checks the existing accepted raw episodes against their saved SHA256, excludes unreadable files or corrupt image chunks, validates three-camera/state/action alignment, deduplicates sources, and converts one episode at a time. Conversion can resume after interruption. Whole object-placement groups are held out; positions used by the initial eight demonstrations cannot become held-out positions. Normalization uses training episodes only.

Initial metadata scan: 63 registered episodes, 57 readable three-camera candidates, 6 unreadable HDF5 files. Do not call all 63 usable data. Conversion performs further checks. The reinforced box currently has only one candidate, so it cannot support a position holdout. The generated `scope.json` explicitly records objects without holdouts, actual episode counts, and position/yaw ranges.

After the repeated pilot gate, the pipeline trains the broader dataset for five equivalent sampling passes with batch 2, then evaluates training and held-out placements separately. Expanded training samples objects uniformly before choosing an episode, so objects with more recordings do not dominate. This is existing position/orientation variation plus image augmentation; it does not establish unseen-object, unseen-background, or real-hardware generalization. Future new-scene claims require additional data and held-out scenes.

## Resource management

One training process, followed by one policy server and one serial simulation evaluator. Training and simulation never overlap. A shared supervisor file lock prevents concurrent copies of this pipeline. Monitor RAM, swap I/O, disk, GPU free memory/temperature, and CPU I/O wait every 10 seconds. Persistent pressure stops only this supervisor's child process groups. Sparse synchronous checkpoint writes avoid accumulating asynchronous save jobs. No other user's processes are modified.

`export_pi05_focus_review.py` produces a compact result JSON, an HTML review, and H.264 videos after each evaluation. It verifies that the transcoded videos decode and records their hashes/frame counts.

Run `inspect_pi05_focus.py` in the remote workspace for a bounded, read-only progress/resource/result summary. It includes the latest loss/gradient values, a rolling 300-step mean, and finite-value checks. Offline evaluation separates arm-angle error from gripper error, reports each action phase, and compares against holding the current arm pose. Closed-loop lift success remains the outcome criterion. It does not start jobs. The approximate step/ETA comes from OpenPI's progress log; the checkpoint and physical verification files remain authoritative for completed work.

Source files in `sources/` are current snapshots intended to be installed into the corresponding remote workspace. `pi05_focus_setup.py` records the initial bootstrap recipe; subsequent live-window, camera-audit, and sampling-seed changes are represented by the final snapshots. The running supervisor invokes `pi05_policy_server.py`; the current server snapshot is `pi05_focus_policy_server.py` (identical code), and the prior server is preserved as `pi05_policy_server_before_seed.py`. The archived source manifest and initial `setup.json` record hashes. Per-run trainer snapshots preserve the source loaded for that stage, including before later logging improvements. Do not relaunch an active supervisor; read `status.json` and inspect its PID first.
