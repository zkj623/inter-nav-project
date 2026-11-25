# ACT progress-conditioning correction

This experiment addresses the closed-loop stall observed in the four-object ACT overfit test. At frame 800, the white-box policy predicted expert actions with about 0.00245 rad arm MAE on expert observations but about 0.249 rad on its own rollout observations. Recomputing the deployed policy reproduced its recorded commands. The lidded-bin rollout showed the same pattern: the robot remained near contact refinement instead of lifting. Pedal-bin and rectangular-basket rollouts instead tilted and contacted the base or terminal arm links.

## Changes

- Add a learned Fourier time embedding to the ACT state token so progress has a stronger representation than a single scalar input.
- During training only, mask normalized joint observations with probability 0.5 and normalized image observations with probability 0.3. This is an ablation intended to reduce dependence on the current pose as a shortcut for the next action. It is not corrective expert data or a claim of robustness.
- Sample 10% initial frames, 20% contact frames, 35% phase-transition windows, and 35% uniform frames.
- Train one shared model for four objects, using the four-demonstration `pilot4_one_each` set, with deterministic latent inference/training and 50-step action chunks. Resume the previous 15000-step ACT time model; train 8000 additional steps.
- Clamp inference time to a point in the trained final hold interval. The 80-second evaluation otherwise exceeds the approximately 46–57-second demonstrations. This uses only stored per-task timing limits, not expert actions or object feedback.

Inputs remain head/two-wrist RGB, joint state, task ID and elapsed time. This is an explicitly time-conditioned ACT variant and a training-position overfit experiment, not a claim of generalization. Sampling phase labels are training annotations only; no expert phase labels are supplied during inference.

## Validation and execution

`test_progress_act.py` checks checkpoint compatibility, output shape, finite outputs, deterministic evaluation, dependence on time and nonzero gradients in the progress encoder. It passed before training started. These checks are not evidence of manipulation success.

`run_progress_guarded.py` runs one training process on GPU 0, stops it before starting inference, then evaluates one simulator at a time on GPU 1. It shares the existing supervisor lock and retains the memory/swap/I/O/disk/GPU pressure limits. Only its own child process groups are terminated on a stop. Other users' jobs are untouched.

The physical acceptance checks are unchanged: 15 cm lift, stable hold, contact geometry, collision, tilt and drift checks. An offline loss improvement or a short lift does not count as task success. Results must be checked in `evaluations/act_progress_v1_8000/summary.json` after the run finishes.

Sources depend on the existing G2 workspace, the previously archived `g2_act_model.py`, patched official ACT implementation, and accepted demonstration caches. Setup order: `setup_progress_act.py`, `fix_progress_terminal.py`, `test_progress_act.py`, then the guarded runner. Do not rerun setup during an active run; setup regenerates source files.

## Conditional input ablation

The 2000/6000-step counterfactual checks still showed large errors when conditioning on old stalled rollout observations. Explicitly masking those observations improved the requested lift progression in sampled frames, although action accuracy was not sufficient evidence of physical success.

`queue_timeonly_ablation.py` waits until the full progress-policy evaluation finishes. If fewer than two objects pass, it runs a further 8000-step ACT ablation with both RGB and joint-state inputs masked during training and inference; only learned task/time conditioning remains. This is a learned fixed-start trajectory model, not visual closed-loop control, corrective demonstration collection, or expert replay. It must be reported separately even if it passes. If the progress policy already passes at least two objects, the ablation is skipped. Any resource stop aborts the queue.

`test_timeonly_mask.py` passed checks for masking observations, preserving elapsed time and its gradients, and avoiding input mutation. The queue runs at lower CPU priority and uses the same exclusive supervisor lock and resource guard as the first experiment.

## CPU scheduling limit

During the run, shared CPU load rose substantially despite abundant RAM and negligible swap/disk pressure. The training tree was first lowered to nice 10. The active supervisor, inference/evaluation descendants and conditional queue were subsequently restricted to eight logical CPUs; future children inherit the restriction. The guarded launchers now apply this limit and nice 10 themselves. `limit_own_experiment_cpu.py` checks ownership and the expected supervisor command before changing the affinity of its descendants. It does not modify other users' processes. Run `persist_cpu_limits.py` after regenerating setup files when using an older baseline guard without this limit.
