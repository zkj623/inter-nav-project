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

## Prepared input ablation (inactive; superseded below)

The 2000/6000-step counterfactual checks still showed large errors when conditioning on old stalled rollout observations. Explicitly masking those observations improved the requested lift progression in sampled frames, although action accuracy was not sufficient evidence of physical success.

`queue_timeonly_ablation.py` waits until the full progress-policy evaluation finishes. If fewer than two objects pass, it runs a further 8000-step ACT ablation with both RGB and joint-state inputs masked during training and inference; only learned task/time conditioning remains. This is a learned fixed-start trajectory model, not visual closed-loop control, corrective demonstration collection, or expert replay. It must be reported separately even if it passes. If the progress policy already passes at least two objects, the ablation is skipped. Any resource stop aborts the queue.

`test_timeonly_mask.py` passed checks for masking observations, preserving elapsed time and its gradients, and avoiding input mutation. The queue runs at lower CPU priority and uses the same exclusive supervisor lock and resource guard as the first experiment.

## CPU scheduling limit

During the run, shared CPU load rose substantially despite abundant RAM and negligible swap/disk pressure. The training tree was first lowered to nice 10. The active supervisor, inference/evaluation descendants and conditional queue were subsequently restricted to eight logical CPUs; future children inherit the restriction. The guarded launchers now apply this limit and nice 10 themselves. `limit_own_experiment_cpu.py` checks ownership and the expected supervisor command before changing the affinity of its descendants. It does not modify other users' processes. Run `persist_cpu_limits.py` after regenerating setup files when using an older baseline guard without this limit.

## Feedback-cadence experiment (historical queue; active evaluation)

The white-box rollout passed independent verification: minimum hold lift 0.16577 m, maximum hold tilt 2.04 degrees, four-finger contact fraction 1.0, relative drift 0.194 mm, and no non-hand or self contact impulses. The lidded bin lifted 0.15667 m but tilted and contacted the base; the pedal bin also contacted the base. These failures remain failures.

The waiting `queue_timeonly_ablation.py` was stopped before it launched any training. The active replacement is `queue_receding_eval.py`: after the main four-object evaluation, if any object failed, evaluate the **same checkpoint** with `execute_steps=1` (20 Hz observation/replanning) instead of 10 (2 Hz replanning). This changes feedback frequency only, without changing model weights, physics gates or the 50-step output horizon. Test all four objects, beginning with the failed bins. It is a hypothesis test, not an assumption that faster feedback fixes the errors.

Only if both execution settings fail to pass at least two objects does the queue launch the explicit task/time-only ablation. Do not run the old queue alongside the replacement. Reports must group successes by checkpoint **and execution setting**, rather than combining successes from different feedback frequencies to claim a unified policy configuration.


## Main result and contact-input correction

The completed progress-model evaluation passed **1/4** objects: white box only. The lidded bin reached 15.67 cm before a base collision, the pedal bin reached 3.55 cm before a base collision, and the rectangular basket reached 14.30 cm before `LOADED_VISIBLE_SURFACE_GAP`. The archived result summary omits per-call inference timings but preserves checkpoint, execution setting and physical verdicts. No multi-object success has been established.

Failure traces show inner-finger contacts going to zero while the outer fingers carry roughly 25–36 N. This reduces the intended four-point support and precedes tipping/base contact. The expert reads contact forces and corrects wrist orientation and squeeze; the original ACT did not receive those forces. This is a hypothesis about missing feedback, not proof of the sole cause. The 20 Hz replanning lidded-bin test still failed from base contact after reaching 16.19 cm.

`setup_force_act.py` prepares `pilot4_force_causal` using external links to the immutable four demonstration caches. The only new arrays are four distal finger-object contact-force norms, averaged over the latest three **past** 10 Hz samples and aligned to each pre-action image/joint observation. Observed ages are 16.7–66.7 ms. No demonstrations were added. Contact values are clipped at 100 N for input encoding and transformed with `log1p(force)/log(36)`; this does not change simulated forces or acceptance thresholds.

The 21-value state is 16 normalized joint positions, four encoded forces, and elapsed time. Three RGB cameras and task ID remain inputs. The added force encoder and expanded state projection preserve the previous model's predictions at initialization. `test_force_act.py` passed causality, stale-input rejection, checkpoint equivalence, gradient, shape and input-dependence checks. These are software checks, not physical success.

The new live adapter supplies exactly the same causal force features and records values and timestamps in each HDF5. `audit_force_episode.py` recomputes them from the physical sample log. An audit failure rejects the rollout even if the ordinary physical verifier passes. The trained model produces actions; there is no scripted expert force-balancing or object pose feedback. Forces are object-filtered **simulator observations**, not a claim that G2 hardware provides these sensors.

The next serial scheduler was prepared as `queue_force_correction.py` (subsequently stopped as described below). Only the old waiting `queue_receding_eval.py` process was stopped; its already-running `run_receding_guarded.py` evaluation continues with its resource guard and lock. After all four feedback-cadence evaluations, if any still fail, the new scheduler trains `act_force_v1` for 8000 steps, then evaluates the same checkpoint on all four objects with `execute_steps=2` (10 Hz replanning, 20 Hz target application). Both old queue scripts and the prepared task/time-only ablation are **inactive**. Do not start them alongside this scheduler. A resource stop prevents automatic continuation.

Setup order on the existing workspace: `setup_force_act.py`, `finalize_force_setup.py`, `test_force_act.py`, then `run_force_guarded.py` once the previous run has fully stopped. Setup intentionally refuses to overwrite an existing force cache. Do not regenerate sources during active work. `switch_to_force_queue.py` is a historical, PID-checked one-time migration script, not a general launcher. Results belong in `evaluations/act_force_v1_8000/summary.json`; until present, the force correction is unvalidated.


## Active run after two negative cadence tests

The `execute_steps=1` evaluation was stopped early after both targeted bins failed: lidded 16.19 cm and pedal 3.84 cm, both `NONHAND_OBJECT_CONTACT`. This is **0/2 completed**, not 0/4. The rectangular-basket attempt was interrupted and the white-box case was not run in this setting. Partial files must not be counted as outcomes. `feedback_execute1_partial_summary.json` includes the incomplete-run marker and planned total.

To avoid spending another full evaluation cycle on a cadence change that had failed both targeted cases, the waiting force scheduler was stopped too. The old background guard inherited SIGINT-ignore behavior, so its own inference-server process was terminated; the guard then detected that exit and executed its cleanup of the remaining evaluation process group. Only after checking that all old model/evaluation processes had exited was `run_force_guarded.py` started directly at 13:54 CST. Its active PID at launch was 173823; this number is a historical observation, not a reusable process selector. All queue scripts are now inactive. The new guard owns the exclusive lock and will train, then evaluate sequentially.

`test_force_adapter.py` also passed: both force and elapsed-time inputs reach the request together, recorded values match the transmitted ones, the independent observation audit passes valid data and rejects a future timestamp. `results/progress_contact_failures.png` compares expert and failed model lift/tilt and shows the model's four distal contact forces. Neither those diagnostics nor training metrics count as successful manipulation; the force model's physical results remain pending.

## Collector concurrency correction

Both inactive collection entry points previously used one worker per asset, even when their GPU queue allowed only one simulator. That still permitted overlapping planners and independent audits. `configure_serial_collectors.py` verified that no collector was running, backed up both exact sources, and changed their thread pools to one worker each. Planning, simulation and verification are now serial across assets within each collector. The source hashes are in `collector_serial_revision.json`. No collection was launched and this finding does not establish the cause of the earlier server outage. A separate run must still be scheduled exclusively with resource monitoring; these collector entry points do not themselves replace the training/evaluation supervisor.


## Contact-input model: partial physical results

`act_force_v1/step_008000.pt` completed 8000 steps (offline train MAE 0.002522 rad). Checkpoint SHA-256: `94b9b6bd88bc8fac3685d073f7faffcd0ddf1fdfdafe8ae65853e914047b9f69`. With execute_steps=2, the lidded bin passed independent verification: minimum hold lift 17.669 cm, maximum hold tilt 2.393 degrees, all four finger contacts throughout hold. The white box held at 16.27 cm but its left inner finger had no contact; four-contact fraction was zero, so it failed. The pedal bin again contacted the base after 3.64 cm. The rectangular basket is still pending in this snapshot. This is 1/3 completed, not a multi-object success claim. The previous white-box success belongs to a different checkpoint and cannot be combined in evaluation totals.

The first force-model audit exposed a missing immutable source-version copy. `register_force_sources.py` confirmed the current exact script bytes against existing result hashes and saved them under `source_versions/<sha>.py`; the physical auditor itself was unchanged. The white-box result remains failed. Future setup now performs this registration before evaluation. The initial helper hashes are retained as `force_source_hashes_at_initial_setup.json` (matching the earlier Git revision); `archived_source_hashes.json` describes the current archived sources.

## Conditional expert-recovery data and retraining

`queue_dagger_correction.py` is the only waiting scheduler. It waits for all four force-model evaluations and launches `run_dagger_guarded.py` only if fewer than two objects pass. No second simulation/training job runs alongside the current experiment.

The prepared collection runs the learned policy for a bounded prefix, then switches to the existing geometry/contact-based expert controller. The expert first restores four-finger contact near the closest nominal lift waypoint, then completes the lift and stable hold. There are no object pose writes, attachments, external lifting forces, or weakened acceptance checks. Videos explicitly say **ACT PREFIX** or **EXPERT RECOVERY**. These mixed-controller rollouts are collection episodes, never pure-policy successes (`expert_feedback=true`).

The first accepted recovery pilot is sufficient to start this bounded correction experiment; failed cases are tried serially until one passes or the list is exhausted. Each attempt must pass the unchanged physical verifier, all three camera/command checks, causal force-input audit, and behavior-source boundary check. The policy prefix is excluded from the corrective training labels. HDF5 `behavior_source=0` identifies policy frames and `=1` identifies expert frames. The model only trains on the expert suffix, with its original elapsed time retained and clamped to the same terminal-progress limit used at inference.

The retraining dataset also includes the original four expert demonstrations and independently verified successful policy rollouts from the earlier white-box model and the new contact-input model. Those are explicitly labelled `verified_policy_distillation`, not expert demonstrations. Their force features are reconstructed only from past physical samples; this does not imply the earlier visual policy used force inputs. Each retained rollout must pass the camera/data audit. Merging validated behavior as **training data** does not merge evaluation success counts: the resulting one new checkpoint must be tested anew.

`act_dagger_v1` is prepared for 8000 steps at learning rate 1e-5, initialized from the contact-input model. The previous observation-masking ablation is removed so training uses the actual corrective observations. The architecture and evaluation behavior are otherwise checkpoint-compatible. `test_dagger_adapter.py` and `test_dagger_model.py` passed switching/label, checkpoint-output, and deterministic unmasked-input checks. The collector has not yet been physically validated. Setup order is `setup_dagger_recovery.py`, `setup_dagger_training.py`, tests, then the guarded runner after the active evaluation exits. All heavy stages remain serial, under the shared lock and the same resource guard. The new guard explicitly handles SIGINT/SIGTERM for cleanup.

This remains a four-object training-position experiment. It does not complete 12 objects x 20 varied demonstrations or establish generalization. Force observations are simulator inputs; hardware availability remains unverified.


## Completed result: one shared checkpoint passes two objects

The complete contact-input-model evaluation is **2/4**, with the same checkpoint SHA above and execute_steps=2 for every object. No expert targets or expert corrective controller were used in these four evaluations.

| Object | Verdict | Hold lift | Hold tilt | Four-contact fraction |
| --- | --- | --- | --- | --- |
| bin_lidded | Passed | 17.669 cm | 2.393 degrees | 1.0 |
| basket_rectangular | Passed | 19.637 cm | 1.873 degrees | 1.0 |
| box_white | Failed: left inner contact absent | 16.272 cm | 4.280 degrees | 0.0 |
| bin_pedal | Failed: base contact | No hold; maximum lift 3.638 cm | Not a hold result | Not a hold result |

Both passing cases also passed the independent three-camera/posture/action-alignment audit. Their body/arm constraint and physical geometry checks remain unchanged. These are one rollout per object at training positions, not a success-rate estimate or a generalization result. The earlier visual/time model's white-box success is not combined with this model's results. Pi0.5 was not retrained during this correction; its earlier four-case configuration remained unsuccessful, and these different input/training configurations do not establish an algorithm ranking.

The conditional recovery scheduler exited without launching collection or retraining because the minimum two-object milestone was reached. DAgger-style recovery sources and tests are prepared but remain physically unvalidated. No corrective demonstrations were collected in this experiment. White-box and pedal-bin recovery, the remaining eight objects, and 12 x 20 varied demonstrations are still outstanding.

`evaluate_checkpoint_guarded.py` is an inference-only entry point for reviewing the existing contact-input checkpoint without accidentally retraining. It defaults to the verified model and current four training scenes, requires a fresh run name, and retains the same exclusive lock, resource limits and serial execution. For example, in the existing remote training workspace:

```bash
/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python evaluate_checkpoint_guarded.py --run review_force_policy --assets bin_lidded basket_rectangular
```

The new entry point passed compilation and its help/argument check; the physical results above were obtained with the original guarded runner. Browser-compatible H.264 copies of all four review videos preserve original frame counts and frame rates; source and output hashes are in `force_video_audit.json`. Original evidence videos are retained. The local review page is `E:/G2_interaction/collection_20261122/policy_reviews/act_force_v1_review.html`.
