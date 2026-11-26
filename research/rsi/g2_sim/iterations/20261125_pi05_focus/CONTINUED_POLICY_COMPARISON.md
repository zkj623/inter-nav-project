# Continue policy after reaching 10 cm

Compare the existing paired 129000-step lock10 trials with `continue10_lift_v1`.
Both conditions use the same checkpoint, per-object seed 20261125, pilot4
placements, initialization, prompts and three-camera/state inputs.

At the 10 cm trigger, the new condition only marks the beginning of a 5.1-second
observation window. It does not clear the action chunk, freeze targets or stop
inference. The policy continues to control all 14 arm joints and two grippers.
The phase and object pose remain unavailable to the policy. No new hold prompt,
force input, feedback controller, friction or actuator-limit change is added.

The original physical checks remain, including permitted base contact. A
continued lift can fail the height-variation threshold without actually
slipping, so report actual height, floor clearance, relative drift and rotation
separately. The auditor verifies continued inference during the window rather
than constant actions. Original lock10 results remain untouched.

`report_continue10_comparison.py` also compares pre-trigger joint/object
trajectories. Identical inputs and seeds do not alone guarantee bit-identical
simulator runs; assess any divergence before attributing outcomes to stopping.

Remote runner: `evaluate_continue10_129000.py`. It owns the shared supervisor
lock, runs simulations serially, monitors system capacity every 10 seconds,
and cleans up its own inference and simulation processes. Training had already
completed its 172000-step stage; this experiment does not restart training.

At creation, syntax and adapter tests passed and physical tests were running.
Final output: `pi05_focus_20261125/continue10_vs_fixed129000.json`, with videos
under `pi05_focus_20261125/reviews/pi05_focus_continued_relative_129000_continue10/`.

## Completed observations

All four trials finished on 2026-11-26. The lidded-bin simulation had one
initialization timeout before any policy action and succeeded in starting on its
automatic retry. No training was started by the experiment.

| Object | Continued-policy observation |
| --- | --- |
| box_white | Reference lift reached 13.06 cm; minimum visual floor clearance was -0.34 mm, so this was not a full airborne hold |
| bin_lidded | Continued ascending to 22.32 cm; left-wrist contact stopped the window at 4.7 s |
| bin_pedal | Completed 5 s fully airborne, final lift 16.59 cm; minimum visual floor clearance 6.87 cm |
| basket_rectangular | Peaked at 3.25 cm then lost the grasp before the 10 cm trigger |

None passed the original static-hold audit. The pedal bin exceeded the height
variation and relative-motion limits while continuing to ascend; do not
describe this as a drop. The wrist-contact trial did not complete its full
window. Do not conflate reference-point height with clearance of the whole mesh.

Pre-trigger trajectories diverged despite the same checkpoint, initial setup
and per-object seed. For the white box, initial qpos was identical but the first
action differed by 0.000954 rad; first-frame camera mean absolute pixel errors
were approximately 0.29-0.36 on an 8-bit scale. Later pre-trigger differences
reached 7.0 cm in object position and 0.47 rad in a joint. These measurements
do not isolate rendering versus inference numeric variation as the cause.
This is a confounded rerun comparison, not proof that fixed targets are better.
The basket failed before the control modes diverged and supplies no evidence
about the post-trigger choice.

The current policy has no additional instruction to stop ascending at 10 cm.
Continuing image/state feedback is distinct from a dedicated object-stabilizing
hold controller. Fixed targets still use low-level joint feedback; they are
not a shutdown of all feedback. Relative object motion is measured against the
left TCP and can include intentional arm adjustment, not only contact slip.

Local annotated video review:
`E:/G2_interaction/collection_20261122/pi05_continue10_review/index.html`.
The original automated review is preserved beside it as `audit_review.html`.
