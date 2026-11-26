# Demonstration-height, ten-second hold evaluation

Run the existing 129000-step policy until it reaches the final-height band of
the actual training demonstrations, then observe another 10.1 seconds with
policy feedback continuously active. No expert actions, fixed targets, added
hold prompt, force feedback, friction changes or actuator changes are applied.
Object state only selects the evaluation window and is not a policy input.

The target is the median final hold height across each object's two pilot4
demonstrations: box_white 16.722 cm, bin_lidded 18.065 cm, bin_pedal 18.529 cm,
basket_rectangular 19.689 cm. A 5 mm trigger tolerance is used. Source paths and
unrounded values are in `sources/sim/terminal_lift_targets.json`.

Allow at most 80 seconds of learned control to reach the band. If reached, allow
the full additional window even if it extends beyond the 80-second deadline.
Reaching the height band alone does not prove that the learned motion phase has
finished or that the object is fully clear of the ground.

Report actual full-mesh floor clearance and ten-second airborne behavior
separately from the static-stability audit. The existing >=9 cm reference-height
floor and >5 mm visual ground-clearance rules remain. Stability is measured in
the final 5 seconds: height variation <=2 cm, object-to-left-TCP relative drift
<=2 cm, orientation change <=10 degrees. Base contact stays diagnostic only.
Whole-window motion is also recorded so early sliding is not hidden.

The terminal auditor verifies continued policy inference and at least 200
recorded hold actions, plus the existing physics/data checks. Unit checks cover
ten-second versus five-second duration, height loss, drift, and absence of a
policy lock. Source syntax was checked before launching.

Remote runner: `evaluate_terminal_129000.py`. It acquires the shared supervisor
lock, monitors capacity every 10 seconds and runs one simulation at a time.
Output run: `pi05_focus_terminal_relative_129000`.
Physical summary: `pi05_focus_20261125/terminal_129000_physical_outcomes.json`.
Training was already complete; this experiment does not restart it.

Completed on 2026-11-26 at 18:13:18 (server local time). One valid rollout per
object: 1/4 completed ten seconds fully airborne; 0/4 passed all static-stability
and contact checks. No training process was started. Prior lock10 and continue10
results remain unchanged and are matched-configuration reruns, not exact state
forks; rendering and early trajectories can differ even with the same seed.

| Object | Maximum reference lift | Observed outcome |
| --- | ---: | --- |
| box_white | 21.04 cm | Reached demonstration band, but full-mesh floor clearance fell to -0.15 mm; right wrist contact stopped the run after 4.1 s of the observation window. Final video tilt approximately 64 degrees. |
| bin_lidded | 22.96 cm | Fully airborne during the observed window (minimum mesh clearance 11.71 cm), but continued ascending; left wrist contact stopped the run after 1.7 s. No ten-second conclusion is available. |
| bin_pedal | 29.31 cm | Completed 10.0 s fully airborne, minimum mesh clearance 13.47 cm, no nonhand/base contact. Continued ascending; final-five-second height range 5.92 cm and orientation change 29.52 degrees. Final video tilt approximately 39 degrees. |
| basket_rectangular | 0.446 cm | Did not reach the band. Loaded visible-surface gap stopped the run at 22.6 s: right outer finger link4, 1.54065 mm gap versus 1.5 mm threshold, force 32.55 N. A marginal geometry-audit stop is not evidence that unrestricted continuation necessarily fails to lift. |

The white box and pedal bin each had one initialization timeout before their
valid rollout. The automatic retry happened before policy execution. These are
infrastructure events, not additional grasp failures. Trials and videos were
retained, including failures. Full summary and physical metrics are in
`records/terminal_129000_summary.json` and
`records/terminal_129000_physical_outcomes.json`.

Interpretation: matching the evaluation trigger to demonstration height did not
make this policy settle into a stable carry pose. The pedal bin proves sustained
airborne behavior in this rollout, but not stable orientation or chassis-transport
readiness. The lidded bin continued upward into wrist contact. Do not infer from
height alone that the learned lift phase has finished. Relative drift during
active motion is measured against the left TCP and is not a direct slip measure.

Next experiment should explicitly include longer stable holding demonstrations
and recovery from small grasp/pose deviations, with consistent target height.
An endpoint-hold controller or learned holding behavior should be evaluated
separately before adding gentle chassis acceleration. The current evidence does
not establish whether more epochs alone, a new controller, or more data would
solve all four cases. No such intervention was injected into this test.

Local video review: `E:/G2_interaction/collection_20261122/pi05_terminal129000_review/index.html`.
All four downloaded videos were checked against the export SHA-256 hashes.
