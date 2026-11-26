# Lift to 10 cm, then hold

Revision: `lock10_lift_v1`, 2026-11-26.

The user accepts object contact with the chassis. Such contact is logged but is
not a rejection, regardless of contact impulse or geometric overlap. Collision
physics remains enabled. Other non-gripper robot links retain their checks.

At the first sampled lift of at least 0.10 m, stop policy inference, discard the
remaining action chunk, and hold the last commanded arm and gripper targets.
Use the existing physical joint controllers and effort limits. Do not attach
the object or kinematically freeze the robot. This stop trigger uses simulator
object state; it is a supervisor, not an additional learned-policy input.

Hold for 5.1 seconds. Initial engineering tolerances for stable lift are:

- Lift remains at least 0.09 m throughout the hold.
- In the final 3 seconds: height variation and object-to-TCP drift are each at
  most 0.02 m; orientation change is at most 10 degrees.
- Visual geometry remains more than 5 mm above the floor.
- Existing joint, contact-surface, self-collision and data-integrity checks apply.
- Independent audit verifies constant held actions and zero inference calls
  after the stop trigger. Absolute tilt is diagnostic, not a rejection.

These tolerances are implementation choices, not additional user requirements.
This verifies stationary lift and hold only. Chassis acceleration, braking,
turning and transport stability still need a separate physical rollout.

`evaluate_pi05_stable.py` now defaults to `--protocol lock10`; `--protocol stable`
retains the old evaluation. Existing result folders reject a protocol mismatch.
Old results are not relabeled. Summaries and review pages identify the protocol.
The running supervisor invokes this evaluator afresh after the 172000-step
stage, so no training restart or overlapping GPU simulation is necessary.

Validation so far: syntax checks and CPU behavior checks (see
`records/lock10_unit_checks.json`). No new physical rollout has completed at
this revision. Earlier 0/4 results use the old continuous-policy, 15 cm protocol
and do not measure this new stopping behavior.

## First physical evaluation: checkpoint 140000

See `records/manual_lock10_140000_combined.json`. One valid rollout per training
object, 1/4 independently accepted. This is not a robustness estimate.

| Object | Maximum lift | Result |
| --- | --- | --- |
| bin_lidded | 10.49 cm | Passed physical and three-camera audits; held 5 s, minimum lift 9.67 cm |
| box_white | 5.57 cm | Contact point 6.40 mm from visual surface; stopped by surface check |
| bin_pedal | 4.75 cm | Dropped to floor, did not recover within the full timeout |
| basket_rectangular | approximately 0 cm | Did not lift within the full timeout |

The lidded bin's first attempt stalled during simulation initialization and was
excluded from policy outcomes. A serial retry on another available simulation
GPU used the same checkpoint and reset seed 20261125. The successful hold had
0.66 mm object-to-TCP drift and 0.11 degree orientation change in its final
3 seconds. The action audit confirmed constant targets and zero policy calls
after the stop trigger. No valid rollout recorded object-to-base contacts.

Training and its supervisor were suspended during evaluation and resumed with
their existing process/optimizer state afterward. Training was observed running
again at step 144390 on GPU 6. The evaluated checkpoint remains backed up at
`checkpoint_backups/pi05_focus/pilot8_relative_manual_140000` on the server.
