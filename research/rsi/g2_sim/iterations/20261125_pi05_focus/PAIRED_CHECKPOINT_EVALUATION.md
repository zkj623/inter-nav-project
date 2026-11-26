# Paired checkpoint comparison

Started 2026-11-26. Compare the 86000, 129000 and 140000 training-step checkpoints
on box_white, bin_lidded, bin_pedal and basket_rectangular under lock10_lift_v1.
The first two checkpoint directory indices are 85999 and 128999.

All 12 trials use the same pilot4 training placements, robot initialization,
prompt, cameras, execution length of 5 actions, and maximum 80 seconds of policy
control. Reset sampling seed 20261125 immediately before **each** object's
rollout, after the offline probe. This prevents previous object rollout lengths
and probe sampling from changing another object's initial random generator
state. The earlier 140000 results are retained but are not substituted into this
comparison. One rollout per object is a screening comparison, not a robustness
or generalization measurement.

Numeric checkpoint directories under
`checkpoint_backups/pi05_focus/paired_lock10_20261126/` contain read-only-in-use
hardlinks to the verified backups' params, assets and metadata. Inference does
not modify these files. Checkpoint loading computes a hash over all params files.

The comparison suspends the existing trainer and supervisor, preserving their
process and optimizer state. It runs a single simulation at a time with a
separate inference GPU, monitors capacity every 10 seconds, and resumes training
in its final cleanup. Do not start another comparison while this one holds
`pi05_focus_20261125/paired_lock10.lock`.

Initialization without a live frame is limited to 240 seconds; stalled live
progress to 180 seconds. Infrastructure failures get one serial retry, with the
incomplete trial directory preserved. Ordinary grasp failures are not retried.
The report does not select a best checkpoint if unresolved infrastructure
failures remain. Ties are reported rather than silently choosing one model.

Remote runner: `compare_pi05_lock10_checkpoints.py` in the overnight workspace.
It writes:

- `pi05_focus_20261125/paired_lock10_runner.log`
- `pi05_focus_20261125/paired_lock10_comparison.json` and `.html`
- `evaluations/pi05_focus_paired_relative_<step>_lock10/summary.json`
- `pi05_focus_20261125/reviews/pi05_focus_paired_relative_<step>_lock10/`

The aggregate updates after each checkpoint, with `complete: true` only after
all three. Per-object summaries update during each checkpoint. Videos are
exported automatically. At the time of this commit, execution has started but
results are pending; do not infer a winner from the experiment setup.
