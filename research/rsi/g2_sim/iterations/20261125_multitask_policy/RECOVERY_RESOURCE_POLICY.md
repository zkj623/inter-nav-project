# 2026-11-25 recovery and resource policy

The host rebooted at approximately 10:36 CST. Kernel/system logs were inaccessible to the training account; the reboot cause is unconfirmed. No previous training processes survived. The recorded accepted dataset contains 63 episodes (52 original-placement and 11 wider-placement episodes).

Pi05 step 3000 loaded successfully after reboot. Step 4000 is an unfinished Orbax temporary directory and must not be treated as a completed checkpoint. ACT checkpoints remain on disk. A completed save or low offline loss is not evidence of closed-loop success.

## Sequential execution

`resume_guarded_eval.py` first evaluates Pi05 step 3000 on four objects, then ACT robust step 12000 on the same four objects. Only one inference server and one simulator run concurrently, on GPU 0 and GPU 1 respectively. It takes an exclusive supervisor lock. Existing other-user jobs are never terminated.

`queue_guarded_time.py` waits for successful completion of the evaluation stage and aborts if the supervisor records a stop. It then runs `resume_guarded_eval.py --time-baseline`: one ACT training job, followed by evaluation, with no training/simulation overlap. This is a four-demonstration diagnostic, one shared checkpoint with task conditioning and an additional elapsed-time input. It does not establish position generalization. The learned model does not retrieve or replay expert actions at inference.

## Resource checks

The supervisor samples system memory, swap usage and actual swap I/O rate, CPU I/O wait, disk free space, GPU free memory and temperature every 10 seconds during evaluation/training (5 seconds during server startup). Logs: `recovery_20261125/resources.jsonl`.

- Startup: at least 64 GiB available RAM and 20,000 MiB free on each selected GPU.
- Immediate stop: available RAM below 16 GiB.
- Stop after three consecutive excessive samples: available RAM below 32 GiB, swap I/O above 20 MiB/s, CPU I/O wait above 25%, root disk free below 30 GiB, data disk free below 100 GiB, selected GPU free below 2,048 MiB, or a GPU temperature above 85 C.
- Monitor failures and task deadlines also abort the supervisor. Only its own child process groups are terminated; files/checkpoints are preserved.
- Swap occupancy alone is not a stop criterion: occupied swap does not prove active memory thrashing.

These conservative operational limits are not a guarantee against hardware or network failures. The monitor does not currently measure per-device request latency. Inspect disk latency/pressure separately when startup or checkpoint saves slow down; do not respond by increasing concurrency.

`test_resource_guard.py` tests healthy operation, immediate low-memory stop, transient tolerance, and sustained memory/swap/I/O/disk/GPU pressure. All four test methods passed locally.

## Acceptance

Keep original physical checks for lift height, stable hold, contact, visible-surface geometry, collisions, tilt and drift. Report learned-policy results separately from expert replay. Multiple objects must pass using the same trained checkpoint SHA; a partially completed evaluation is not an all-object result.
