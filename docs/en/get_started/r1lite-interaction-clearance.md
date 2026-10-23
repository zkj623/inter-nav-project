# R1 Lite mandatory-interaction clearance

This experimental task is separate from the default navigation demos. Upstream
has already designated a mandatory interaction; the task clears the passage
through real-contact bimanual box transport and release, without deciding
between bypassing and interacting.

The execution modules remain in `grutopia_extension/interaction_clearance/`
and `grutopia_extension/manipulation/`. Research orchestration, benchmark
configurations, evaluation and video tools live under `research/rsi/`.
The old `grutopia/demo/evolve_r1lite_clearance_skill.py` remains a compatibility
entry point for the relocated skill generator.

See [the RSI research README](../../../research/rsi/README.md) for the current
skill sequence, environment/asset requirements, commands, dataset boundaries,
all-success/all-failure video presentation, and implementation limitations.

The frozen round-06 execution version passes 12/12 original configurations and
30/30 development states, but 56/60 validation states; restart evaluation never
started. The published validation configurations are reproducibility inputs,
not a new unseen holdout for later policy tuning. Automatic bimanual policy
learning and a complete two-way policy-skill learning loop remain unfinished.
