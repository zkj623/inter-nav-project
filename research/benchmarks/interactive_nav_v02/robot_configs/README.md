# Genie2 default posture

The current benchmark navigation code is in `../g2_navigation/`. Its default pose is `standby_joint_positions_rad` from the bundled `standby_preparation_config.json`, copied from the `genie2-rsi` branch with pinned commit/SHA-256 provenance.

Use `../g2_navigation/prepare_g2_runtime.py` to initialize the link transforms, joint state, drives and gripper mimics consistently. It also regenerates the enabled native collision projection for the selected pose. The current runner loads this run-specific projection and holds the configured arm/body targets while steering the wheels.

`genie2_navigation_tucked.json` and `run_g2_multiview_review.py` in this directory are historical experiment snapshots, not the current default posture or runner. Current posture checks and their scope are recorded in `../g2_navigation/validation-summary.json`.
