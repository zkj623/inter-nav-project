# Genie2 forward-preferred navigation

This directory implements observed-only, noninteractive navigation for Genie2 CRS/Omnipicker. Its default posture is the exact `standby_joint_positions_rad` field in `genie2-rsi:research/rsi/g2_sim/standby_preparation_config.json`, bundled here with source commit and SHA-256 provenance. The adapter targets the minimum-5m GRScenes benchmark. Latest complete original-start retests cover **S03-T09** and **S07-T10**. See the current validation summary for the checks actually completed with this posture.

## Method

- Prefer turning and driving forward. Translation in body-relative direction `delta` has cost multiplier `1 + 4*abs(sin(delta)) + 2*max(0,-cos(delta))`; a 22.5 degree turn costs 2.5 grid steps. Thus a forward 5 cm step costs 1, a lateral step 5, and a reverse step 3. Lateral and reverse actions remain available for confined passages and short position adjustments.
- Execute planned turns at their checked lattice positions with zero translational command. Recheck the observed full-body sweep at intermediate angles no more than approximately 2.8 degrees apart, using a 5 cm margin. When that sweep is unavailable, stop and replan. Use a consistent 0.05 rad heading tolerance; small residual errors are corrected during forward motion. Extend lookahead to at most 20 cm only along a straight segment with unchanged planned heading, without cutting corners.
- Check turning space at the measured robot center, including its offset within the grid cell. A safe nearby pose can turn immediately; do not repeatedly center a robot that is already within a few centimeters of a safe waypoint. If centering is required, command 0.045–0.06 m/s rather than an ineffective sub-centimeter correction, then recheck the sweep.
- Load all 32 independent standby joint targets without changing any values, including the -0.3 rad gripper targets. Validate joint names and URDF limits. Prepare consistent link transforms, joint state, drives and mimics; retain these targets while navigating. Recompute the enabled native collision projection for this pose during preparation and load the run-specific footprint during execution. The planning grid is 5 cm; the pose lattice has 16 orientations. Translation uses a 2.5 cm footprint margin; rotation checks both adjacent poses and intermediate angles with a 5 cm margin.
- Fuse simulated range measurements from an upper scanner and four low chassis scanners. Every beam is queried synchronously against the PhysX collision scene. Each observation includes its own origin and actual hit position; no-hit endpoints contribute only free-ray evidence. Robot actor hits are removed by identity. The planner never loads an offline scene grid, reference route, or scene-geometry metadata.
- Preserve measured obstacles in this **static noninteractive** test. This prevents a high beam passing over a low object from erasing its low-level return. Dynamic interaction requires a separate moving-object update model.
- Search observed pose space for goal-region routes and observation poses. Revalidate the near portion of the path as new obstacle evidence arrives. Retain the task start, goal and 0.5 m tolerance.
- Drive four independently steered wheels. Choose the feasible steering angle nearest the measured angle among equivalent directions, with signed wheel speed. Project each rolling command onto its current wheel direction while steering; one unaligned wheel no longer disables all wheel drives. All joint effort limits remain those of the imported URDF (wheel 8 Nm, steering 15 Nm). Apply arm holding targets throughout navigation.

## Files

| File | Purpose |
|---|---|
| `prepare_g2_runtime.py` | Prepare the configured RSI standby robot and task stage from existing model/scene assets. |
| `run_g2_holonomic.py` | Physical execution, synchronous range simulation, logging and optional video. |
| `g2_pose_planning.py` | Pose-lattice planning, swept rotations and observation poses. |
| `g2_holonomic_navigation.py` | Legacy fixed-heading navigator; this runner uses its telemetry container initialization only and never calls its control update. |
| `g2_online_map.py` | G2-specific online evidence fusion. |
| `g2_collision_footprint.json` | Measured native-collision projection. |
| `standby_preparation_config.json`, `standby_pose_origin.json` | Original RSI configuration and its pinned source provenance; only the standby field is loaded. |
| `g2_standby_pose.py` | Validate standby targets and compute their enabled native collision projection. |
| `g2_task_recording.py`, `g2_visualization.py` | Adapted rsi-nav-v1 review renderer: first person, yaw-following overhead, global/local maps, accumulated point cloud. |
| `restore_g2_visuals.py`, `g2_preview_materials.py` | Recover original scene material networks, textures and UVs, with supported MDL-to-preview conversion and physics-material checks. |
| `audit_g2_navigation.py` | Audit completion, stopping, sampled continuity and contact forces. |
| `audit_forward_motion.py` | Measure actual displacement relative to measured body yaw, distinguishing forward, lateral and reverse movement. |
| `check_forward_planning.py` | Behavior checks for open-room forward travel, necessary lateral passage, small heading errors, and checked in-place turns. |

## Run

Use the existing Isaac Sim 4.2 environment and an accessible `rsi-nav-v1` checkout. This experiment is an adapter to those assets, not an independent installer. Set paths to the directories on the machine running the test:

```bash
export G2_ACCEPT_ROOT=/path/to/rsi-nav
export G2_MODEL_ROOT=/path/to/prepared-genie2-model
export G2_SCENE_RUNTIME=/path/to/benchmark-runtime/S01
export G2_VISUAL_SCENE="$G2_SCENE_RUNTIME/scene.usda"
export G2_TASK_ID=S01-T09
export G2_OUT=/path/to/new-run
export G2_GPU=0
export G2_RECORD=1
export OPENBLAS_NUM_THREADS=1

# The python command here must belong to the Isaac Sim environment.
python prepare_g2_runtime.py
python run_g2_holonomic.py
python audit_g2_navigation.py "$G2_OUT"
python audit_forward_motion.py "$G2_OUT"
```

The model directory must contain `generated/g2_crs_omnipicker.urdf`, `robot_assets/g2_contact_inertia_fixed.usdc` and its dependency closure. The scene directory must contain the benchmark `episodes.json`, `physics_only.usdc`, and the original visual `scene.usda`, including all referenced MDL and texture assets. The required kinematics helper is included here and may also exist in the prepared model directory. The preparation script initializes link transforms, joint state, drives and gripper mimic state consistently before execution. Runtime base-pose teleportation is not used. Use a fresh output directory for every run; the runner refuses an existing `result.json`.

Set `G2_RECORD=0` for a physical run without video. Both modes advance physics exclusively through `world.step(render=False)` at 120 Hz; optional `world.render()` calls cannot add physics steps. The controller and range fusion run at 10 Hz, global publication at 1 Hz, and the review video at 25 fps. Recorded results include measured physics elapsed time to check the timebase.

## Evidence and checks

The runner saves `result.json`, `trajectory.json`, `events.json`, `contacts.json`, online maps, source hashes and `visual-material-audit.json`. The audit requires error <=0.5 m, final linear speed <0.03 m/s, final angular speed <0.05 rad/s, ten consecutive quiet 10 Hz samples, body tilt <20 degrees, no sampled pose jump and no force-bearing non-support contact above 0.001 Ns. Ten quiet sample timestamps span 0.9 seconds; this is reported explicitly rather than claiming a longer measured hold.

Contact reports are not collision counts. The audit treats a force-bearing point as support when its absolute vertical normal exceeds 0.7 and its height is below 0.08 m; it reports the remaining force-bearing points separately. This geometric rule handles floor pieces whose actor names are `door` or `other`. Review the raw normals, positions and impulses when using a different terrain.

Videos are recordings of the same physical run, with source diffuse textures/colors converted to UsdPreviewSurface and review lighting. `restore_g2_visuals.py` copies source material networks and UV attributes into a visual namespace without importing source geometry; `visual-material-audit.json` checks texture availability and unchanged effective physics material parameters. Collision proxy meshes are hidden visually while remaining active physically. No cross-run camera composition is used for the final validation video. The point-cloud view uses actual range hits, excludes no-hit endpoints, and extends the acceptance renderer's height range to 0.035–1.8 m for G2. The original rsi-nav-v1 module hashes in a result are baseline provenance; `source-hashes.json` identifies the actual adapted modules.

See `validation-summary.json` for the validated run results and their scope.

For a five-second posture-only recording, use a fresh output directory and prepare it normally, then run `G2_STANDBY_CHECK=1 python run_g2_holonomic.py` followed by `python audit_standby_pose.py "$G2_OUT"`. This checks exact initialization, measured holding error, stability, self contacts and environment contacts. Leave `G2_STANDBY_CHECK` unset for navigation. Previously prepared runs must be prepared again after changing the default posture, because link transforms, mimic state and collision footprint depend on it.

Run `python check_forward_planning.py` for the eight synthetic planning/control behavior checks. `audit_forward_motion.py` classifies measured displacement relative to measured body yaw: forward within 30 degrees, lateral between 60 and 120 degrees, reverse beyond 120 degrees. Motion below 0.015 m/s is excluded from the distance fractions; turning in place is measured separately. These are measured motion statistics, not counts of issued commands.


## Original-start recovery and 2026-11-22 retest

Preparation also writes recovery-footprint.wkb from the union of actual enabled native robot collision-part mesh projections. If the ordinary planner rejects the initial pose, a single straight startup recovery of at most 0.30 m may be selected using current synchronized range returns and the observed map. Every sampled swept position is checked against the projected parts with 5 mm point clearance. The endpoint and its nearby longitudinal offsets must pass the ordinary planner envelope; handoff waits until the actual current pose passes it. New obstacle returns, absent returns, unknown swept space, heading deviation or the distance limit stop the recovery. Normal SE2 navigation is retained afterward. Task start/goal, standby posture and scene layout are not moved or replaced.

Both S03-T09 and S07-T10 reached their original goal regions and stopped in the same frozen corrected version. Actual paths were 6.791837 m and 4.872446 m; errors were 0.474185 m and 0.477260 m. Both also passed the original diagnostic audits, with no non-support force contact samples or force-bearing self contacts. These are two single-run functional results, not a full-suite reliability claim. See validation-summary.json.

Navigation success is result.json success (goal arrival plus stopped confirmation). navigation-audit.json preserves additional contact and physical diagnostics separately; its aggregate passed field is not the sole navigation success metric. The S04 turn/translation switching issue was not changed or retested here. Runtime requires Shapely in addition to the existing Isaac Sim, NumPy and SciPy dependencies.
