# Remaining two common-start grasps

30 physical trials; 1 new objects independently verified; 11/12 total complete common-start baselines. The full initial q_start, body posture, mass, friction and actuator torque limits are unchanged. Existing ten successes use their archived baselines.

Case geometry candidates compare contact heights, wrist angles and stance distances, including forward clearance during lift. Basket candidates examine rim pinch and external four-finger support. Every candidate is connected to the identical common home through collision-audited joint motion; no teleportation or object pose driving. Runtime witness, collision, tracking and acceptance gates are retained. Load-control targets are controller parameters within original motor limits, not a change to the physical object.

Additional diagnostics compare softer arm gains (kp600/kd80) and gripper gains (kp15/kd2) with unchanged effort caps. Precise fingertip variants replace only the four distal collision meshes in each trial's prepared_scene: max384 hulls, 2M voxels, 0.1% requested error, shrinkwrap. Shared assets remain untouched. The first precision prototype omitted explicit material binding and is diagnostic-only; the accurate variant restores each original ContactMaterial (static1/dynamic0.9), and only material-preserving variants can be accepted. These geometry experiments are not a regression of the previous ten objects.

The solver variant retains original collision meshes/materials and raises articulation solver iterations from 32 position/8 velocity to 64/16; solverfine tests 256/64. They log native mimic residuals during actual simulation. The configuration prefers passing original-geometry solver trials over experimental replacement collision meshes.

The micro-load variant, only after bilateral contact, commands a 2cm forward/1cm upward wrist motion over a slow segment before four-finger refinement. This is a physical load-transfer experiment with the same IK/static/tracking/collision checks, not object pose driving; final lift/hold thresholds remain unchanged.

The transfer revision triggers on first bilateral contact during refinement; the earlier post-close micro-load variant did not trigger if bilateral contact was absent at that instant. The plane revision measures actual simulator distal link transforms and the visual mesh support patches, rather than assuming exact mimic geometry, and uses bounded wrist corrections. All executed variants are individually archived.

The holdangle variant uses gripper master position gains kp80/kd2 (from kp20/kd1), retaining original motor effort caps and target velocity limits. It tests whether load-induced master aperture error is the source of loss of one support finger. This is separately archived from the softer-gain variant.

The transfer8 revision waits for >8N bilateral contact before commanding 4cm forward/1cm up. The guide revision also retains at least 4cm forward displacement in the first two subsequent lift waypoints, so the following lift cannot immediately undo the clearance; its full joint path passes IK, effort, visible and native collision screening before simulation.

The center variant reads actual object x/y during preload, low-pass filters and clamps a shared wrist offset to +/-3cm, then freezes it after capture; it never follows object z or writes object state. This is privileged simulation-state feedback for the demonstration expert. Narrow/wide geometric searches regenerate pad support patches at master angles -0.4/-0.75 rad rather than merely changing the opening in an old trajectory; all approach paths are re-audited.

| Asset | Trial | Verified | Failure | Hold min lift m | Four-contact coverage |
| --- | --- | --- | --- | --- | --- |
| basket_rectangular | commonfix_accurate_trials/commonfix_pinch_plans/basket_rectangular/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.19431669265031815 | 1.0 |
| case_handheld | commonfix_accurate_trials/commonstart_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_center_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_compliant_trials/commonfix_compliant_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_guide_trials/commonfix_guide_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_holdangle_trials/commonstart_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_micro_trials/commonfix_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_plane_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| basket_rectangular | commonfix_precise_trials/commonfix_pinch_plans/basket_rectangular/candidate0_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| case_handheld | commonfix_precise_trials/commonstart_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_solver_trials/commonfix_high_plans/case_handheld/candidate1_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_solver_trials/commonfix_high_plans/case_handheld/candidate2_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_solver_trials/commonfix_narrow_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_solver_trials/commonfix_narrow_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_solver_trials/commonfix_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_solver_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_solver_trials/commonfix_wide_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| basket_rectangular | commonfix_solver_trials/commonstart_plans/basket_rectangular/candidate0_repeat1/result.json | True | None | 0.19023435562849045 | 1.0 |
| basket_rectangular | commonfix_solver_trials/commonstart_plans/basket_rectangular/candidate0_repeat2/result.json | True | None | 0.19023435562849045 | 1.0 |
| case_handheld | commonfix_solver_trials/commonstart_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_solverfine_trials/commonfix_high_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_solverfine_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_transfer8_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_transfer_trials/commonfix_plans/case_handheld/candidate0_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| basket_rectangular | commonfix_trials/commonfix_pinch_plans/basket_rectangular/candidate0_repeat1/result.json | False | CONTACT_POINT_OFF_VISIBLE_SURFACE | None | None |
| basket_rectangular | commonfix_trials/commonfix_pinch_plans/basket_rectangular/candidate1_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |
| case_handheld | commonfix_trials/commonfix_plans/case_handheld/candidate0_repeat1/result.json | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None |
| case_handheld | commonfix_trials/commonfix_plans/case_handheld/candidate1_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_trials/commonfix_plans/case_handheld/candidate2_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None |
| case_handheld | commonfix_trials/commonfix_plans/case_handheld/candidate3_repeat1/result.json | False | NONHAND_OBJECT_CONTACT | None | None |

Acceptance: lift >=15cm, hold >=3s, tilt <=20deg, relative drift <=2cm, four contact fraction >=90%; visible contact surfaces and runtime non-hand contacts checked. Independent review checks full visual floor clearance >5mm, sampled non-hand penetration <=1mm, fixed-body joint errors, original torso margins, root drift, actual initial pose and identical initial targets.

Prepare the preceding commonstart workspace, overlay this source and fixtures; shared assets remain read-only. Per-asset entries in ../../ground_lift_training_config.json specify the iteration/harness/plan/candidate/repeat. trial_index.json records all successes and failures and full video/sample server paths and SHA256. Fixed root/non-arm joints, assumed 1kg objects; no randomized robustness or hardware validation. Batch data collection awaits video review.
