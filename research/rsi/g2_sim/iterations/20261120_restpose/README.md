# Custom arms-down simulation standby

This replaces the preferred raised-hand scene startup with a custom upright, arms-down, mirror-symmetric standby. It is NOT an official G2 factory standby. Work-home joint targets are byte-for-byte unchanged from20261120_symhome; policy action dimensions remain16 and torso freezes only after preparation. Grippers idle at-0.3rad. Twenty-second seventh-order position and analytical velocity targets connect standby directly to the existing low work home, without pose resets after initialization.

Standby hold and preparation independently passed. All12 object-inclusive standby/preparation/approach paths passed native and visible collision screening at<=0.06rad increments. This revision physically tested only white box and hard case full workflows: 2/2 verified; other full workflows are not automatically revalidated from the new start. The historical11/12 lift baselines retain their prior startup and remain archived. The root validated_assets_scope and separate standby_end_to_end_validated_assets make that distinction explicit. No data collection/training started.

The first exploration mirrored arm link origins but missed the final wrist offset: full end-link matching showed joint6 uses the SAME angle sign across sides. The corrected map is[-1,+1,-1,+1,-1,+1,-1] for joints1..7; two generic configurations audit both end-link position and orientation using the local reflected end frame. The old v1 preparation sample failed the independent wrist mirror check (~3cm) and is excluded; two v1 example runs were stopped. Final v2 samples verify actual arm/end positions and orientation, original joint ranges, real sampled speed, self contact<=0.001Ns, settling errors and original full lift acceptance.

**Scope:** existing pinned root and ideal torso drive. Upright standby is modeled dynamically; preparation to the low work home still has the previously documented original-50Nm torso limit mismatch. No hardware feasibility claim. Command smoothness does not mean zero tracking lag; actual measured errors are indexed. The additional0.1rad precise-following goal from the preceding iteration is not asserted here.

| Asset | Workflow | Verified | Failure | Standby mirror error m | Prepared tracking rad |
| --- | --- | --- | --- | --- | --- |
| box_white | standby_hold_and_preparation | True | None | 0.0001953394869735189 | 0.0003590250015258478 |
| box_white | standby_preparation_and_grasp | True | None | 0.0001953394869735189 | 0.0003590250015258478 |
| case_handheld | standby_preparation_and_grasp | True | None | 0.00011989189240644472 | 0.0030633755707014654 |

Overlay previous symhome/caseyaw runtime dependencies, then source/fixtures here into one workspace. Preview: ISAAC_PY restpose_v2_box_white_trial.py --asset box_white --plan-dir restpose_v2_plans --pose-only --gpu GPU. Complete example: same command without --pose-only and with --repeat 2. Each asset's preferred runtime is indexed by ../../standby_preparation_config.json.execution. Original mass/materials/arm effort/contact limits remain unchanged. Samples/videos stay at SHA-indexed server paths; images and videos are in the existing Feishu review document.
