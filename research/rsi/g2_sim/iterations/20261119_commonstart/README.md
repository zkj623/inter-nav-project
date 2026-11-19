# Unified initial arm and gripper pose

10/12 objects pass the full common-start approach, grasp and lift trial. Full target q_start is identical across plans, including 14 arm and two gripper master joints. Body/root fixed as before.

Initial measured error is checked after 1.5 seconds of settling (<0.05 rad); collision paths sampled with <=0.06 rad increments; all physical collision/contact gates and lift acceptance retained. Previous successful pregrasp lifts are historical and do not substitute for these complete trials.

| Asset | Verified | Failure | Actual initial max error rad |
| --- | --- | --- | --- |
| basket_rectangular | False | LOADED_VISIBLE_SURFACE_GAP | 0.00031807661056515446 |
| basket_woven | True | None | 0.00031807661056515446 |
| bin_lidded | True | None | 0.00031807661056515446 |
| bin_pedal | True | None | 0.00031807661056515446 |
| bin_round_lidded | True | None | 0.00031807661056515446 |
| bin_round_pedal | True | None | 0.00031807661056515446 |
| box_reinforced | True | None | 0.00031807661056515446 |
| box_white | True | None | 0.00031807661056515446 |
| case_handheld | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | 0.00031807661056515446 |
| stool_rectangular | True | None | 0.00031807661056515446 |
| stool_wood_four_leg | True | None | 0.00031807661056515446 |
| table_side | True | None | 0.00031807661056515446 |

Prepare the workspace using ../20261119_liftpair/README.md, then overlay source and fixtures. Use each per-asset harness in config. Full videos and samples with SHA256 are indexed in trial_index.json. Mass 1kg assumed; fixed root/non-arm joints; random robustness/hardware not tested. Training collection awaits user review.
