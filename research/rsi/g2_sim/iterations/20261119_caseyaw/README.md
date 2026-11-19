# Hard case: contact layout, capture and leveling

This independent workspace contains 14 fresh regular physical trials, 2 independently verified passes, plus 1 diagnostic-only trial(s). Historical commonfix outputs are not new trials. Wall timestamps are retained in trial_index.json. Diagnostic trials are excluded from accepted baselines.

Old candidates already included yaw90. New yaw90 trials reproduce the earlier failure; explicit yaw180 planning found no viable path. Yaw270, narrower opening and steeper approach also failed. Reverse finger-pair orientation moves wrists from about 2.6cm ahead of the object center to about 2.5cm behind it; four common-start paths were planned and three distances physically tested.

The level harness changes capture to bilateral force >12N for three updates, a physical 15mm test lift, and four measured contact manifolds inside existing visual gap/witness tolerances. It adds opposite object-tilt feedback (gain0.4, <=8deg, <=1deg/s), commanding only robot joints. The damped variant reduces load-balance wrist rate from1.5 to0.4deg/s, averages five force samples, targets28N/side, and requires four forces>2N for eight settling updates. This uses privileged simulated object orientation. Final acceptance still requires four positive distal forces >=90%; geometric contact availability alone is not success. No attachment or object pose writes.

| Trial | Verified | Failure | Hold min lift m | Max hold tilt deg | Four force fraction |
| --- | --- | --- | --- | --- | --- |
| caseyaw_damped_trials/caseyaw_reverse_plans/case_handheld/candidate1_repeat1/result.json | True | None | 0.18461069464683533 | 2.7700244148301203 | 0.9333333333333333 |
| caseyaw_damped_trials/caseyaw_reverse_plans/case_handheld/candidate1_repeat2/result.json | True | None | 0.18461069464683533 | 2.7700244148301203 | 0.9333333333333333 |
| caseyaw_level_trials/caseyaw_reverse_plans/case_handheld/candidate0_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.1796443611383438 | 11.386281342586093 | 0.3333333333333333 |
| caseyaw_level_trials/caseyaw_reverse_plans/case_handheld/candidate1_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.18250274658203125 | 8.199778668164344 | 0.8666666666666667 |
| caseyaw_level_trials/caseyaw_reverse_plans/case_handheld/candidate3_repeat1/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.18165741860866547 | 12.187871106348831 | 0.23333333333333334 |
| caseyaw_level_trials/caseyaw_reverse_plans/case_handheld/candidate3_repeat2/result.json | False | LIFT_HOLD_CRITERIA_NOT_MET | 0.18165741860866547 | 12.187871106348831 | 0.23333333333333334 |
| caseyaw_trials/caseyaw_narrow90_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_other_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_plans/case_handheld/candidate1_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_reverse_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_reverse_plans/case_handheld/candidate1_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_reverse_plans/case_handheld/candidate3_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| caseyaw_trials/caseyaw_steep_plans/case_handheld/candidate0_repeat1/result.json | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |

Acceptance remains >=15cm lift, >=3s hold, <=20deg tilt, <=2cm relative drift, four distal forces >0.5N in >=90% of hold samples, visual gap<=1.5mm and witness<=2.5mm. Independent review checks visual floor clearance>5mm, nonhand geometry penetration<=1mm, nonhand impulse<=0.001Ns, fixed joints/root, torso margins and actual initial pose. Exact full q_start matches the other eleven objects. Original geometry/materials, 1kg mass assumption and effort limits retained; articulation solver64/16 as in basket baseline. Same-condition repeats are not randomized robustness tests. No hardware validation, policy training or data collection.

Prepare commonfix dependencies, overlay source/fixtures, run the active per-asset baseline in ../../ground_lift_training_config.json. Old successful baselines remain in their original archives. Full samples/videos remain at SHA-indexed server paths; selected review videos are appended to the existing Feishu document.
