# Symmetric tucked common home and continuous preparation

The old common reset was shared across objects but asymmetric between arms. This revision mirrors arm positions and reduces elbow-axis lateral span from1.053m to0.762m. Work torso angles remain[-40,142,-87,0,0]deg. Both wrists at about[x0.552,y+/-0.276,z0.726]m. Exact initial targets are identical across all twelve objects.

Preparation starts at the shipped scene_flat_g2_op.yaml joint defaults (degrees converted to radians; omitted joints zero; root at z0.04 fixed to world, unlike that YAML free-base mode). It drives all preparation joints through original angular ranges with a seventh-order profile, >=20s, peak speed<=40% published velocity, commanded acceleration<=0.5rad/s2 and jerk<=2rad/s3, zero endpoint velocity/acceleration/jerk. Analytical joint-velocity targets accompany position targets during preparation; zero velocity targets are restored afterward. After settling, torso joint limits lock at the work posture. Preparation is outside the16-dimensional arm/gripper policy. No reset or robot/object pose writes after default initialization. Hands approach the old audited per-object pregrasp. Original grasp/lift controllers are retained except the explicitly indexed basket force feedback and round-lidded-bin balance comparisons. Runtime camera-only top-view snapshots do not advance physics.

**Scope limitation:** existing ideal torso drives and pinned root are retained. Independent static gravity audit under original URDF50Nm torso caps gives maximum required226.0Nm (4.52x) along this preparation. Thus smooth simulation is NOT a demonstrated motor-feasible or hardware trajectory. This limitation is explicit in active configuration. Arms retain original effort caps; held non-arm joints/root are fixed as before. Do not use this preparation for hardware without a corrected torque model and separate validation.

All12 object-inclusive preparation/approach paths passed native and visual collision screening. Full physical results below; 11/12 independently verified with new home. Old12 successful asymmetric-home baselines remain archived, and are not counted as new-home success. New results do not establish randomized robustness. No collection/training started.

| Asset | Smooth preparation | Full lift verified | Failure | Hold lift min m | Tilt max deg | Four force fraction |
| --- | --- | --- | --- | --- | --- | --- |
| basket_rectangular | True | False | CONTACT_POINT_OFF_VISIBLE_SURFACE | None | None | None |
| basket_woven | True | True | None | 0.1980060189962387 | 2.326459520427783 | 1.0 |
| bin_lidded | True | True | None | 0.17981448769569397 | 0.912824358186453 | 1.0 |
| bin_pedal | True | True | None | 0.18506792187690735 | 12.039129334300252 | 0.9666666666666667 |
| bin_round_lidded | True | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None | None |
| bin_round_pedal | True | True | None | 0.1926090270280838 | 14.74429502573134 | 1.0 |
| box_reinforced | True | True | None | 0.1880885362625122 | 11.691639454250215 | 0.9 |
| box_white | True | True | None | 0.1801363155245781 | 13.264056490319003 | 1.0 |
| case_handheld | True | True | None | 0.1847236454486847 | 2.5991232719368167 | 1.0 |
| stool_rectangular | True | True | None | 0.1983066201210022 | 0.5154973888309566 | 1.0 |
| stool_wood_four_leg | True | True | None | 0.20312970876693726 | 0.7502306368926981 | 1.0 |
| table_side | True | True | None | 0.1997806429862976 | 1.0367223169903674 | 1.0 |
| bin_round_lidded | True | False | FOUR_FINGER_PRELOAD_NOT_ESTABLISHED | None | None | None |
| basket_rectangular | True | True | None | 0.18752914667129517 | 3.5563193946201057 | 1.0 |
| basket_rectangular | True | False | JOINT_TRACKING_ERROR | None | None | None |
| bin_round_lidded | True | False | NO_SETTLED_FOUR_FINGER_GRASP | None | None | None |

Original full lift acceptance is retained. Independent verification checks original angular limits for the entire trajectory, actual preparation tracking under the existing0.25rad runtime rejection, mirror error<5mm, original velocity bounds (including finite differences of actual sampled positions), commanded smoothness caps, non-assembly self-contact impulse<=0.001Ns, visual floor clearance, object nonhand collisions, frozen work joints/root, original work torso margins and force/tilt/lift/drift criteria. The additional exploratory0.1rad precise-following goal is recorded separately as precision_tracking_within_0p1_rad; it remains unmet in some20s trials. Smooth limited-speed movement and exact following are distinct observations; tracking lag is disclosed, never treated as zero. Physical samples/videos are SHA-indexed server artifacts; code/results/plans archived here. Source runtime_v1 stopped on invalid USD-prim references after preparation. Runtime_v2 had9.8-10.6deg maximum position-only tracking lag; runtime_v3 added velocity targets; runtime_v4 extends preparation to20s. Only v4-family outputs count toward this revision. The basket precise comparison raises articulation solver64/16 to128/32, retaining original geometry, materials, effort caps and visible-contact thresholds. Earlier incomplete debugging trials were stopped; they are not successful validations or new repeat evidence.

Basket force feedback retains solver64/16 and original collisions/materials/effort limits. It bounds gripper master target offsets to[-0.12,0]rad and opens against excessive force, targeting20N per hand. The prior right-side force exceeded100N at first lift and witness error reached2.622mm; the visible2.5mm threshold is unchanged. The round-bin comparisons use the existing case-style force/tilt balancing, either throughout feedback or only after original preload is established; all original force preload, settling and final acceptance remain unchanged. Tilt balancing uses privileged simulated object orientation. One basket-force launch ran out of GPU memory before producing a completed physical result; it was rerun in a fresh repeat2 folder and the incomplete launch is not repeat evidence.

Overlay prior commonfix/caseyaw dependencies, source and fixtures in one workspace; run the asset entry in ../../ground_lift_training_config.json. The simulator is IsaacSim4.2, 120Hz physics/10Hz sampled evidence and video; no hardware claim.
