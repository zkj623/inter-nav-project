# Robot interaction perception

`AsyncInteractionPerception(config, backend=None, clock=time.monotonic, catalog=None)`
returns `(tuple[BoxState], RecognitionStore)` from `update(snapshot, now)`.
It shares the release `InteractionCatalog`, aliases, classifier labels, RGB crop
processing, `RecognitionEvidence` and `RecognitionStore`. Detection, segmentation
and classification still use the existing DINO, SAM and Qwen-VL HTTP interfaces.
The obsolete target-search and Qwen3 frontier-ranking clients were removed.

The input is the **captured** RGB-D frame in `snapshot.observation['sensors']['camera']`:
`rgba`, `depth`, `capture_step` and `pointcloud_image` (H × W × 3). The existing
dense `pointcloud` (H W × 3) is accepted and reshaped without dropping pixels.
XYZ must already be metric coordinates in `snapshot.planning_frame`; this module
does not apply another camera or robot transform. Calibrated geometry and a named
planning frame are required. `camera_at` is the capture time, not packet receipt
or inference completion time. Missing RGB-D, mismatched shapes, wrong declared
point frame, future captures and stale captures cannot supply interaction boxes.

A segmented object's finite points provide a measured axis-aligned bounding box,
expanded by `geometry_margin_m` (default 0.03 m per side). An axis with less than
0.04 m observed extent is rejected; padding cannot manufacture hidden thickness.
This is an envelope of visible returns. It does **not** establish full object
dimensions, mass, complete occluded surfaces or safe contact geometry. Therefore
`geometry_verified: false` is the default. In this state proposals are available
in `last_delivery` only and never enter the movable-object map or execution.
Enabling the flag means that the installed sensor's complete conservative bounds
have been separately accepted. It does not cause this client to reconstruct them.

IDs are local geometric tracks; scene names and simulator metadata are ignored.
Verified, fresh geometry can produce a live proposal after one frame, including
an unknown category, so the release object inspection planner can acquire more
views. A proposal is an obstacle, **not** a push permission. Two distinct, recent
RGB classifications of the same tracked object and catalog authorization are
required for `push_aside`. Unknown or conflicting categories revoke the preceding
permission. Ambiguous spatial matches start a new identity rather than inheriting
another object's permission. The engine's permission checks and planner's
interactable IDs must be retained. Robot motion acceptance is a separate gate.

One daemon worker and a single replaceable pending frame bound inference work.
Every HTTP timeout is capped by the remaining `result_max_age_s` capture budget
(default 8 s). `reset()` increments the task generation and discards queued and
late results; it cannot kill the already running HTTP request, but prevents the
next request in that pipeline. `close()` has the same cancellation behavior and
does not wait indefinitely for a service. No fallback grants interaction access.

The separate `input_max_age_s` (0.5 s) and `geometry_max_age_s` (0.75 s) apply to
camera capture time. Late classification may contribute recent semantic evidence,
but never renews old geometry. A slow model may therefore produce diagnostics and
category evidence while producing no usable live boxes. Increasing the geometry
timeout is not a substitute for fresh contact feedback. Every vote also expires
individually after `recognition_max_age_s` (10 s).

`last_delivery` contains per-object capture coordinates, source, classifier model,
score, frame ID, response and the actual classification crop in
`crop_jpeg_base64` (JPEG, maximum side 384 pixels; at most six candidates per frame).
The normal recorder can preserve that JSON payload and reconstruct its images
after the client exits. `image_path` refers to a temporary local crop unless an
explicit `evidence_dir` is configured; temporary paths do not survive `close()`.

Offline contract tests: `deploy/tests/test_interaction_perception.py`. These use
fake RGB-D and fake HTTP and do not establish camera calibration, classifier
accuracy, complete object bounds or physical push acceptance.
