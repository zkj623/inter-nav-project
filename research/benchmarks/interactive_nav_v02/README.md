# GRScenes × Genie2 fixed-layout benchmark v0.2 — floor support and route length revision

This research prototype contains **10 scenes, 10 fixed initial layouts, and 100 navigation tasks**. Each scene has movable rigid objects and ten distinct, non-reversed start/goal pairs: eight interaction-required tasks and two optimal non-interaction controls. Every episode resets all objects to the same scene-level initial poses. Task selection never changes the layout. The 15 asset types come from `genie2-rsi` commit `c9ce1a5ba405905f4d626fe630c485ca60fe128a`.

This benchmark is separate from the earlier stage-one delivery. Its platform is **Genie2/G2 CRS**, not Go2. It is a geometric task dataset and scene-loading prototype; it does not yet provide a validated G2 navigation/manipulation agent.

This revision sets a **5 m minimum reference-route length for every task**, including controls, while retaining short/medium/long strata and a 0.65 corridor-overlap limit. S08's raster and fixed object layout are rebuilt; S04 is also redesigned to accommodate longer controls; the other eight fixed object layouts are unchanged.

## S08 floor-support correction

The previous floor extractor used only the `Base/ground` semantic group. S08 contains actual coplanar architectural floor strips in `Base/other`; omitting them created false gaps around living/dining-room flooring. Robot-radius inflation then disconnected the upper and lower portions of the apartment, and largest-component sampling concentrated tasks in the upper rooms. Total projected floor area is therefore distinct from usable connected robot-center area.

`floor_support.py` additionally rasterizes existing architectural `Base` triangles within 0.04 m of the reference floor, with absolute vertical-normal cosine at least 0.98. It excludes furniture from support inference. No morphological hole filling, obstacle removal, robot-radius reduction, or change to the original USD scene is used. `fine_maps/S08.floor_support.json` lists the source meshes; `floor_support_validation.json` records the before/after connectivity and affected routes. The nominal radius remains 0.36 m.

The frozen cache retains the previous maps for S01–S07 and S09–S10 and the corrected S08 map. Reuse `fixtures/fine_maps` for this exact revision. A full source regeneration with the improved extractor may also update other scenes and creates a new dataset revision. To refresh S08 only in a copied cache:

```bash
cp -r fixtures/fine_maps fine_maps
python fine_maps.py --home "$HOME_ASSETS" --selected fixtures/selected.json --out fine_maps --only S08 --refresh
```

## Scope and task definitions

- The ten residential GRScenes maps span approximately 58–201 m² of projected ground area. This is not architectural floor area or commercial-scene coverage.
- `interaction_required`: fixed initial objects disconnect start and goal in an eight-connected, 2.5 cm raster using an optimistic 0.25 m navigation radius and obstacles projected from the 0.12–0.40 m height band. No initially reachable cell is within the 0.5 m goal tolerance. This searches the entire map rather than checking only one original route.
- `no_interaction_optimal`: an eight-neighbor shortest path in the map with added objects absent remains collision-free in the initial layout. Thus interaction cannot improve its geometric navigation-distance optimum under this model.
- A recorded cleared state relocates every object to its `parking_position_m`; no object is deleted. A route exists in the nominal 0.36 m radius model, using projected obstacles up to 1.8 m. Diagonal corner cutting is forbidden for reference routes.
- `minimum_relocations_in_optimistic_model` enumerates subsets of the recorded initial/parked object poses. It is not a global minimum over all physical manipulation actions. Objects may require carrying or other skills; a collision-free initial/terminal pose does not establish an executable transfer trajectory.
- Required-task route lengths describe the **all-parked reference state**, excluding robot approach, contact, and object transfer. They are not the optimal joint interaction/navigation cost. Control routes refer to the initial state.
- The G2 base mesh measures approximately 0.756 × 0.636 m. Circular projected models are screening abstractions, not an exact full-body configuration-space model. Rasterization projects intersecting triangles without clipping them at height-band boundaries. These results must not be reported as full-body collision validation or robot success rates.

The benchmark freezes the upstream rigid-object conventions: 1 kg mass, static/dynamic friction 1.0/0.9, no independently articulated lids or wheels. Dimensions are recomputed from asset vertices, including the corrected `basket_woven` size.

## Files

`fixtures/Sxx/episodes.json` is the authoritative fixed layout and ten tasks. The same directory contains navigation grids, maps, and the original-machine passive settling report. Root reports summarize geometric and physical checks. `viewer_data.json` and `index.html` provide an offline local review page. `tasks.csv` exposes classes, route lengths, and relocation counts.

The archive contains code, configurations, maps, and reports. It does **not** contain the original GRScenes or robot/object binaries. Preserve their source licenses and access terms. Asset identities and hashes are recorded in the included source audit.

## View the frozen benchmark

```bash
python -m http.server 8786 --bind 127.0.0.1 --directory fixtures
```

Open <http://127.0.0.1:8786/>. Select a scene and task, then toggle initial layout, candidate cleared state, and initial connectivity regions. Dashed routes in the initial state of required tasks are counterfactual cleared-state references, not currently traversable routes.

## Recreate scene stages on another machine

Use Python 3.10. The offline geometry tools were tested with numpy 1.26.0, scipy 1.11.2, opencv-python 4.11.0.86, Pillow, and usd-core 25.5.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.0 scipy==1.11.2 opencv-python==4.11.0.86 Pillow usd-core==25.5.1
```

Supply `HOME_ASSETS` as your GRScenes-100 `home_scenes` directory, containing `scenes/`, `models/`, and `Materials/`. Supply `OBJECT_ASSETS` as your authorized `genie2-rsi` object-assets directory containing the 15 `.usdc`/`.npz` pairs. These are explicit inputs; no server-specific directory is embedded in the frozen fixtures. The benchmark does not provide a public download mirror for restricted project assets.

```bash
python materialize.py --fixtures fixtures --home "$HOME_ASSETS" --objects "$OBJECT_ASSETS" --out runtime
python validate.py --out runtime --objects "$OBJECT_ASSETS"
python validate_usd.py --out runtime
python build_viewer.py --out runtime
```

`materialize.py` writes `scene.usda` and `cleared_scene.usda` with paths relative to your own asset directories. It resets physical validation to `not_run_on_this_machine`; recorded test results are not reused as new-machine successes. Geometry validation can run without Isaac Sim. Original appearance/material dependencies remain those of GRScenes.

## Passive placement validation in Isaac Sim

Isaac Sim 4.2.0.2 was used. Run offline USD preparation and simulation in **separate Python processes**; do not import standalone `pxr` before `SimulationApp` in the simulator process. Use the simulator's supported Python environment for `smoke.py`.

```bash
# Offline process: strip rendering shaders while retaining collisions/materials.
python prepare_physics.py --out runtime --objects "$OBJECT_ASSETS"
# Isaac Sim Python environment; choose an available GPU index.
python batch_physics.py --out runtime --smoke smoke.py --gpu 0
python build_viewer.py --out runtime
```

The smoke test advances 240 steps at 1/120 s and checks finite object poses, vertical displacement <0.2 m, planar drift <0.25 m, speed <0.1 m/s, and tilt <15°. Its scope is **initial-layout passive settling only**. It does not validate parked-state stability, robot collision, contact feasibility, grasping/pushing, or navigation success. Scene appearance is hidden only in the physics copies; original stages retain materials.

## Regenerate layouts and tasks

Frozen fixtures are preferred for reproducible evaluation. To regenerate the seeded placement search, use the included fine map cache or recompute it from the source scenes:

```bash
python fine_maps.py --home "$HOME_ASSETS" --selected fixtures/selected.json --out fine_maps
python design_multi_cut.py --source fixtures --fine fine_maps --objects "$OBJECT_ASSETS" --out designed
python validate_multicut.py --out designed --objects "$OBJECT_ASSETS"
python materialize.py --fixtures designed --home "$HOME_ASSETS" --objects "$OBJECT_ASSETS" --out generated
python validate.py --out generated --objects "$OBJECT_ASSETS"
```

The placement search fits object footprints to static geometry and rewards new splits of previously connected regions. An existing cut left unchanged by another object is not counted as a new blocking location. Each accepted layout must have at least three meaningful disconnected initial regions (each containing at least 0.65 m² of nominal endpoint core), at least three region-pair combinations among required tasks, and at least two objects used in selected clearing witnesses. Parking must preserve usable endpoint space in at least three initial regions while restoring a nominal connected route network. `validate_multicut.py` recomputes these conditions and checks static/object collisions of initial and parked raster footprints.

`layout_validation.json` records the topology and each object's role: which tasks become connected when only that object moves, and which remain blocked when every other object is parked. These are binary pose-state connectivity tests, not physical object-transfer trajectories. `diversify_tasks.py` selects spatially distributed tasks within a frozen layout and is also used by the placement builder. Future diversity should be added through new scenes, retaining one frozen layout per scene.

## Task diversity revision

Farthest-point samples cover the feasible navigation component. Up to ten endpoint regions are seeded at least 2 m apart in geodesic distance; these are spatial coverage bins, not semantic room labels. Short, medium, and long tasks are defined relative to each scene's candidate routes. The eight required tasks target a 3/3/2 split, with a flexible allocation of one to four tasks per stratum if the feasible route distribution requires it. Both controls are selected for additional coverage.

No endpoint position is used more than twice. Route similarity is checked in both directions using the Jaccard overlap of 0.25 m rasterized route corridors with a one-cell buffer. Every pair of selected tasks has corridor IoU at most 0.65; this is not the fraction of identical trajectory length. `task_diversity.json` and `diversity_validation.json` report the complete scene-wise distributions and checks of the frozen tasks. `coverage.json` separately records asset presence and use in selected clearing plans.

The viewer opens in **all-task overview** mode, with numbered starts and goals. Select one task to inspect its route, length stratum, initial connected regions (C labels), and selected clearing objects. R labels describe spatial endpoint bins; C labels describe actual initial connectivity in the optimistic model. The multiple-bottleneck revision changes object poses, so initial passive placement tests must be rerun for these layouts. G2 robot execution is still pending.
