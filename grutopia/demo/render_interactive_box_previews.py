"""Render registered before/after cutaway top views and validate box dynamics."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--gpu', type=int, default=4)
parser.add_argument('--scenes', default='mv7,mv7dq,mvucby,mvuh5ai,mvuhwaq')
parser.add_argument('--skip-physics-check', action='store_true')
args = parser.parse_args()
out = Path(args.output)
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': args.gpu,
                     'multi_gpu': False, 'renderer': 'RayTracedLighting'})
try:
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    import cv2
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
    from omni.isaac.core.utils.semantics import add_update_semantics
    from grutopia_extension.interactive_navigation.scene_paths import configure_scene_mdl_paths
    from grutopia_extension.interactive_navigation.material_fallback import apply_koostruct_material_fallbacks
    from grutopia_extension.interactive_navigation.interactive_box_scenarios import author_box, CATALOG, save_json, digest
    cv2.setNumThreads(2)
    manifest = json.loads((out/'scenario_manifest.json').read_text())
    report = {'scene_xy_scale': 1.0, 'navigation_run': False, 'scenes': {}}

    for scene in args.scenes.split(','):
        print(json.dumps({'scene': scene, 'phase': 'label_overlay'}), flush=True)
        entry = manifest['scenes'][scene]
        folder = out/scene
        # Semantic labels are authored only in the newly generated overlay.
        asset_stage = Usd.Stage.Open(entry['after_scene'])
        for body in asset_stage.GetPrimAtPath('/Scene/InteractiveBoxes').GetChildren():
            add_update_semantics(body, 'movable_box')
        asset_stage.GetRootLayer().Save()
        assert digest(entry['original_asset']) == entry['original_sha256']
        print(json.dumps({'scene': scene, 'phase': 'load_preview_stage'}), flush=True)

        omni.timeline.get_timeline_interface().stop()
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(world.GetPrim())
        reference = UsdGeom.Xform.Define(stage, '/World/Scene')
        reference.GetPrim().GetReferences().AddReference(entry['after_scene'])
        stage.SetEditTarget(stage.GetSessionLayer())
        hidden = []
        overrides = []
        physics_scenes = []
        for prim in Usd.PrimRange(reference.GetPrim()):
            if prim.IsA(UsdPhysics.Scene): physics_scenes.append(str(prim.GetPath()))
            if prim.IsInstanceProxy(): continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI): overrides.append((str(prim.GetPath()), 'physics:rigidBodyEnabled'))
            if prim.HasAPI(UsdPhysics.CollisionAPI): overrides.append((str(prim.GetPath()), 'physics:collisionEnabled'))
        # Batch low-level layer edits; per-prim USD notices are quadratic on
        # these large referenced scenes. No original or overlay file is edited.
        with Sdf.ChangeBlock():
            for path in physics_scenes:
                Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path).active = False
            for path, name in overrides:
                spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
                attribute = spec.attributes.get(name)
                if attribute is None:
                    attribute = Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.Bool, Sdf.VariabilityUniform)
                attribute.default = False
        print(json.dumps({'scene': scene, 'phase': 'preview_physics_overrides_ready', 'count': len(overrides)}), flush=True)
        for path in ['/World/Scene/BaseScene/Meshes/Base/ceiling', '/World/Scene/BaseScene/__default_setting', '/World/Scene/BaseScene/lights']:
            prim = stage.GetPrimAtPath(path)
            if prim:
                if path.endswith('/lights'):
                    prim.SetActive(False)
                else:
                    UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        with np.load(folder/'geometry_grid.npz') as grid:
            floor_free = grid['free']
        limits = entry['limits']
        for prim in Usd.PrimRange(reference.GetPrim()):
            if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy(): continue
            path = str(prim.GetPath())
            if '/Base/ground/' not in path and not any('/'+x+'/' in path for x in ['light', 'ceiling_light']): continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty(): continue
            lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
            x0, y0 = np.maximum(np.floor((lo[:2] - [limits[0][0], limits[1][0]])/.025).astype(int), 0)
            x1, y1 = np.minimum(np.ceil((hi[:2] - [limits[0][0], limits[1][0]])/.025).astype(int)+1,
                                [floor_free.shape[1], floor_free.shape[0]])
            over_walkway = x1>x0 and y1>y0 and floor_free[y0:y1,x0:x1].mean() > .3
            fixture = any('/'+x+'/' in path for x in ['light', 'ceiling_light'])
            if lo[2] > 1.8 and (fixture or over_walkway):
                UsdGeom.Imageable(prim).MakeInvisible(); hidden.append(path)
        configure_scene_mdl_paths(entry['original_asset'])
        materials = apply_koostruct_material_fallbacks(stage, '/World/Scene/BaseScene')
        print(json.dumps({'scene': scene, 'phase': 'materials_ready'}), flush=True)
        dome = UsdLux.DomeLight.Define(stage, '/World/PreviewDome')
        dome.CreateIntensityAttr(1100)
        camera = UsdGeom.Camera.Define(stage, '/World/TopCamera')
        camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
        limits = entry['limits']
        center = [(limits[0][0]+limits[0][1])/2, (limits[1][0]+limits[1][1])/2]
        span = max(limits[0][1]-limits[0][0], limits[1][1]-limits[1][0]) + 1.2
        camera.CreateHorizontalApertureAttr(span*10)
        camera.CreateVerticalApertureAttr(span*10)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 1000))
        UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], span+10))
        product = rep.create.render_product('/World/TopCamera', (1600, 1600))
        rgb = rep.AnnotatorRegistry.get_annotator('LdrColor'); rgb.attach(product)
        instances = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation'); instances.attach(product)
        boxes = UsdGeom.Imageable(stage.GetPrimAtPath('/World/Scene/InteractiveBoxes'))
        images = {}
        visibility = {}
        for label in ['before', 'after']:
            print(json.dumps({'scene': scene, 'phase': 'render', 'image': label}), flush=True)
            boxes.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible if label=='before' else UsdGeom.Tokens.inherited)
            for _ in range(2): app.update()
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=True)
            data = rgb.get_data()
            if isinstance(data, dict): data = data['data']
            image = np.asarray(data).reshape(1600, 1600, 4)
            assert image[:, :, :3].std() > 3, 'Empty render'
            cv2.imwrite(str(folder/(label+'_topdown.png')), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
            images[label] = label+'_topdown.png'
            if label == 'after':
                data = instances.get_data()
                ids = np.asarray(data['data']).reshape(1600, 1600)
                labels = data['info']['idToLabels']
                for item in entry['objects']:
                    token = '/InteractiveBoxes/'+item['id']
                    matches = [int(key) for key, value in labels.items() if token in str(value)]
                    pixels = int(np.isin(ids, matches).sum())
                    parts = item.get('parts', [{'dimensions': item['dimensions']}])
                    area = sum(p['dimensions'][0]*p['dimensions'][1] for p in parts)
                    expected = area*(1600/span)**2
                    visibility[item['id']] = {'visible_pixels': pixels, 'expected_top_pixels': expected,
                                             'visible_fraction': pixels/max(expected, 1)}
                save_json(folder/'instance_labels.json', labels)
        rgb.detach(product); instances.detach(product); product.destroy()
        report['scenes'][scene] = {'images': images, 'camera_center': center, 'camera_span_m': span,
            'resolution': [1600, 1600], 'hidden_for_preview': hidden, 'visibility': visibility,
            'material_report': materials, 'original_hash_unchanged': True,
            'physics_disabled_in_preview_session_only': True}
        save_json(out/'preview_report.json', report)
        print(json.dumps({'scene': scene, 'phase': 'rendered', 'visibility': visibility}), flush=True)

    if not args.skip_physics_check:
        # A small separate world verifies that these are dynamic rigid bodies,
        # without cooking every mesh in all five source scenes for a static preview.
        omni.timeline.get_timeline_interface().stop()
        omni.usd.get_context().new_stage()
        from omni.isaac.core import World
        from omni.isaac.core.prims import RigidPrim
        simulation = World(stage_units_in_meters=1.0, physics_dt=.005, rendering_dt=.02,
                           backend='numpy', device='cpu')
        stage = omni.usd.get_context().get_stage()
        floor = UsdGeom.Cube.Define(stage, '/World/ValidationFloor')
        floor.CreateSizeAttr(1)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
        floor.AddScaleOp().Set(Gf.Vec3d(2*len(CATALOG)+4, 8, .1))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim()).CreateCollisionEnabledAttr(True)
        bodies = []
        for index, (shape, spec) in enumerate(CATALOG.items()):
            item = {'id': 'T'+str(index), 'position': [-(len(CATALOG)-1)+index*2, 0, spec['dimensions'][2]/2+.008],
                    'yaw_degrees': 0, 'interactions': ['push'], **spec}
            prim = '/World/ValidationBoxes/'+item['id']
            author_box(stage, prim, item, '/World/ValidationMaterials')
            body = RigidPrim(prim_path=prim, name=shape)
            simulation.scene.add(body); bodies.append((shape, spec, body))
        simulation.reset()
        for _ in range(240): simulation.step(render=False)
        validation = {}
        for shape, spec, body in bodies:
            position, quat = body.get_world_pose()
            mass = float(body.get_mass())
            assert abs(position[2]-spec['dimensions'][2]/2) < .02, (shape, position)
            assert abs(mass-spec['mass_kg']) < .01, (shape, mass)
            validation[shape] = {'mass_kg': mass, 'settled_position': position.tolist(), 'initial_orientation': quat.tolist()}
            body.set_linear_velocity(np.array([.35, 0, 0]))
        for _ in range(120): simulation.step(render=False)
        for shape, spec, body in bodies:
            position, _ = body.get_world_pose()
            start = np.array(validation[shape]['settled_position'])
            displacement = float(np.linalg.norm(position[:2]-start[:2]))
            assert displacement > .005, (shape, displacement)
            validation[shape].update(horizontal_motion_m=displacement, dynamic_motion_verified=True)
        simulation.stop()
        save_json(out/'box_physics_validation.json', {'passed': True, 'shapes': validation,
                  'scope': 'gravity settling and response to an initial horizontal velocity; no robot grasp test'})
    save_json(out/'render_status.json', {'status': 'complete', 'scenes': list(report['scenes']), 'navigation_run': False})

except Exception as error:
    import traceback
    traceback.print_exc()
    (out/'render_error.json').write_text(json.dumps({'type': type(error).__name__, 'error': str(error),
                                                   'traceback': traceback.format_exc()}, indent=2))
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
