"""World-metre box layouts and USD overlays for the five original Go2 scenes."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import cv2
from scipy import ndimage
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

SCENES = ('mv7', 'mv7dq', 'mvucby', 'mvuh5ai', 'mvuhwaq')
REPO = Path('/data20t/embodied/zkj/inter-nav-project')
RESULTS = REPO.parent / 'inter-nav-results'
RESOLUTION = 0.025
CATALOG = {
    'cube': {'dimensions': [.35, .35, .35], 'mass_kg': 1.2, 'tape': [.25, .45, .85]},
    'long': {'dimensions': [.75, .25, .28], 'mass_kg': 1.8, 'tape': [.85, .35, .15]},
    'flat': {'dimensions': [.65, .45, .26], 'mass_kg': 1.5, 'tape': [.15, .65, .35]},
    'tall': {'dimensions': [.28, .28, .60], 'mass_kg': 1.0, 'tape': [.70, .20, .45]},
    'medium': {'dimensions': [.60, .42, .45], 'mass_kg': 2.0, 'tape': [.85, .70, .15]},
    'large_high': {'dimensions': [.85, .55, .95], 'mass_kg': 6.0, 'tape': [.20, .65, .85]},
    'large_wide': {'dimensions': [.85, .65, .70], 'mass_kg': 5.0, 'tape': [.65, .40, .85]},
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''): h.update(block)
    return h.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def base_profile(scene):
    if scene == 'mv7':
        path = RESULTS / 'mv7-height05-inflation030-035-cloud080-20k-20261019/profiles/mv7.json'
    else:
        path = RESULTS / 'four-scenes-height05-inflation030-035-cloud080-20k-20261019/profiles' / (scene + '.json')
    profile = json.loads(path.read_text())
    assert profile['scene_scale'] == [.01, .01, .01]
    return profile, path

def pixel_polygon(points, limits, resolution=RESOLUTION):
    return np.rint((np.asarray(points) - np.array([limits[0][0], limits[1][0]])) / resolution).astype(np.int32)

def rectangle(xy, dimensions, yaw=0):
    w, d = dimensions[:2]
    local = np.array([[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]])
    angle = math.radians(yaw)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return local @ rot.T + xy

def mesh_arrays(stage, scope, profile):
    root = stage.GetPrimAtPath(scope)
    if not root: return
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh): continue
        mesh = UsdGeom.Mesh(prim)
        raw = mesh.GetPointsAttr().Get()
        if raw is None or len(raw) < 3: continue
        points = np.asarray(raw, dtype=np.float64)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = (points @ matrix[:3, :3] + matrix[3, :3]) * profile['scene_scale'] + profile['scene_position']
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not len(counts) or len(indices) != counts.sum(): continue
        if np.all(counts == 3):
            faces = indices.reshape(-1, 3)
        else:
            faces = []; offset = 0
            for count in counts:
                face = indices[offset:offset + count]; offset += count
                faces.extend((face[0], face[j], face[j+1]) for j in range(1, count-1))
            faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(faces) and np.isfinite(points).all(): yield str(prim.GetPath()), points, faces

def geometry_grid(profile):
    limits = [profile['mapping']['x_limits'], profile['mapping']['y_limits']]
    width = int(math.ceil((limits[0][1] - limits[0][0]) / RESOLUTION)) + 1
    height = int(math.ceil((limits[1][1] - limits[1][0]) / RESOLUTION)) + 1
    ground = np.full((height, width), -np.inf, dtype=np.float32)
    walls = np.zeros((height, width), dtype=np.uint8)
    stage = Usd.Stage.Open(str(REPO / profile['scene_asset_path']))
    counts = {'floor_meshes': 0, 'wall_meshes': 0, 'wall_triangle_projections': 0}
    for _, points, faces in mesh_arrays(stage, '/Root/Meshes/Base/ground', profile):
        counts['floor_meshes'] += 1
        tri = points[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(normals, axis=1)
        planar = (length > 1e-8) & (np.abs(normals[:, 2]) > .95 * length)
        planar &= (tri[:, :, 2].max(axis=1) < .18) & (tri[:, :, 2].min(axis=1) > -.15)
        planar &= np.ptp(tri[:, :, 2], axis=1) < .015
        for t in tri[planar]:
            polygon = pixel_polygon(t[:, :2], limits)
            x0, y0 = np.maximum(polygon.min(axis=0), 0)
            x1, y1 = np.minimum(polygon.max(axis=0) + 1, [width, height])
            if x1 <= x0 or y1 <= y0: continue
            mask = np.zeros((y1-y0, x1-x0), np.uint8)
            cv2.fillConvexPoly(mask, polygon - [x0, y0], 1)
            patch = ground[y0:y1, x0:x1]
            patch[mask.astype(bool)] = np.maximum(patch[mask.astype(bool)], float(t[:, 2].mean()))
    assert np.isfinite(ground).sum() * RESOLUTION**2 > 10, counts
    # GRScenes walls are separate open face meshes, not watertight solids.
    # Project faces intersecting box height, including architectural panels
    # stored in Base/other, rather than requiring closed mesh sections.
    for scope in ['/Root/Meshes/Base/wall', '/Root/Meshes/Base/other']:
        for _, points, faces in mesh_arrays(stage, scope, profile):
            counts['wall_meshes'] += 1
            triangles = points[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            chosen = (triangles[:, :, 2].min(axis=1) <= .70) & (triangles[:, :, 2].max(axis=1) >= .10)
            chosen &= (lengths > 1e-8) & (np.abs(normals[:, 2]) < .8 * lengths)
            for triangle in triangles[chosen]:
                polygon = pixel_polygon(triangle[:, :2], limits)
                cv2.fillConvexPoly(walls, polygon, 1)
                cv2.polylines(walls, [polygon], True, 1, 2)
                counts['wall_triangle_projections'] += 1
    assert walls.any(), counts
    return ground, walls.astype(bool), limits, counts

def free_grid(profile, ground, walls, max_height=.65):
    limits = [profile['mapping']['x_limits'], profile['mapping']['y_limits']]
    occupied = walls.astype(np.uint8).copy()
    metadata = json.loads((REPO / profile['static_obstacle_metadata_path']).read_text())
    for item in metadata.values():
        lo, hi = np.array(item['min_points']), np.array(item['max_points'])
        if lo[2] >= max_height + .05 or hi[2] < .04: continue
        polygon = pixel_polygon([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], limits)
        cv2.fillConvexPoly(occupied, polygon, 1)
    occupied = ndimage.binary_dilation(occupied, iterations=2)
    free = np.isfinite(ground) & ~occupied
    labels, count = ndimage.label(free)
    sizes = np.bincount(labels.ravel()) * RESOLUTION**2
    free &= sizes[labels] > 1.0
    distance = ndimage.distance_transform_edt(free) * RESOLUTION
    return free, distance, metadata

def placement_valid(xy, yaw, spec, profile, free, ground, existing, min_center_spacing=1.7):
    if np.linalg.norm(np.array(xy) - profile['start'][:2]) < 1.1: return None
    if any(np.linalg.norm(np.array(xy) - o['position'][:2]) < min_center_spacing for o in existing): return None
    # Check actual oriented footprints, including a 6 cm surface clearance,
    # so tighter doorway layouts do not rely on arbitrary centre spacing.
    padded = np.array(spec['dimensions'], float); padded[:2] += .12
    footprint_xy = rectangle(xy, padded, yaw).astype(np.float32)
    for other in existing:
        other_xy = rectangle(other['position'][:2], other['dimensions'], other['yaw_degrees']).astype(np.float32)
        area, _ = cv2.intersectConvexConvex(footprint_xy, other_xy)
        if area > 1e-6: return None
    limits = [profile['mapping']['x_limits'], profile['mapping']['y_limits']]
    polygon = pixel_polygon(rectangle(xy, spec['dimensions'], yaw), limits)
    x0, y0 = polygon.min(axis=0); x1, y1 = polygon.max(axis=0) + 1
    if x0 < 0 or y0 < 0 or x1 >= free.shape[1] or y1 >= free.shape[0]: return None
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.fillConvexPoly(mask, polygon - [x0, y0], 1)
    footprint = mask.astype(bool)
    if not free[y0:y1, x0:x1][footprint].all(): return None
    z = ground[y0:y1, x0:x1][footprint]
    if not len(z) or np.ptp(z) > .015: return None
    return float(z.max())

def choose_layout(profile, ground, walls):
    free, distance, metadata = free_grid(profile, ground, walls)
    limits = [profile['mapping']['x_limits'], profile['mapping']['y_limits']]
    selected = []
    doors = [o for o in metadata.values() if o['category'] == 'door']

    def accept(shape, xy, yaw, site, near_door=None, width=None):
        spec = CATALOG[shape]
        floor_z = placement_valid(xy, yaw, spec, profile, free, ground, selected)
        if floor_z is None: return False
        item = {'id': f'B{len(selected)+1:02d}', 'shape': shape, **spec,
                'position': [float(xy[0]), float(xy[1]), floor_z + spec['dimensions'][2]/2 + .004],
                'floor_height_m': floor_z, 'yaw_degrees': float(yaw), 'site_type': site,
                'near_door': near_door, 'passage_width_estimate_m': width,
                'interactions': ['push', 'bimanual_grasp_candidate']}
        selected.append(item)
        return True

    for shape in ['cube', 'long']:
        candidates = []
        for door in doors:
            lo, hi = np.array(door['min_points']), np.array(door['max_points'])
            size = hi[:2] - lo[:2]; along = int(np.argmax(size)); normal = 1-along
            if not .65 <= size[along] <= 1.35 or size[along] < CATALOG[shape]['dimensions'][0] + .04: continue
            yaw = 0 if along == 0 else 90
            for side in [-1, 1]:
                for setback in [.38, .50, .65, .85]:
                    xy = np.array(door['position'][:2]); xy[normal] += side * setback
                    if placement_valid(xy, yaw, CATALOG[shape], profile, free, ground, selected) is None: continue
                    spread = min([np.linalg.norm(xy - o['position'][:2]) for o in selected] or [2.0])
                    candidates.append((spread - .2*setback, xy, yaw, door))
        assert candidates, f'No valid {shape} doorway placement'
        _, xy, yaw, door = max(candidates, key=lambda c: c[0])
        assert accept(shape, xy, yaw, 'door_approach', door['instance_id'],
                      float(max(np.array(door['max_points'][:2]) - door['min_points'][:2])))

    ridge = free & (distance >= ndimage.maximum_filter(distance, size=5) - 1e-6)
    for shape, band, site in [('flat', (.65, 1.3), 'wide_passage'),
                              ('medium', (.40, .625), 'narrow_passage'),
                              ('tall', (.32, .9), 'passage_junction')]:
        cells = np.argwhere(ridge & (distance >= band[0]) & (distance <= band[1]))
        candidates = []
        for row, col in cells:
            xy = np.array([limits[0][0] + col*RESOLUTION, limits[1][0] + row*RESOLUTION])
            door_distance = min(np.linalg.norm(xy - o['position'][:2]) for o in doors)
            if door_distance > 3.0: continue
            for yaw in [0, 90]:
                if placement_valid(xy, yaw, CATALOG[shape], profile, free, ground, selected) is None: continue
                spread = min(np.linalg.norm(xy - o['position'][:2]) for o in selected)
                candidates.append((spread - .25*door_distance, xy, yaw, 2*float(distance[row, col])))
        assert candidates, f'No valid {shape} passage placement'
        _, xy, yaw, width = max(candidates, key=lambda c: c[0])
        assert accept(shape, xy, yaw, site, width=width)
    return add_large_door_boxes(profile, ground, walls, selected), free, distance


def add_large_door_boxes(profile, ground, walls, selected):
    """Place substantial single cartons across two distinct room approaches."""
    free, _, metadata = free_grid(profile, ground, walls, max_height=1.0)
    doors = [o for o in metadata.values() if o['category'] == 'door']
    used = set()
    for shape, preferred_room in [('large_high', '1/bedroom'), ('large_wide', '0/bedroom')]:
        original = CATALOG[shape]
        width, depth, height = original['dimensions']
        variants = [(original['dimensions'], 1.7), (original['dimensions'], .9),
                    ([.80, depth, height], .9), ([.75, depth, height], .9),
                    ([.75, depth-.10, height], .9)]
        for dimensions, spacing in variants:
            spec = {**original, 'dimensions': list(dimensions)}
            candidates = []
            for door in doors:
                if door['instance_id'] in used: continue
                lo, hi = np.array(door['min_points']), np.array(door['max_points'])
                size = hi[:2] - lo[:2]; along = int(np.argmax(size)); normal = 1-along
                if not spec['dimensions'][0] + .03 <= size[along] <= 1.35: continue
                yaw = 0 if along == 0 else 90
                for side in [-1, 1]:
                    # Stay outside the actual door leaf and directly in its approach.
                    for setback in [.50, .55, .60, .65, .70, .75, .85]:
                        xy = np.array(door['position'][:2]); xy[normal] += side * setback
                        floor_z = placement_valid(xy, yaw, spec, profile, free, ground, selected, spacing)
                        if floor_z is None: continue
                        gap = setback - size[normal]/2 - spec['dimensions'][1]/2
                        if not .04 <= gap <= .25: continue
                        score = (100 if door.get('room') == preferred_room else 0) + (10 if side == -1 else 0) - gap
                        candidates.append((score, xy, yaw, floor_z, door, gap, float(size[along])))
            if candidates: break
        assert candidates, f'No valid large carton directly at a doorway: {shape}'
        _, xy, yaw, floor_z, door, gap, width = max(candidates, key=lambda c: c[0])
        selected.append({'id': f'B{len(selected)+1:02d}', 'shape': shape, **spec,
            'position': [float(xy[0]), float(xy[1]), floor_z + spec['dimensions'][2]/2 + .004],
            'floor_height_m': floor_z, 'yaw_degrees': float(yaw), 'site_type': 'room_entry_obstruction',
            'near_door': door['instance_id'], 'door_room': door.get('room'),
            'gap_to_door_bbox_m': float(gap), 'passage_width_estimate_m': width,
            'projected_door_width_coverage': spec['dimensions'][0]/width,
            'interactions': ['push', 'bimanual_grasp_candidate']})
        if spacing != 1.7: selected[-1]['placement_center_spacing_m'] = spacing
        if spec['dimensions'] != original['dimensions']:
            selected[-1]['dimensions_adjusted_for_entry'] = True
        used.add(door['instance_id'])
    return selected

def make_material(stage, path, color, physics=False):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.85)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    if physics:
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(.6); api.CreateDynamicFrictionAttr(.5); api.CreateRestitutionAttr(0)
    return material

def author_box(stage, root_path, item, material_root='/Scene/BoxMaterials'):
    body = UsdGeom.Xform.Define(stage, root_path)
    body.AddTranslateOp().Set(Gf.Vec3d(*item['position']))
    body.AddRotateZOp().Set(item['yaw_degrees'])
    api = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    api.CreateRigidBodyEnabledAttr(True); api.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(item['mass_kg'])
    body.GetPrim().SetCustomDataByKey('inter_nav:object_id', item['id'])
    body.GetPrim().SetCustomDataByKey('inter_nav:interactable', True)
    body.GetPrim().CreateAttribute('inter_nav:interaction_modes', Sdf.ValueTypeNames.StringArray).Set(item['interactions'])
    brown = make_material(stage, material_root + '/' + item['id'] + '_Cardboard', [.48, .28, .12], True)
    tape = make_material(stage, material_root + '/' + item['id'] + '_Tape', item['tape'])
    assert 'parts' not in item, 'Each object must be one rectangular carton'
    parts = [{'dimensions': item['dimensions'], 'offset': [0, 0, 0]}]
    for index, part in enumerate(parts):
        cube = UsdGeom.Cube.Define(stage, root_path + f'/Carton{index+1}')
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(Gf.Vec3d(*part['offset']))
        cube.AddScaleOp().Set(Gf.Vec3d(*part['dimensions']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(brown)
        cube.GetPrim().CreateRelationship('material:binding:physics').SetTargets([brown.GetPath()])
        stripe = UsdGeom.Cube.Define(stage, root_path + f'/Tape{index+1}')
        stripe.CreateSizeAttr(1)
        offset = np.array(part['offset'], float); offset[2] += part['dimensions'][2]/2 + .001
        stripe.AddTranslateOp().Set(Gf.Vec3d(*map(float, offset)))
        stripe.AddScaleOp().Set(Gf.Vec3d(part['dimensions'][0]*.14, part['dimensions'][1]*.95, .002))
        UsdShade.MaterialBindingAPI.Apply(stripe.GetPrim()).Bind(tape)
    return body.GetPrim()

def author_scene(path, profile, items=()):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdGeom.Xform.Define(stage, '/Scene'); stage.SetDefaultPrim(scene.GetPrim())
    base = UsdGeom.Xform.Define(stage, '/Scene/BaseScene')
    base.GetPrim().GetReferences().AddReference(str((REPO / profile['scene_asset_path']).resolve()))
    base.AddTranslateOp().Set(Gf.Vec3d(*profile['scene_position']))
    base.AddScaleOp().Set(Gf.Vec3d(*profile['scene_scale']))
    scene.GetPrim().SetCustomDataByKey('inter_nav:original_scale_factor', 1.0)
    if items:
        UsdGeom.Xform.Define(stage, '/Scene/InteractiveBoxes')
        for item in items: author_box(stage, '/Scene/InteractiveBoxes/' + item['id'], item)
    stage.GetRootLayer().Export(str(path))

def prepare_all(output, scenes=SCENES, source_layout_root=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = {'scene_xy_scale': 1.0, 'original_assets_modified': False,
                'objects_per_scene': 7, 'all_objects_single_rectangular_cartons': True,
                'catalog': CATALOG, 'scenes': {}}
    for scene in scenes:
        folder = output / scene; folder.mkdir(exist_ok=True)
        profile, profile_path = base_profile(scene)
        asset = (REPO / profile['scene_asset_path']).resolve(); original_hash = digest(asset)
        save_json(folder/'original_profile.json', profile)
        print(json.dumps({'scene': scene, 'phase': 'geometry'}), flush=True)
        if source_layout_root:
            source = Path(source_layout_root)/scene
            previous = json.loads((source/'layout.json').read_text())
            with np.load(source/'geometry_grid.npz') as grid:
                ground, walls = grid['ground_height'], grid['walls']
            counts = previous['geometry_counts']
            limits = [profile['mapping']['x_limits'], profile['mapping']['y_limits']]
            items = []
            for previous_item in previous['objects']:
                item = dict(previous_item)
                if item['shape'] == 'l_shape':
                    item.pop('parts', None)
                    item.update(shape='medium', **CATALOG['medium'])
                    item['position'] = list(item['position'])
                    item['position'][2] = item['floor_height_m'] + item['dimensions'][2]/2 + .004
                items.append(item)
            free, distance, _ = free_grid(profile, ground, walls, max_height=1.0)
            for i, item in enumerate(items):
                assert placement_valid(item['position'][:2], item['yaw_degrees'], item,
                                       profile, free, ground, items[:i]) is not None, item['id']
            items = add_large_door_boxes(profile, ground, walls, items)
        else:
            ground, walls, limits, counts = geometry_grid(profile)
            items, free, distance = choose_layout(profile, ground, walls)
        np.savez_compressed(folder/'geometry_grid.npz', ground_height=ground, walls=walls, free=free, clearance=distance)
        save_json(folder/'layout.json', {'scene': scene, 'scale_factor': 1.0, 'objects': items, 'geometry_counts': counts})
        author_scene(folder/'before_scene.usda', profile)
        author_scene(folder/'after_scene.usda', profile, items)
        for label in ['before', 'after']:
            augmented = dict(profile)
            augmented.update(name=f'{scene}_{label}_boxes_original_scale', scene_asset_path=str(folder/(label+'_scene.usda')),
                             scene_scale=[1, 1, 1], scene_position=[0, 0, 0])
            save_json(folder/(label+'_profile.json'), augmented)
        assert digest(asset) == original_hash
        manifest['scenes'][scene] = {'original_asset': str(asset), 'original_sha256': original_hash,
            'profile_source': str(profile_path), 'limits': limits, 'objects': items,
            'before_scene': str(folder/'before_scene.usda'), 'after_scene': str(folder/'after_scene.usda')}
        save_json(output/'scenario_manifest.json', manifest)
        print(json.dumps({'scene': scene, 'phase': 'prepared', 'objects': len(items)}), flush=True)
    return manifest
