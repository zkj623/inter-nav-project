"""Copy selected existing GRScenes assets and dependencies; never run physics."""
import argparse
import hashlib
import json
import re
import shutil
import numpy as np
from pathlib import Path
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf

SELECTION = [
    ('bin_lidded', 'trashcan', '034a4c4664e66477a2deacfe4347742a'),
    ('bin_pedal', 'trashcan', 'c4fc0d2d39c4cc84eef3dc87dc9e2400'),
    ('stool_wood_four_leg', 'stool', '661ae8e648ca4a0aeef8dac2b7fb968c'),
    ('stool_rectangular', 'stool', 'b948ee8a683ef3abef37609da599b963'),
    ('stool_small', 'stool', '69e27aee25ac30acbaf07d2020c4156c'),
    ('box_white', 'box', '9467e0245df7cd93179b5a3a64b61cd2'),
]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    selected_categories = {item['category'] for item in selection}
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    home = source / 'scenes/GRScenes-100/home_scenes'
    inventory = []
    for file in sorted((source / 'benchmark/meta').glob('*/object_dict.json')):
        for key, value in json.loads(file.read_text()).items():
            if value.get('category') not in selected_categories:
                continue
            mid = key.split('/')[-1].removeprefix('model_').rsplit('_', 1)[0]
            inventory.append(dict(model_id=mid, category=value['category'], scene=file.parent.name,
                                  instance=key, scope=value['scope'],
                                  scene_bbox_dimensions_m=[b-a for a,b in zip(value['min_points'],value['max_points'])]))
    write_json(output/'catalog_search.json', inventory)
    shutil.copy2(source/'README.md', output/'GRScenes_SOURCE_README.md')
    manifest = dict(source='https://huggingface.co/datasets/OpenRobotLab/GRScenes',
                    license='CC-BY-NC-SA-4.0', physics_executed=False, experiment_status='paused',
                    origin='Existing GRScenes household USDs for Isaac Sim; not NVIDIA stock assets',
                    assets=[])
    for selected in selection:
        asset_id, category, mid = selected['id'], selected['category'], selected['model_id']
        file = next((home/'models/object').glob('*/'+category+'/'+mid+'/instance.usd'))
        dest = output/'assets'/asset_id
        raw = dest/'source'
        raw.mkdir(parents=True, exist_ok=True)
        copied = []
        pending = [Path('instance.usd'), Path('metadata.json')]
        pending += [Path(x) for x in json.loads(file.with_name('metadata.json').read_text())['materials']]
        visited = set()
        while pending:
            relative = pending.pop()
            # Preserve original relative paths, including MDL's relative module imports.
            relative = Path(*[part for part in relative.parts if part != '.'])
            if relative in visited:
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unexpected external dependency: {relative}')
            visited.add(relative)
            original = file.parent/relative
            if not original.is_file():
                raise FileNotFoundError(original)
            target = raw/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            checksum = digest(original)
            assert digest(target) == checksum
            copied.append(dict(path=target.relative_to(output).as_posix(), sha256=checksum, bytes=target.stat().st_size))
            if relative.suffix.lower() == '.mdl':
                text = original.read_text(encoding='utf-8-sig')
                for module in re.findall(r'(?:using|import)\s+\.::([A-Za-z0-9_]+)', text):
                    pending.append(relative.parent/(module+'.mdl'))
                for texture in re.findall(r'texture_\w+\s*\(\s*"([^"\n]+)"', text):
                    if texture and not texture.startswith('::'):
                        pending.append(relative.parent/texture)
        src_stage = Usd.Stage.Open(str(file))
        src_root = src_stage.GetDefaultPrim()
        # Derive bounds from visible vertices, not potentially stale source extents.
        cache = UsdGeom.XformCache()
        vertex_bounds = []
        corrected_extents = []
        for prim in src_stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                continue
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            if not len(points):
                continue
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            points_world = points @ matrix[:3,:3] + matrix[3,:3]
            vertex_bounds.extend([points_world.min(axis=0), points_world.max(axis=0)])
            local_extent = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
            corrected_extents.append((str(prim.GetPath().MakeRelativePath(src_root.GetPath())), local_extent))
        lo_array, hi_array = np.min(vertex_bounds,axis=0), np.max(vertex_bounds,axis=0)
        source_bbox = Gf.Range3d(Gf.Vec3d(*lo_array), Gf.Vec3d(*hi_array))
        scale = UsdGeom.GetStageMetersPerUnit(src_stage)
        # These assets' internal transforms already orient the geometry Z-up.
        # Their home scene is Z-up; individual file metadata says Y-up.
        # Do not rotate them a second time. Validate against scene dimensions below.
        lo, hi = source_bbox.GetMin(), source_bbox.GetMax()
        offset = Gf.Vec3d(-(lo[0]+hi[0])*scale/2, -(lo[1]+hi[1])*scale/2, -lo[2]*scale)
        stage = Usd.Stage.CreateNew(str(dest/'visual.usda'))
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, '/Asset')
        stage.SetDefaultPrim(root.GetPrim())
        root.GetPrim().SetCustomDataByKey('source:model_id', mid)
        geom = UsdGeom.Xform.Define(stage, '/Asset/Geometry')
        geom.GetPrim().GetReferences().AddReference('./source/instance.usd')
        xf = UsdGeom.Xformable(geom)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(offset)
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale))
        for relative, extent in corrected_extents:
            mesh = UsdGeom.Mesh.Get(stage, '/Asset/Geometry/'+relative)
            mesh.CreateExtentAttr([Gf.Vec3f(*point) for point in extent])
        disabled_joints = []
        normal_repairs = []
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
                # Raw GRScenes lids have joints without rigid bodies. Preserve
                # them in source, but suppress them in this single-body import.
                disabled_joints.append(str(prim.GetPath()))
                prim.SetActive(False)
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                normals = mesh.GetNormalsAttr().Get()
                if normals is not None and len(normals):
                    point_count = len(mesh.GetPointsAttr().Get())
                    corner_count = len(mesh.GetFaceVertexIndicesAttr().Get())
                    if mesh.GetNormalsInterpolation() == 'vertex' and len(normals) != point_count:
                        if len(normals) != corner_count:
                            raise ValueError(f'Unrecognized source normal layout: {prim.GetPath()}')
                        mesh.SetNormalsInterpolation('faceVarying')
                        normal_repairs.append(str(prim.GetPath()))
        stage.GetRootLayer().Save()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render']).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
        dimensions = ((hi_array - lo_array) * scale).tolist()
        assert abs(lo[2] * scale + offset[2]) < 1e-6
        assert min(dimensions) > 0.05 and max(dimensions) < 2.1, (asset_id,dimensions)
        scene_refs = [v for v in inventory if v['model_id'] == mid]
        assert scene_refs
        # Source metadata uses tighter vertex bounds than authored USD extents.
        height_delta = dimensions[2]-scene_refs[0]['scene_bbox_dimensions_m'][2]
        assert abs(height_delta) < 0.10, (asset_id, 'source height differs', height_delta)
        mesh_count = sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse())
        missing = []
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    value = attr.Get()
                    if value and value.path and not value.resolvedPath:
                        missing.append([str(prim.GetPath()), attr.GetName(), value.path])
        assert not missing, missing
        record = dict(id=asset_id, category=category, model_id=mid, source_usd=str(file),
                      source_sha256=digest(file), source_file_up_axis=str(UsdGeom.GetStageUpAxis(src_stage)),
                      source_scene_up_axis='Z', unit_conversion=scale, shape_rescale=1.0,
                      dimensions_m=dimensions, bottom_z_m=float(lo[2] * scale + offset[2]),
                      dimensions_method='visible mesh vertices with source transforms; corrected extent cache only',
                      disabled_source_joints=disabled_joints,
                      repaired_normals_interpolation=normal_repairs,
                      source_instances=scene_refs, mesh_count=mesh_count,
                      usd=(dest/'visual.usda').relative_to(output).as_posix(), dependencies=copied,
                      dependency_check='passed', physics_validation='not run; experiments paused',
                      mass_kg=None, collision_cooking='not run', grasp_controller_integration='not performed')
        record.update({k:v for k,v in selected.items() if k not in ('id','category','model_id')})
        record['scene_height_difference_m'] = height_delta
        manifest['assets'].append(record)
        print(json.dumps(dict(id=asset_id, dimensions_m=dimensions, meshes=mesh_count, dependencies=len(copied))), flush=True)
    write_json(output/'asset_manifest.json', manifest)
    # Concrete Isaac Sim stage importing all candidates at the same physical scale.
    stage = Usd.Stage.CreateNew(str(output/'asset_gallery.usda'))
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(root.GetPrim())
    for i, asset in enumerate(manifest['assets']):
        xform = UsdGeom.Xform.Define(stage, '/World/'+asset['id'])
        xform.GetPrim().GetReferences().AddReference('./'+asset['usd'])
        xform.AddTranslateOp().Set(Gf.Vec3d((i%5)*2.0, (i//5)*2.0, 0))
    stage.GetRootLayer().Save()
    print('COMPLETE: asset import finished without policy, evaluation, or physics stepping', flush=True)

if __name__ == '__main__':
    main()
