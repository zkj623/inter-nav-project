"""Restore source visual materials without importing source collision geometry."""
import json
from pathlib import Path

def restore_original_visuals(stage, source_scene, robot_asset, audit_path):
    from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf
    from grutopia_extension.interactive_navigation.material_compaction import compact_scene_materials
    from grutopia_extension.interactive_navigation.material_fallback import apply_koostruct_material_fallbacks
    from g2_preview_materials import convert_exported_pbr_materials
    source = Usd.Stage.Open(str(source_scene))
    source.DefinePrim('/robot', 'Xform').GetReferences().AddReference(str(robot_asset), '/robot')
    while True:
        instances = [p for p in source.Traverse() if p.IsInstance()]
        if not instances:
            break
        for p in instances:
            p.SetInstanceable(False)
    compact = compact_scene_materials(source, '/World/Environment', prune_unused=True)
    pbr = convert_exported_pbr_materials(source, '/World')
    fallback = apply_koostruct_material_fallbacks(source)
    flat = source.Flatten()
    namespace = Sdf.Path('/World/SourceVisualMaterials')
    UsdGeom.Scope.Define(stage, namespace)
    copied = {}
    texture_files = set()
    before = {}
    def physics_signature(p):
        m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics')
        if not m or not m.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            return None
        return {a.GetName(): str(a.Get()) for a in m.GetPrim().GetAttributes()
                if a.GetName().startswith(('physics:', 'physxMaterial:'))}
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    for p in colliders:
        before[str(p.GetPath())] = physics_signature(p)
    matched = uv_count = 0
    bindings = []
    missing = []
    for p in list(stage.Traverse()):
        if not (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Subset)):
            continue
        original = source.GetPrimAtPath(p.GetPath())
        if not original:
            continue
        material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial()
        if not material:
            missing.append(str(p.GetPath()))
            continue
        oldpath = material.GetPath()
        newpath = oldpath.ReplacePrefix(Sdf.Path.absoluteRootPath, namespace)
        if oldpath not in copied:
            # Define namespace parents only. No source mesh or physics properties copied.
            for parent in reversed(newpath.GetParentPath().GetPrefixes()):
                if parent != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(parent):
                    stage.DefinePrim(parent, 'Scope')
            Sdf.CopySpec(flat, oldpath, stage.GetEditTarget().GetLayer(), newpath)
            copied[oldpath] = newpath
        api = UsdShade.MaterialBindingAPI.Apply(p)
        physical, _ = api.ComputeBoundMaterial(materialPurpose='physics')
        if physical and physical.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            api.Bind(physical, materialPurpose='physics')
        api.Bind(UsdShade.Material(stage.GetPrimAtPath(newpath)))
        bindings.append((str(p.GetPath()), newpath))
        if p.IsA(UsdGeom.Mesh):
            UsdGeom.Mesh(p).CreateDoubleSidedAttr(True)
            for a in original.GetAttributes():
                if a.GetName().startswith('primvars:') and ('st' in a.GetName() or 'uv' in a.GetName().lower()):
                    target = p.CreateAttribute(a.GetName(), a.GetTypeName(), custom=a.IsCustom())
                    value = a.Get()
                    if value is not None:
                        target.Set(value)
                    for key, value in a.GetAllMetadata().items():
                        if key not in ('typeName', 'custom', 'variability'):
                            target.SetMetadata(key, value)
                    uv_count += 1
        matched += 1
    for p in Usd.PrimRange(stage.GetPrimAtPath(namespace)):
        for a in p.GetAttributes():
            connections = a.GetConnections()
            if connections:
                a.SetConnections([v if v.HasPrefix(namespace) else v.ReplacePrefix(Sdf.Path.absoluteRootPath, namespace) for v in connections])
            value = a.Get()
            if isinstance(value, Sdf.AssetPath):
                resolved = value.resolvedPath or value.path
                if resolved:
                    a.Set(Sdf.AssetPath(resolved))
                    if a.GetName() == 'inputs:file':
                        texture_files.add(resolved)
        for rel in p.GetRelationships():
            targets = rel.GetTargets()
            if targets:
                rel.SetTargets([v if v.HasPrefix(namespace) else v.ReplacePrefix(Sdf.Path.absoluteRootPath, namespace) for v in targets])
    invalid_connections = [(str(a.GetPath()), str(v)) for p in Usd.PrimRange(stage.GetPrimAtPath(namespace)) for a in p.GetAttributes() for v in a.GetConnections() if not stage.GetPropertyAtPath(v)]
    incorrect_bindings = [path for path, expected in bindings
                          if UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()[0].GetPath() != expected]
    changed = [str(p.GetPath()) for p in colliders if before[str(p.GetPath())] != physics_signature(p)]
    report = dict(materials_copied=len(copied), meshes_and_subsets_bound=matched,
                  uv_attributes_restored=uv_count, texture_files=len(texture_files),
                  missing_texture_files=[v for v in sorted(texture_files) if not Path(v).is_file()],
                  source_without_material=missing, compact=compact, pbr=pbr, fallback=fallback,
                  colliders_checked=len(colliders), changed_physics_materials=changed,
                  geometry_imported=False, invalid_shader_connections=invalid_connections,
                  incorrect_visual_bindings=incorrect_bindings)
    Path(audit_path).write_text(json.dumps(report, indent=2))
    print('RESTORED_VISUALS', json.dumps({k:v for k,v in report.items() if k != 'source_without_material'}), flush=True)
    assert not invalid_connections, 'Restored shader network contains unresolved connections'
    assert not incorrect_bindings, 'Source visual material bindings did not take effect'
    assert not changed, 'Visual material restoration changed effective physics parameters'
    assert not report['missing_texture_files'], 'Original texture files missing'
    return report
