"""Share identical single-shader MDL materials without simplifying appearance."""
def compact_scene_materials(stage, root_path, prune_unused=False):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f'Scene root not found: {root_path}')
    prims = list(Usd.PrimRange(root))
    materials = {p.GetPath(): p for p in prims if p.IsA(UsdShade.Material)}
    # Older assets author valid binding relationships without their API schema.
    # Apply the schema before computing bindings, without changing relationships.
    for prim in prims:
        if any(r.GetName().startswith('material:binding') for r in prim.GetRelationships()):
            UsdShade.MaterialBindingAPI.Apply(prim)

    def material_parent(prim):
        while prim and not prim.IsPseudoRoot():
            if prim.IsA(UsdShade.Material): return prim.GetPath()
            prim = prim.GetParent()
        return None

    external = set()
    for prim in prims:
        owner = material_parent(prim)
        for attr in prim.GetAttributes():
            for connection in attr.GetConnections():
                target_owner = material_parent(stage.GetPrimAtPath(connection.GetPrimPath()))
                if target_owner is not None and target_owner != owner:
                    external.add(target_owner)

    def key(material):
        if material.GetPath() in external or material.HasAPI(UsdPhysics.MaterialAPI): return None
        children = material.GetChildren()
        if len(children) != 1 or not children[0].IsA(UsdShade.Shader): return None
        shader = UsdShade.Shader(children[0])
        if shader.GetImplementationSourceAttr().Get() != 'sourceAsset': return None
        if UsdShade.Material(material).GetInputs(): return None
        attributes = []
        for attr in children[0].GetAttributes():
            if attr.GetConnections(): return None
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                # Relative paths with different resolved files must not merge.
                if value.path and not value.resolvedPath: return None
                value = value.resolvedPath
            attributes.append((attr.GetName(), str(attr.GetTypeName()), repr(value), repr(attr.GetAllMetadata())))
        outputs = []
        for output in UsdShade.Material(material).GetOutputs():
            connections = output.GetAttr().GetConnections()
            if any(c.GetPrimPath() != children[0].GetPath() for c in connections): return None
            outputs.append((output.GetFullName(), str(output.GetTypeName()),
                tuple(c.name for c in connections), repr(output.GetAttr().Get())))
        return repr((sorted(attributes), sorted(outputs)))

    canonical = {}; replacement = {}
    for path, material in materials.items():
        fingerprint = key(material)
        if fingerprint is None: continue
        if fingerprint in canonical: replacement[path] = canonical[fingerprint]
        else: canonical[fingerprint] = path

    before = []
    for prim in prims:
        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset)): continue
        for purpose in ['', 'preview', 'full', 'physics']:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose=purpose)
            if material: before.append((prim.GetPath(), purpose, material.GetPath(), material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)))
    rewrites = 0
    for prim in prims:
        for relationship in prim.GetRelationships():
            if not relationship.GetName().startswith('material:binding'): continue
            targets = relationship.GetTargets()
            updated = [replacement.get(target, target) for target in targets]
            if targets != updated:
                relationship.SetTargets(updated); rewrites += 1
    for path in replacement:
        materials[path].SetActive(False)
    unused = []
    if prune_unused:
        keep = {replacement.get(path,path) for _,_,path,_ in before} | external
        keep |= {path for path,prim in materials.items() if prim.HasAPI(UsdPhysics.MaterialAPI)}
        # A retained nested material also keeps its material ancestors active.
        keep |= {parent for path in list(keep) for parent in materials if path.HasPrefix(parent)}
        for path,material in materials.items():
            if path in replacement or path in keep: continue
            # Never deactivate geometry or contact data placed under a material.
            descendants=list(Usd.PrimRange(material))[1:]
            if any(not (p.IsA(UsdShade.Shader) or p.IsA(UsdShade.NodeGraph)) for p in descendants): continue
            material.SetActive(False);unused.append(path)
    physics = 0
    for path, purpose, previous, physical in before:
        current, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(materialPurpose=purpose)
        expected = replacement.get(previous, previous)
        if not current or current.GetPath() != expected:
            raise RuntimeError(f'Material compaction changed effective binding at {path} ({purpose})')
        if physical:
            if expected != previous: raise RuntimeError('Physical material was merged')
            physics += 1
    return {'materials_before': len(materials), 'duplicate_materials_disabled': len(replacement),
        'unused_materials_disabled':len(unused),
        'canonical_single_shader_materials': len(canonical), 'binding_relationships_rewritten': rewrites,
        'effective_bindings_verified': len(before), 'physical_bindings_preserved': physics,
        'texture_resampling': False, 'geometry_modified': False}
