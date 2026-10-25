"""Restore authored carton appearance and friction after visual simplification."""


def restore_carton_materials(stage, scene_root, box_ids):
    from pxr import Usd, UsdPhysics, UsdShade

    audit=[]
    for box_id in box_ids:
        material_prim=stage.GetPrimAtPath(f'{scene_root}/BoxMaterials/{box_id}_Cardboard')
        if not material_prim:raise ValueError(f'Missing authored material for {box_id}')
        material_prim.SetActive(True)
        material=UsdShade.Material(material_prim)
        physics=UsdPhysics.MaterialAPI(material_prim)
        root=stage.GetPrimAtPath(f'{scene_root}/InteractiveBoxes/{box_id}')
        count=0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            binding=UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            binding.Bind(material,materialPurpose='physics',bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            count+=1
        if not count:raise ValueError(f'Missing collider for {box_id}')
        audit.append({'id':box_id,'colliders':count,'material':str(material.GetPath()),
                      'static_friction':physics.GetStaticFrictionAttr().Get(),
                      'dynamic_friction':physics.GetDynamicFrictionAttr().Get(),
                      'restitution':physics.GetRestitutionAttr().Get()})
    return audit
