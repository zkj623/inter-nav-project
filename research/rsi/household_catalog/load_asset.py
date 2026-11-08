"""Import the portable, metre-scale USD assets into an existing Isaac Sim stage.

The default is visual import. To prepare a dynamic body, the caller must supply
mass and friction explicitly. For dynamic operation, call bake_collision_meshes before starting physics and
use the matching cooked-geometry robot adapter. Raw visual references alone are
not a validated manipulation setup.
This helper never starts the timeline or calls a physics step.
"""
import json
import math
from pathlib import Path


def add_asset(stage, asset_id, prim_path, position=(0, 0, 0), yaw_degrees=0,
              *, mass_kg=None, static_friction=None, dynamic_friction=None,
              collision_approximation='convexDecomposition', package_root=None):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    root = Path(package_root).resolve() if package_root else Path(__file__).resolve().parent
    manifest = json.loads((root/'asset_manifest.json').read_text(encoding='utf-8'))
    records = {a['id']: a for a in manifest['assets']}
    if asset_id not in records:
        raise KeyError(asset_id)
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError('Target stage must use Z-up and metres.')
    path = Sdf.Path(prim_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError('An absolute prim path below the stage root is required.')
    if stage.GetPrimAtPath(path):
        raise ValueError(f'Refusing to overwrite existing prim: {path}')
    if len(position) != 3 or not all(math.isfinite(float(x)) for x in (*position, yaw_degrees)):
        raise ValueError('Pose must be finite.')
    if mass_kg is not None:
        if not math.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('Mass must be positive and explicitly supplied.')
        if any(x is None or not math.isfinite(x) or x < 0 for x in (static_friction, dynamic_friction)):
            raise ValueError('Static and dynamic friction must be explicitly supplied.')
        if collision_approximation not in ('convexDecomposition', 'sdf'):
            raise ValueError('Use a concavity-aware collider; whole-object convexHull fills cavities.')
    elif static_friction is not None or dynamic_friction is not None:
        raise ValueError('Friction requires an explicit mass.')
    xf = UsdGeom.Xform.Define(stage, path)
    prim = xf.GetPrim()
    prim.GetReferences().AddReference(str(root/records[asset_id]['usd']))
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, position)))
    xf.AddRotateZOp().Set(float(yaw_degrees))
    if mass_kg is not None:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg))
        material = UsdShade.Material.Define(stage, path.AppendChild('ContactMaterial'))
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(static_friction))
        physics.CreateDynamicFrictionAttr(float(dynamic_friction))
        physics.CreateRestitutionAttr(0.)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(collision_approximation)
                UsdShade.MaterialBindingAPI.Apply(child).Bind(material, materialPurpose='physics')
        prim.SetCustomDataByKey('validation:physics', 'pending; collision cooking and cavity checks required')
        # All original parts move as one body. Lid articulation is not configured.
        prim.SetCustomDataByKey('interaction:mode', 'single rigid body; no lid articulation')
    return prim
