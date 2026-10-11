"""Scene-directory helpers shared by navigation runners."""

from pathlib import Path


def configure_scene_mdl_paths(scene_asset_path):
    """Register a GRScenes Materials folder so Isaac can resolve MDL assets."""

    if not scene_asset_path or '://' in scene_asset_path:
        return
    scene_directory = Path(scene_asset_path).resolve().parent
    material_directory = scene_directory / 'Materials'
    if not material_directory.exists():
        return
    import carb

    setting = '/app/mdl/additionalUserPaths'
    settings = carb.settings.get_settings()
    paths = list(settings.get(setting) or [])
    material_path = str(material_directory.resolve())
    if material_path not in paths:
        paths.append(material_path)
        settings.set_string_array(setting, paths)


def use_runtime_physics_scene(stage, scene_root, runtime_scene_path):
    """Keep imported geometry in the runtime's single physics world.

    GRScenes can reference a PhysicsScene of its own. Isaac's global physics
    timer receives callbacks from both worlds, so its time no longer matches
    the robot's physical integration time. Deactivate only imported scenes,
    before reset creates PhysX actors; leave the referenced asset untouched.
    """
    from pxr import Sdf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(scene_root)
    if not root.IsValid():
        return []
    runtime_path = Sdf.Path(runtime_scene_path)
    if not stage.GetPrimAtPath(runtime_path).IsA(UsdPhysics.Scene):
        raise ValueError('Runtime physics scene must exist before importing geometry')
    embedded = [
        prim for prim in Usd.PrimRange(root)
        if prim.IsA(UsdPhysics.Scene) and prim.GetPath() != runtime_path
    ]
    paths = {prim.GetPath() for prim in embedded}
    # Most assets use the default world. Preserve explicitly assigned actors
    # too, instead of leaving their owner pointing at a deactivated scene.
    for prim in Usd.PrimRange(root):
        owner = prim.GetRelationship('physics:simulationOwner')
        if owner and paths.intersection(owner.GetTargets()):
            targets = [runtime_path if target in paths else target for target in owner.GetTargets()]
            owner.SetTargets(list(dict.fromkeys(targets)))
    for prim in embedded:
        prim.SetActive(False)
    return sorted(str(path) for path in paths)


def prepare_navigation_scene(stage, scene_root, material_mode):
    """Prepare navigation assets in the existing task setup phase, before reset."""
    import json

    from omni.isaac.core.simulation_context import SimulationContext
    from pxr import UsdPhysics
    from .material_fallback import apply_koostruct_material_fallbacks, apply_simple_scene_material

    physics = SimulationContext.instance().get_physics_context()
    disabled = use_runtime_physics_scene(stage, scene_root, physics.prim_path)
    active = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    print(json.dumps({'event': 'scene_physics_setup', 'runtime_scene': physics.prim_path,
                      'disabled_embedded_scenes': disabled, 'active_physics_scenes': active,
                      'physics_dt': physics.get_physics_dt()}), flush=True)
    if active != [str(physics.prim_path)]:
        raise RuntimeError(f'Go2 navigation requires one runtime physics scene; found {active}')
    if material_mode not in ('original', 'preview', 'simple'):
        raise ValueError(f'Unknown navigation material mode: {material_mode}')
    if material_mode != 'original':
        convert = apply_simple_scene_material if material_mode == 'simple' else apply_koostruct_material_fallbacks
        print(json.dumps({'event': 'material_fallback', 'mode': material_mode,
                          **convert(stage, scene_root)}), flush=True)
