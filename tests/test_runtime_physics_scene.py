import pytest
pytest.importorskip('pxr.Usd')
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def test_imported_physics_scene_is_disabled_without_changing_geometry_or_asset():
    from grutopia_extension.interactive_navigation.scene_paths import use_runtime_physics_scene

    source = Usd.Stage.CreateInMemory()
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, '/Scene').GetPrim())
    UsdPhysics.Scene.Define(source, '/Scene/physicsScene')
    cube = UsdGeom.Cube.Define(source, '/Scene/Wall')
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateSimulationOwnerRel().SetTargets(['/Scene/physicsScene'])
    original = source.GetRootLayer().ExportToString()

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    stage.DefinePrim('/World/env_0/scene').GetReferences().AddReference(source.GetRootLayer().identifier)
    disabled = use_runtime_physics_scene(stage, '/World/env_0/scene', '/physicsScene')
    assert disabled == ['/World/env_0/scene/physicsScene']
    assert stage.GetPrimAtPath('/physicsScene').IsActive()
    assert not stage.GetPrimAtPath(disabled[0]).IsActive()
    wall = stage.GetPrimAtPath('/World/env_0/scene/Wall')
    assert wall.IsActive() and wall.HasAPI(UsdPhysics.CollisionAPI)
    assert wall.GetRelationship('physics:simulationOwner').GetTargets() == [Sdf.Path('/physicsScene')]
    assert source.GetRootLayer().ExportToString() == original
    assert use_runtime_physics_scene(stage, '/World/env_0/scene', '/physicsScene') == []
