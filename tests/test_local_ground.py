from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from grutopia_extension.manipulation.local_ground import add_local_ground


class LocalGroundTests(unittest.TestCase):
    def test_ground_preserves_unbound_infinite_plane_without_remote_assets(self):
        stage, ground_factory = Mock(), Mock()
        stage.DefinePrim.return_value.GetPath.return_value = '/World/defaultGroundPlane/GroundPlane/CollisionPlane'
        usd_geom = SimpleNamespace(Xform=Mock(), Xformable=Mock(), Mesh=Mock())
        physics = SimpleNamespace(CollisionAPI=Mock())
        gf = SimpleNamespace(Vec3f=Mock(side_effect=lambda *values: values))
        sdf = SimpleNamespace(ValueTypeNames=SimpleNamespace(Token='token'))
        scene = SimpleNamespace(add=Mock())
        with patch.dict('sys.modules', {'pxr': SimpleNamespace(Gf=gf, Sdf=sdf, UsdGeom=usd_geom, UsdPhysics=physics),
                                       'omni.isaac.core.prims': SimpleNamespace(GeometryPrim=ground_factory),
                                       'omni.isaac.core.utils.stage': SimpleNamespace(get_current_stage=lambda: stage)}):
            add_local_ground(scene)
        stage.DefinePrim.assert_called_once_with('/World/defaultGroundPlane/GroundPlane/CollisionPlane', 'Plane')
        usd_geom.Xformable.return_value.AddScaleOp.return_value.Set.assert_called_once_with((0.01, 0.01, 0.01))
        physics.CollisionAPI.Apply.assert_called_once_with(stage.DefinePrim.return_value)
        self.assertTrue(ground_factory.call_args.kwargs['collision'])
        self.assertNotIn('physics_material', ground_factory.call_args.kwargs)
        scene.add.assert_called_once_with(ground_factory.return_value)


if __name__ == '__main__':
    unittest.main()
