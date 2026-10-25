import unittest
from types import SimpleNamespace
import numpy as np
from grutopia.demo.go2_semantic_exploration import build_grscene_floor

class FloorConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.profile=SimpleNamespace(mapping=SimpleNamespace(x_limits=[.5,18.3],y_limits=[-6.2,6.2]))

    def test_one_cm_floor_has_requested_physical_top(self):
        floor=build_grscene_floor(self.profile,floor_top=.01)[0]
        self.assertAlmostEqual(floor.position[2]+floor.scale[2]/2,.01)
        self.assertAlmostEqual(floor.position[2]-floor.scale[2]/2,0.)
        self.assertFalse(floor.visible)
        np.testing.assert_allclose(floor.scale[:2],[17.8,12.4])

    def test_legacy_default_retains_fifteen_cm_reproduction(self):
        floor=build_grscene_floor(self.profile)[0]
        self.assertAlmostEqual(floor.position[2]+floor.scale[2]/2,.15)

    def test_invalid_floor_does_not_create_inverted_collision_geometry(self):
        for value in [0.,-.01,float('nan'),float('inf'),.31]:
            with self.subTest(value=value),self.assertRaises(ValueError):build_grscene_floor(self.profile,floor_top=value)

if __name__=='__main__':unittest.main()
