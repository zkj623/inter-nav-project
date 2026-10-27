import unittest
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig,OccupancyGridMap
from grutopia_extension.interactive_navigation.online_lidar_evidence import online_lidar_masks


class OnlineRayEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=MappingConfig(x_limits=(-1,7),y_limits=(-2,2),grid_resolution=.05,
            obstacle_height=(.06,.51),ray_traversal='exact')
        self.grid=OccupancyGridMap(self.cfg)
    def masks(self,points,origin=(0,0,.3)):
        return online_lidar_masks(self.cfg,origin,np.array(points),5.)
    def test_no_hit_range_ray_clears_only_the_measured_range(self):
        hit,free,report=self.masks([[5.25,0,.3]])
        self.assertFalse(hit.any());self.assertTrue(free[self.grid.world_to_cell((3,0))])
        self.assertFalse(free[self.grid.world_to_cell((5.05,0))]);self.assertEqual(report['range_rays'],1)
    def test_nearer_obstacle_clips_other_vertical_channels(self):
        hit,free,_=self.masks([[1,0,.3],[5.25,0,.35]])
        self.assertTrue(hit[self.grid.world_to_cell((1,0))])
        self.assertFalse(free[self.grid.world_to_cell((1,0))])
        self.assertFalse(free[self.grid.world_to_cell((2,0))])
    def test_floor_return_is_not_an_obstacle_and_clips_at_height_boundary(self):
        hit,free,_=self.masks([[2,0,0]],origin=(0,0,.5))
        self.assertFalse(hit.any());self.assertTrue(free[self.grid.world_to_cell((1,0))])
        self.assertFalse(free[self.grid.world_to_cell((1.9,0))])
    def test_overhead_ray_does_not_clear_the_floor(self):
        hit,free,_=self.masks([[3,0,2]],origin=(0,0,.7))
        self.assertFalse(hit.any());self.assertFalse(free.any())
    def test_no_reveal_behind_hit_or_outside_beam(self):
        hit,free,_=self.masks([[1.1,0,.3]])
        self.assertTrue(hit.any());self.assertFalse(free[self.grid.world_to_cell((1.2,0))])
        self.assertFalse(free[self.grid.world_to_cell((.5,.2))])
    def test_invalid_no_hit_sentinel_cannot_reveal_the_map(self):
        hit,free,_=self.masks([[1e10,0,.3],[float('nan'),0,.3],[0,0,.3]])
        self.assertFalse(hit.any());self.assertFalse(free.any())
    def test_range_endpoints_do_not_create_phantom_pointcloud_surfaces(self):
        from grutopia_extension.interactive_navigation.mapping import FusedMap
        fused=FusedMap(self.cfg)
        fused.update_lidar((0,0,.3),np.array([[1,0,.3],[5.25,0,.3]]),step=0,max_range=5.)
        self.assertEqual(fused.voxels.voxel_count,1)
        self.assertEqual(fused.lidar_points,1)

if __name__=='__main__':unittest.main()
