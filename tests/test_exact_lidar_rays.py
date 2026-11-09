import unittest
from dataclasses import replace
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig,OccupancyGridMap,world_ray_cells,_bresenham


class ExactLidarRayTests(unittest.TestCase):
    def config(self):
        return MappingConfig(x_limits=(0,2),y_limits=(0,2),grid_resolution=.25,
            obstacle_height=(.2,.5),allow_lidar_free_only_rays=False,ray_traversal='exact')
    def test_world_ray_clears_a_crossed_cell_missed_by_center_snapping(self):
        cfg=self.config();origin=(.01,.20,.3);end=(1.4,.60,.3)
        ray=world_ray_cells(origin,end,cfg)
        self.assertIn((1,0),ray)
        self.assertNotIn((1,0),_bresenham((0,0),(2,5)))
        grid=OccupancyGridMap(cfg);grid.update_lidar(origin,np.array([end]),5.)
        self.assertTrue(grid.observed[1,0]);self.assertFalse(grid.occupied_mask()[1,0])
    def test_measured_endpoint_stays_occupied_and_behind_it_unknown(self):
        grid=OccupancyGridMap(self.config());grid.update_lidar((.01,.20,.3),np.array([[1.4,.60,.3]]),5.)
        self.assertTrue(grid.occupied_mask()[2,5]);self.assertFalse(grid.observed[2,6])
    def test_exact_corner_does_not_clear_adjacent_touched_cells(self):
        ray=world_ray_cells((.125,.125,.3),(1.125,1.125,.3),self.config())
        self.assertEqual(ray,[(0,0),(1,1),(2,2),(3,3),(4,4)])
        self.assertEqual(world_ray_cells((1.125,1.125,.3),(.125,.125,.3),self.config()),list(reversed(ray)))
    def test_endpoint_roundoff_never_crosses_past_the_measured_cell(self):
        cfg=MappingConfig(x_limits=(-10.,10.),y_limits=(-10.,10.),grid_resolution=.1)
        grid=OccupancyGridMap(cfg)
        # (0.2 + 10) / 0.1 is 101.99999999999999, while the
        # corner-crossing times still compare equal at 1e-10 tolerance.
        for endpoint in ((1.,.2,.5),(.2,1.,.5),(-1.,-.2,.5),(-.2,-1.,.5)):
            for start,end in (((0.,0.,.4),endpoint),(endpoint,(0.,0.,.4))):
                with self.subTest(start=start,end=end):
                    ray=world_ray_cells(start,end,cfg)
                    first,last=grid.world_to_cell(start[:2]),grid.world_to_cell(end[:2])
                    self.assertEqual(ray[0],first)
                    self.assertEqual(ray[-1],last)
                    self.assertTrue(all(min(first[0],last[0])<=r<=max(first[0],last[0])
                        and min(first[1],last[1])<=c<=max(first[1],last[1]) for r,c in ray))
    def test_ray_model_changes_free_evidence_not_obstacle_hits(self):
        cfg=self.config();points=np.array([[1.4,.6,.3],[1.3,.7,.3],[1.8,1.5,.3],[.4,1.8,.3]])
        exact=OccupancyGridMap(cfg);old=OccupancyGridMap(replace(cfg,ray_traversal='bresenham'))
        for grid in [exact,old]:grid.update_lidar((.01,.2,.3),points,5.)
        np.testing.assert_array_equal(exact.occupied_mask(),old.occupied_mask())


if __name__=='__main__':unittest.main()
