import math,unittest
from types import SimpleNamespace
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig,OccupancyGridMap
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import ClearanceWeightedVoronoiPlanner,ExplorationVoronoiRoutes


class ClearanceRouteTests(unittest.TestCase):
    def test_wide_detour_beats_short_one_cell_corridor(self):
        blocked=np.zeros((21,21),bool);blocked[[0,-1]]=True;blocked[:,[0,-1]]=True
        blocked[4,4:17]=True;blocked[6,4:17]=True
        skeleton={(5,c) for c in range(2,19)}|{(12,c) for c in range(2,19)}
        skeleton|={(r,c) for r in range(5,13) for c in [2,18]}
        occupancy=OccupancyGridMap(MappingConfig(x_limits=(0,1),y_limits=(0,1),grid_resolution=.05))
        graph=SimpleNamespace(occupancy=occupancy)
        shortest=ClearanceWeightedVoronoiPlanner(graph,clearance_weight=0)._skeleton_path((5,2),(5,18),skeleton,blocked)
        safer=ClearanceWeightedVoronoiPlanner(graph,clearance_weight=.30)._skeleton_path((5,2),(5,18),skeleton,blocked)
        self.assertEqual(max(r for r,c in shortest),5)
        self.assertEqual(max(r for r,c in safer),12)

    def test_fallback_cannot_shrink_the_robot_to_invent_a_route(self):
        free=np.zeros((201,201),bool);free[88:113]=True
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]],robot_radius=.10)
        self.assertTrue(p.path((-1,0),(1,0),[]))
        p.navigation_pose_heading=math.pi/2
        routes=ExplorationVoronoiRoutes(p)
        self.assertFalse(routes.plan((-1,0),(1,0),[]))
        self.assertTrue(routes.last_plan['same_roadmap_occupancy'])


if __name__=='__main__':unittest.main()
