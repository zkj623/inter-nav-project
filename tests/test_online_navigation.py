import unittest
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig,OccupancyGridMap
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import ExplorationVoronoiRoutes,route_blocked


class OnlineMapTests(unittest.TestCase):
    def make(self):
        cfg=MappingConfig(x_limits=(-3,3),y_limits=(-3,3),grid_resolution=.05,
            obstacle_height=(.05,.8),robot_radius=.25,obstacle_inflation_radius=.30)
        shape=OccupancyGridMap(cfg).observed.shape
        p=SidePushPlanner(np.zeros(shape,bool),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_route_radius=.30;p.navigation_half_width=.22;p.full_navigation_footprint=True
        return p,OnlineNavigationMap(p,cfg)
    def obs(self,step=0,points=()):
        return {'position':[0,0,.41],'orientation':[1,0,0,0],
            'sensors':{'lidar':{'position':[0,0,.5],'pointcloud':np.asarray(points,float),'physics_step':step,'max_range':5.}}}
    def test_initial_map_is_unknown_and_hidden_proposals_are_not_admitted(self):
        p,m=self.make();self.assertFalse(m.observed.any());self.assertFalse(m.global_occupied.any())
        hidden=BoxState('hidden',(2,0),(.5,.5,.5))
        m.update(0,self.obs(),[hidden],force=True)
        self.assertEqual(m.current_boxes,[])
        self.assertFalse(m.observed[p.cell((2,0))])
        with m.global_planning() as boxes:self.assertFalse(p.path((0,0),(2,0),boxes))
    def test_visibility_is_required_even_for_forced_global_refresh(self):
        p,m=self.make();a=BoxState('visible',(1,0),(.4,.4,.5));b=BoxState('behind',(2,0),(.4,.4,.5))
        m.update(0,self.obs(points=[(.8,-.05,.3),(.8,0,.3),(.8,.05,.3)]),[a,b],force=True)
        self.assertEqual([x.id for x in m.current_boxes],['visible'])
        m.refresh_global([a,b]);self.assertEqual([x.id for x in m.global_boxes],['visible'])
    def test_vacated_footprint_is_unknown_until_free_rays_observe_it(self):
        p,m=self.make();a=BoxState('A',(1,0),(.4,.4,.5))
        m.update(0,self.obs(points=[(.8,-.05,.3),(.8,0,.3),(.8,.05,.3)]),[a],force=True)
        moved=BoxState('A',(2,0),(.4,.4,.5))
        m.update(.1,self.obs(1,[(1.8,-.02,.3),(1.8,0,.3),(1.8,.02,.3)]),[moved])
        self.assertTrue(m.observed[p.cell((1,0))])
        self.assertFalse(m.local_occupied[p.cell((1,0))])
        self.assertFalse(m.observed[p.cell((1,.15))])
        self.assertTrue(m.global_occupied[p.cell((1,0))])
        m.refresh_global(reason='passage_cleared')
        self.assertFalse(m.global_occupied[p.cell((1,0))])
    def test_free_evidence_does_not_reveal_cells_outside_sensor_rays(self):
        p,m=self.make();hit=np.zeros_like(m.observed);free=hit.copy()
        free[p.cell((0,0))]=True;hit[p.cell((1,0))]=True
        m.ingest_scan_masks(hit,free);m.refresh_global()
        self.assertTrue(m.global_occupied[p.cell((1,0))])
        self.assertFalse(m.global_observed[p.cell((1,1))])
        self.assertEqual(m.statistics()['initial_observed_cells'],0)
    def test_frontier_is_reachable_and_stays_inside_observed_region(self):
        p,m=self.make()
        yy,xx=np.indices(m.observed.shape);free=(xx>35)&(xx<85)&(yy>40)&(yy<80)
        m.ingest_scan_masks(np.zeros_like(free),free);m.refresh_global();m._latest_static=m.evidence.occupied_mask();m._bind_local()
        d,target=OnlineGoalSelector().decide(m,(0,0),(2.5,0))
        self.assertEqual(d.kind,'explore');self.assertIsNotNone(target)
        self.assertTrue(m.observed[p.cell(target)])
        with m.global_planning() as boxes:self.assertTrue(p.path((0,0),target,boxes))
        self.assertFalse(m.observed[p.cell((2.5,0))])
    def test_goal_is_used_only_when_connected_in_observed_map(self):
        p,m=self.make();m.ingest_scan_masks(np.zeros_like(m.observed),np.ones_like(m.observed));m.refresh_global();m._bind_local()
        d,target=OnlineGoalSelector().decide(m,(0,0),(1,0))
        self.assertEqual(d.kind,'navigate');self.assertEqual(target,(1,0))

    def test_unknown_centers_are_blocked_but_do_not_inflate_like_obstacles(self):
        p,m=self.make();m.ingest_scan_masks(np.zeros_like(m.observed),np.ones_like(m.observed))
        hole=p.cell((1,.10));m.observed[hole]=False;m.refresh_global();m._bind_local()
        with m.global_planning() as boxes:
            blocked=p.blocked(boxes)
            self.assertTrue(blocked[hole]);self.assertFalse(blocked[p.cell((1,0))])
            np.testing.assert_array_equal(route_blocked(p,boxes),blocked)

    def test_selected_frontier_is_reachable_by_actual_voronoi_execution_planner(self):
        p,m=self.make();yy,xx=np.indices(m.observed.shape)
        known=(xx>20)&(xx<95)&(yy>35)&(yy<85)
        m.ingest_scan_masks(np.zeros_like(known),known);m.refresh_global();m._bind_local()
        d,target=OnlineGoalSelector().decide(m,(0,0),(2.7,0))
        self.assertEqual(d.kind,'explore')
        with m.global_planning() as boxes:
            path=ExplorationVoronoiRoutes(p).plan((0,0),target,boxes)
            self.assertTrue(path)
            self.assertTrue(all(m.global_observed[p.cell(xy)] for xy in path))

    def test_observed_islands_cannot_plan_across_unknown_band(self):
        p,m=self.make();free=np.ones_like(m.observed);free[:,58:64]=False
        m.ingest_scan_masks(np.zeros_like(free),free);m.refresh_global()
        with m.global_planning() as boxes:self.assertFalse(p.path((-1,0),(1,0),boxes))


if __name__=='__main__':unittest.main()
