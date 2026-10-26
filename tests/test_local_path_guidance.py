import unittest
import numpy as np
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear,ForwardExplorationPathTracker
from grutopia_extension.interactive_navigation.local_path_guidance import local_guidance_path


class LocalGuidanceTests(unittest.TestCase):
    def test_rejoin_can_start_outside_global_halo_when_oriented_body_is_free(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        p=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        p.navigation_route_radius=.30
        path,detail=local_guidance_path(p,(-.52,0),[(.2,0,0),(1.,0,0)],[],np.pi/2)
        self.assertTrue(path);self.assertEqual(path[-1],(1.,0.,0.))
        self.assertEqual(detail['engine'],'local_grid_connector')

    def test_centre_clear_recovery_must_not_drag_front_paw_through_corner(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        obstacle=p.box_mask(BoxState('fixed',(.8,.1),(.1,.1,.5)))
        p.free &= ~obstacle
        self.assertTrue(AStarMapPlanner.segment_clear(p.cell((0,0)),p.cell((.6,.6)),~p.free))
        self.assertFalse(body_translation_recovery_clear(p,[],(0,0),0.,(.6,.6)))

    def test_existing_mild_overlap_can_decrease_until_clear(self):
        free=np.ones((201,201),bool);free[:,118:]=False
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        self.assertTrue(body_translation_recovery_clear(p,[],(0,0),0.,(-.3,0)))
        self.assertFalse(body_translation_recovery_clear(p,[],(0,0),0.,(.3,0)))

    def test_recovery_can_turn_out_when_no_translation_pose_fits(self):
        free=np.zeros((201,201),bool);free[90:111]=True
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        p.layered_local_control=True;p.full_navigation_footprint=True
        tracker=ForwardExplorationPathTracker((0,0),[(1,0)],guard_motion=False)
        action,state=tracker.action((0,0),np.pi/4,p,[])
        self.assertEqual(state['controller'],'body_safe_recovery_turn')
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertLess(action['move_by_speed'][2],0.)


if __name__=='__main__':unittest.main()
