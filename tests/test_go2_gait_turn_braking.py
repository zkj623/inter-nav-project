import math
import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import ForwardExplorationPathTracker


class GaitTurnBrakingTests(unittest.TestCase):
    def setup_tracker(self,path=((1.,1.,0.),(2.,2.,0.))):
        planner=SidePushPlanner(np.ones((401,401),bool),[[-5.,5.],[-5.,5.]])
        planner.layered_local_control=True
        planner.full_navigation_footprint=True
        planner.navigation_half_width=.22
        planner.navigation_route_radius=.30
        planner.navigation_pose_heading=None
        return ForwardExplorationPathTracker((0.,0.),path,guard_motion=False),planner

    def test_clear_space_sharp_turn_brakes_before_yaw_and_translation(self):
        tracker,planner=self.setup_tracker()
        action,state=tracker.action((0.,0.),0.,planner,[])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])
        self.assertEqual(state['controller'],'gait_turn_braking')
        self.assertTrue(state['local_turn_maneuver'])
        self.assertFalse(state['needs_replan'])

    def test_straight_tracking_preserves_original_command(self):
        tracker,planner=self.setup_tracker(((2.,0.,0.),))
        action,state=tracker.action((0.,0.),0.,planner,[])
        self.assertAlmostEqual(action['move_by_speed'][0],.4)
        self.assertEqual(action['move_by_speed'][1:],[0.,0.])
        self.assertEqual(state['controller'],'pure_exploration_velocity_action_for')

    def test_braking_waits_for_actual_pose_to_stop_drifting(self):
        tracker,planner=self.setup_tracker()
        tracker.action((0.,0.),0.,planner,[])
        for tick in range(1,111):
            action,state=tracker.action((.003*tick,0.),0.,planner,[])
            self.assertEqual(action['move_by_speed'],[0.,0.,0.])
            self.assertFalse(state['needs_replan'])
        for _ in range(25):
            action,state=tracker.action((.33,0.),0.,planner,[])
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertGreater(action['move_by_speed'][2],0.)
        self.assertEqual(state['controller'],'local_align_to_path')

    def test_heading_wrap_does_not_prevent_settling(self):
        tracker,planner=self.setup_tracker(((-1.,-1.,0.),(-2.,-2.,0.)))
        tracker.action((0.,0.),math.pi,planner,[])
        for tick in range(110):
            heading=math.pi-.001 if tick%2 else -math.pi+.001
            action,state=tracker.action((0.,0.),heading,planner,[])
        self.assertNotEqual(state['controller'],'gait_turn_braking')
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertGreater(action['move_by_speed'][2],0.)

    def test_unsettled_motion_times_out_without_commanding_a_turn(self):
        tracker,planner=self.setup_tracker()
        tracker.action((0.,0.),0.,planner,[])
        for tick in range(1,401):
            action,state=tracker.action((.003*tick,0.),0.,planner,[])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])
        self.assertTrue(state['needs_replan'])
        self.assertEqual(state['controller'],'gait_turn_braking_timeout')

    def test_changed_local_map_is_checked_after_braking(self):
        tracker,planner=self.setup_tracker()
        tracker.action((0.,0.),0.,planner,[])
        yy=planner.limits[1][0]+(np.arange(planner.free.shape[0])+.5)*planner.resolution
        planner.free[abs(yy)>.25,:]=False
        planner.static_clear=distance_transform_edt(planner.free)*planner.resolution>planner.radius
        for _ in range(100):
            action,state=tracker.action((0.,0.),0.,planner,[])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])
        self.assertTrue(state['needs_replan'])


if __name__=='__main__':unittest.main()
