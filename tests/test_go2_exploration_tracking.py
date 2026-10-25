import unittest
import numpy as np
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker,decision_buffer_recovery
from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


class ExplorationTrackerTests(unittest.TestCase):
    def setUp(self):
        self.planner=BoxPushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]],robot_radius=.1)

    def test_small_heading_error_keeps_continuous_translation(self):
        tracker=ExplorationPathTracker((0,0),[(1,0)])
        action,_=tracker.action((0,0),.10,self.planner,[])
        forward,lateral,yaw=action['move_by_speed']
        self.assertGreater(forward,.30)
        self.assertLess(lateral,0.)
        self.assertAlmostEqual(yaw,-.18,places=6)

    def test_near_waypoint_chord_fallback_has_occupancy_contract(self):
        tracker=ExplorationPathTracker((0,0),[(.005,0)])
        calls=[]
        tracker._tracking_segment_clear=lambda left,right:bool(calls.append(1) or len(calls)>1)
        action,state=tracker.action((0,0),0,self.planner,[])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])
        self.assertTrue(state['needs_replan'])

    def test_sideways_route_keeps_the_heading_used_for_footprint_planning(self):
        planner=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        tracker=ExplorationPathTracker((0,0),[(0,1)],fixed_heading=0.)
        action,state=tracker.action((0,0),.10,planner,[])
        self.assertEqual(planner.robot_heading,0.)
        self.assertAlmostEqual(action['move_by_speed'][2],-.18)
        self.assertGreater(action['move_by_speed'][1],.19)
        # Original translation remains parallel to the checked world segment.
        vx,vy,_=action['move_by_speed']
        self.assertAlmostEqual(np.cos(.1)*vx-np.sin(.1)*vy,0.,places=6)
        self.assertFalse(state['needs_replan'])

    def test_replanning_retains_consistent_body_heading_in_a_narrow_corridor(self):
        free=np.ones((201,201),bool)
        free[:85]=False;free[116:]=False
        planner=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        planner.robot_heading=0.
        path=planner.path((0,0),(1,0),[])
        self.assertTrue(path)
        tracker=ExplorationPathTracker((0,0),path,fixed_heading=0.)
        action,state=tracker.action((0,0),np.pi/4,planner,[])
        self.assertEqual(planner.robot_heading,0.)
        self.assertTrue(planner.path((0,0),(1,0),[]))
        self.assertFalse(state['needs_replan'])

    def test_near_corner_is_not_skipped_before_the_next_segment_is_clear(self):
        free=np.ones((201,201),bool)
        free[104:117,104:113]=False
        planner=BoxPushPlanner(free,[[-2.5,2.5],[-2.5,2.5]],robot_radius=.01)
        tracker=ExplorationPathTracker((.18,0),[(.4,0),(.4,.6)],fixed_heading=0.)
        action,state=tracker.action((.18,0),0.,planner,[])
        self.assertEqual(tracker._waypoint_index,0)
        self.assertFalse(state['needs_replan'])
        self.assertGreater(action['move_by_speed'][0],0.)
        self.assertAlmostEqual(action['move_by_speed'][1],0.)

    def test_long_shortcut_corrects_local_lateral_drift(self):
        planner=BoxPushPlanner(np.ones((501,501),bool),[[-2.5,10.0],[-2.5,10.0]],robot_radius=.1)
        tracker=ExplorationPathTracker((0,0),[(8,0)],fixed_heading=0.)
        action,state=tracker.action((1,.3),0.,planner,[])
        self.assertAlmostEqual(state['target'][0],1.55)
        self.assertAlmostEqual(state['target'][1],0.)
        self.assertLess(action['move_by_speed'][1],-.15)
        self.assertGreater(action['move_by_speed'][0],.3)

    def test_redecision_leaves_wall_buffer_before_declaring_no_route(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        planner=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        position=np.array([-.60,0.]);goal=(.2,0.)
        self.assertTrue(planner.free[planner.cell(position)])
        self.assertEqual(planner.decide(position,goal,[]).kind,'failed')
        action,state=decision_buffer_recovery(position,np.pi/2,planner,[],goal,np.pi/2)
        self.assertTrue(state['recovering_buffer'])
        self.assertAlmostEqual(action['move_by_speed'][1],-.25,places=2)
        self.assertEqual(action['move_by_speed'][2],0.)
        moved=position+np.array([-action['move_by_speed'][1],action['move_by_speed'][0]])
        self.assertFalse(planner.blocked([])[planner.cell(moved)])
        self.assertEqual(planner.decide(moved,goal,[]).kind,'navigate')

    def test_redecision_does_not_drive_out_of_real_center_collision(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        planner=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        self.assertIsNone(decision_buffer_recovery((-.90,0.),np.pi/2,planner,[],(.2,0.),np.pi/2))

    def test_redecision_keeps_clear_start_in_normal_navigation(self):
        planner=SidePushPlanner(np.ones((101,101),bool),[[-1.25,1.25],[-1.25,1.25]])
        self.assertIsNone(decision_buffer_recovery((0.,0.),0.,planner,[],(.5,0.),0.))


if __name__=='__main__':unittest.main()
