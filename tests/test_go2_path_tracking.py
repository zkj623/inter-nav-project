import unittest
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner,BoxState
from grutopia_extension.interactive_navigation.go2_path_tracking import Go2VelocityPathTracker
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.planner=BoxPushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]],robot_radius=.10)

    def test_corner_lookahead_never_cuts_inflated_obstacle(self):
        box=BoxState('corner',(-.27,.25),(.30,.30,.5))
        blocked=self.planner.blocked([box])
        tracker=Go2VelocityPathTracker((-1,0),[(0,0,0),(0,1,0)])
        target,recovery=tracker.target((-.10,0),self.planner,blocked)
        self.assertFalse(AStarMapPlanner.segment_clear(self.planner.cell((-.10,0)),self.planner.cell((0,.25)),blocked))
        self.assertFalse(recovery)
        self.assertTrue(AStarMapPlanner.segment_clear(self.planner.cell((-.10,0)),self.planner.cell(target),blocked))
        self.assertAlmostEqual(target[1],0.)

    def test_inflation_buffer_recovery_targets_traversable_cell(self):
        box=BoxState('obstacle',(0,0),(.2,.2,.5))
        blocked=self.planner.blocked([box]);tracker=Go2VelocityPathTracker((-.5,0),[(.5,0,0)])
        target,recovery=tracker.target((-.15,0),self.planner,blocked)
        self.assertTrue(recovery)
        self.assertFalse(blocked[self.planner.cell(target)])

    def test_body_commands_follow_world_target_while_rotating(self):
        tracker=Go2VelocityPathTracker((-1,0),[(1,0,0)])
        action,_=tracker.action((-1,0),np.pi/2,self.planner,[])
        forward,lateral,rotation=action['move_by_speed']
        self.assertLess(lateral,0.)
        self.assertLess(rotation,0.)
        self.assertLessEqual(abs(lateral),.25)

    def test_lateral_drift_is_corrected_before_forward_progress(self):
        tracker=Go2VelocityPathTracker((-1,0),[(1,0,0)],fixed_heading=0)
        action,state=tracker.action((-.5,.08),0,self.planner,[])
        self.assertTrue(state['correcting_drift'])
        self.assertAlmostEqual(action['move_by_speed'][0],0.)
        self.assertLess(action['move_by_speed'][1],-.25)
        action,state=tracker.action((-.5,.005),0,self.planner,[])
        self.assertFalse(state['correcting_drift'])
        self.assertGreater(action['move_by_speed'][0],0.)


if __name__=='__main__':unittest.main()
