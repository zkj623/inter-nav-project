import unittest,math
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_path_tracking import Go2VelocityPathTracker


class BodyFootprintTests(unittest.TestCase):
    def test_heading_controls_whether_body_fits_doorway(self):
        free=np.zeros((161,161),bool);free[:,70:91]=True
        planner=SidePushPlanner(free,[[-2,2],[-2,2]])
        center=planner.cell((0,0))
        planner.robot_heading=0
        self.assertTrue(planner.blocked([])[center])
        planner.robot_heading=math.pi/2
        self.assertFalse(planner.blocked([])[center])

    def test_sidestep_does_not_turn_body_toward_narrow_corner(self):
        planner=SidePushPlanner(np.ones((161,161),bool),[[-2,2],[-2,2]])
        planner.robot_heading=0
        tracker=Go2VelocityPathTracker((0,0),[(0,1,0)],fixed_heading=0)
        action,_=tracker.action((0,0),0,planner,[])
        self.assertGreater(action['move_by_speed'][1],0)
        self.assertAlmostEqual(action['move_by_speed'][2],0)

    def test_turn_checks_intermediate_headings_not_only_endpoints(self):
        free=np.ones((161,161),bool)
        # A small fixture lies outside the axis-aligned bodies but inside
        # the front corner swept while the quadruped turns.
        free[92,92]=False
        planner=SidePushPlanner(free,[[-2,2],[-2,2]])
        center=planner.cell((0,0))
        planner.robot_heading=0
        self.assertFalse(planner.blocked([])[center])
        planner.robot_heading=math.pi/2
        self.assertFalse(planner.blocked([])[center])
        self.assertTrue(planner.turning_blocked([],0,math.pi/2,margin=0)[center])
        self.assertAlmostEqual(planner.robot_heading,math.pi/2)

    def test_robot_turns_to_fit_doorway_before_advancing(self):
        free=np.zeros((161,161),bool);free[:,70:91]=True
        planner=SidePushPlanner(free,[[-2,2],[-2,2]]);planner.robot_heading=math.pi/2
        tracker=Go2VelocityPathTracker((0,0),[(0,1,0)],fixed_heading=math.pi/2)
        action,_=tracker.action((0,0),0,planner,[])
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertGreater(action['move_by_speed'][2],0.)
        action,_=tracker.action((0,0),math.pi/2,planner,[])
        self.assertGreater(action['move_by_speed'][0],0.)


if __name__=='__main__':unittest.main()
