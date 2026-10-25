import unittest
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


class SideParkingTests(unittest.TestCase):
    def setUp(self):
        self.planner=SidePushPlanner(np.ones((161,161),bool),[[-2,2],[-2,2]],
            passages={'box':{'axis':[0,1],'center':[0,0]}},robot_radius=.30)
        self.box=BoxState('box',(0,0),(.85,.55,.95),0,6)
        self.planner.configure_route((0,-1.5),(0,1.5),[self.box])

    def test_forward_push_is_not_accepted_as_side_clearance(self):
        self.assertFalse(self.planner.side_parking_allowed(self.box,(0,1.3),(0,1)))

    def test_side_parking_requires_full_robot_passage_width(self):
        self.assertFalse(self.planner.side_parking_allowed(self.box,(.7,0),(1,0)))
        self.assertTrue(self.planner.side_parking_allowed(self.box,(.95,0),(1,0)))
        clearance=self.planner.clearance(self.box,(.95,0))
        self.assertTrue(clearance['side_parked'])
        self.assertAlmostEqual(clearance['robot_passage_width_m'],.76)

    def test_selected_contact_push_is_perpendicular_to_passage(self):
        decision=self.planner.decide((0,-1.5),(0,.1),[self.box])
        self.assertEqual(decision.kind,'push')
        self.assertAlmostEqual(decision.push.direction[1],0)
        self.assertGreater(abs(decision.push.destination_xy[0]),.8)
        self.assertLess(abs(decision.push.destination_xy[1]),.05)

    def test_rotated_carton_footprint_is_used_for_clearance(self):
        rotated=BoxState('box',(0,0),(.85,.55,.95),90,6)
        clearance=self.planner.clearance(rotated,(.7,0))
        self.assertAlmostEqual(clearance['required_lateral_offset_m'],.275+.30+.08)


if __name__=='__main__':unittest.main()
