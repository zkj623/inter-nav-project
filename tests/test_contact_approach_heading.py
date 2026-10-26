import math
import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_body_blocked,body_rotation_clear
from grutopia_extension.interactive_navigation.go2_contact_control import approach_alignment_ready


class ContactApproachHeadingTests(unittest.TestCase):
    def test_northward_last_segment_can_handoff_to_safe_eastward_push(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        y=-2.5+(np.arange(201)+.5)*p.resolution
        p.free[y>.655,:]=False
        start=(0.,.05);contact=(.10,.2775)
        walking=math.pi/2;push=0.
        self.assertFalse(approach_alignment_ready(start,contact,p,navigation_body_blocked(p,[],walking)))
        self.assertTrue(approach_alignment_ready(start,contact,p,navigation_body_blocked(p,[],push)))
        self.assertTrue(body_rotation_clear(p,[],start,walking,push))


if __name__=='__main__':unittest.main()
