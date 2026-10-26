import math
import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_body_blocked,body_rotation_clear
from grutopia_extension.interactive_navigation.go2_contact_control import approach_alignment_ready
from grutopia_extension.interactive_navigation.go2_contact_control import safe_contact_retreat,contact_pose_ready
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


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

    def test_near_box_westward_alignment_requires_withdrawal_first(self):
        p=SidePushPlanner(np.ones((241,241),bool),[[-3.,3.],[-3.,3.]])
        box=BoxState('B07',(0.,0.),(.861,.665,.70))
        # Relative version of the actual T05 pose which trapped the front paws.
        position=np.array([.682,-.203]);heading=2.242;direction=(-1.,0.)
        self.assertFalse(body_rotation_clear(p,[box],position,heading,math.pi))
        self.assertFalse(contact_pose_ready(position,heading,box.xy,box.dimensions,direction,.4,.1))
        retreat=safe_contact_retreat(p,[],position,heading,direction,.36)
        self.assertIsNotNone(retreat)
        velocity,target=retreat
        self.assertGreater(velocity[0],0.)
        self.assertTrue(body_rotation_clear(p,[box],target,heading,math.pi))


if __name__=='__main__':unittest.main()
