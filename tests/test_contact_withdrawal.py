import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_control import safe_contact_retreat
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear


class ContactWithdrawalTests(unittest.TestCase):
    def planner(self):
        return SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])

    def test_withdrawal_can_include_side_escape_from_wall_overlap(self):
        p=self.planner()
        x=-2.5+(np.arange(201)+.5)*p.resolution;xx,yy=np.meshgrid(x,x)
        p.free[(xx>=-1.2)&(xx<=.3)&(yy>=.20)&(yy<=.6)]=False
        self.assertFalse(body_translation_recovery_clear(p,[],(0,0),0,(-.35,0)))
        retreat=safe_contact_retreat(p,[],(0,0),0,(1,0),.30)
        self.assertIsNotNone(retreat)
        velocity,target=retreat
        self.assertLess(velocity[0],0);self.assertLess(velocity[1],0)
        self.assertTrue(body_translation_recovery_clear(p,[],(0,0),0,target))

    def test_no_safe_withdrawal_returns_no_command(self):
        p=self.planner();p.free[:]=False
        self.assertIsNone(safe_contact_retreat(p,[],(0,0),0,(1,0),.30))

    def test_open_space_prefers_straight_withdrawal(self):
        p=self.planner()
        velocity,target=safe_contact_retreat(p,[],(0,0),0,(1,0),.30)
        np.testing.assert_allclose(velocity,(-.35,0))
        self.assertLessEqual(target[0],-.4)


if __name__=='__main__':unittest.main()
