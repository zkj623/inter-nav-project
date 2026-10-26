import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_control import safe_contact_retreat,push_centerline_velocity,ContactAlignmentGate
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

    def test_centerline_feedback_limits_drift_with_gait_lag_and_bias(self):
        # A bounded first-order actuator model, not an Isaac success test.
        # The old 0.08 m/s cap cannot reject this measured-scale lateral bias.
        position=0.;velocity=0.;maximum_error=0.
        for _ in range(1000):
            command=push_centerline_velocity((1.,0.),.4,-position,velocity)
            velocity+=(command[1]+.14-velocity)*.02/.25
            position+=velocity*.02
            maximum_error=max(maximum_error,abs(position))
        self.assertLess(maximum_error,.07)
        self.assertLess(abs(velocity),.001)

    def test_feedback_rotates_with_the_opposite_push_direction(self):
        forward=push_centerline_velocity((1.,0.),.4,.03,-.10)
        backward=push_centerline_velocity((-1.,0.),.4,.03,-.10)
        np.testing.assert_allclose(forward,-backward)

    def test_alignment_does_not_switch_axes_at_noisy_yaw_threshold(self):
        gate=ContactAlignmentGate()
        self.assertEqual(gate.select(.08,.101,.10),'lateral')
        for yaw in [.099,.102,.098,.101]:
            self.assertEqual(gate.select(.05,yaw,.10),'lateral')
        self.assertEqual(gate.select(.03,.101,.10),'yaw')
        for yaw in [.099,.09,.071]:
            self.assertEqual(gate.select(.03,yaw,.10),'yaw')
        self.assertIsNone(gate.select(.03,.069,.10))


if __name__=='__main__':unittest.main()
