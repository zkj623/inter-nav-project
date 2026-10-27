import math,unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_contact_control import safe_contact_retreat,observed_contact_retreat
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


class OnlineContactRetreatTests(unittest.TestCase):
    def make(self,boundary=.55):
        _,col=np.indices((120,120));free=-3+(col+.5)*.05<boundary
        p=SidePushPlanner(free,[[-3,3],[-3,3]],resolution=.05);p.navigation_half_width=.22
        box=BoxState('contact',(-.7,0),(.8,.6,.6));limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0.)
        return p,box,limiter
    def test_observed_prefix_can_be_used_without_revealing_the_unknown_remainder(self):
        p,box,limiter=self.make();before=p.free.copy()
        self.assertIsNone(safe_contact_retreat(p,[],[0,0],math.pi,[-1,0],.4))
        answer=observed_contact_retreat(p,[box],[0,0],math.pi,[-1,0],limiter,.02)
        self.assertIsNotNone(answer);command,target=answer
        self.assertLess(command[0],0.)
        self.assertTrue(body_translation_recovery_clear(p,[box],[0,0],math.pi,target,require_clear_end=False))
        np.testing.assert_array_equal(p.free,before)
    def test_no_short_observed_corridor_means_no_motion(self):
        p,box,limiter=self.make(boundary=.40)
        self.assertIsNone(observed_contact_retreat(p,[box],[0,0],math.pi,[-1,0],limiter,.02))
    def test_existing_forward_momentum_must_settle_before_retreat(self):
        p,box,limiter=self.make();limiter.measured=np.array([.2,0,0])
        self.assertIsNone(observed_contact_retreat(p,[box],[0,0],math.pi,[-1,0],limiter,.02))
    def test_smoothed_previous_fast_command_is_checked_too(self):
        p,box,limiter=self.make();limiter.accept([-.5,0,0],0.)
        self.assertIsNone(observed_contact_retreat(p,[box],[0,0],math.pi,[-1,0],limiter,.02))

if __name__=='__main__':unittest.main()
