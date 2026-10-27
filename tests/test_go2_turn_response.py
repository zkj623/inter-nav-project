import unittest
import numpy as np
from grutopia_extension.interactive_navigation.go2_turn_response import TurnResponseAssist
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


class TurnResponseTests(unittest.TestCase):
    def make(self):
        p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22
        return p,VelocitySlewLimiter(),TurnResponseAssist()
    def test_sustained_stall_uses_a_short_collision_checked_arc(self):
        p,limiter,assist=self.make()
        cmd,state=assist.update([0,0,1],[0,0],0,p,[],limiter,0.)
        np.testing.assert_array_equal(cmd,[0,0,1])
        cmd,state=assist.update([0,0,1],[0,0],0,p,[],limiter,1.)
        self.assertEqual(cmd[0],-.2)
        cmd,state=assist.update([0,0,1],[-.1,0],.2,p,[],limiter,1.4)
        np.testing.assert_array_equal(cmd,[0,0,1]);self.assertEqual(state['turn_assist'],'gait_started')
    def test_measured_turn_progress_does_not_add_unrequested_translation(self):
        p,limiter,assist=self.make()
        assist.update([0,0,1],[0,0],0,p,[],limiter,0.)
        cmd,state=assist.update([0,0,1],[0,0],.2,p,[],limiter,1.)
        np.testing.assert_array_equal(cmd,[0,0,1])
    def test_unknown_space_cannot_be_used_to_start_the_turn(self):
        p,limiter,assist=self.make();p.free[:]=False
        assist.update([0,0,1],[0,0],0,p,[],limiter,0.)
        cmd,state=assist.update([0,0,1],[0,0],0,p,[],limiter,1.)
        np.testing.assert_array_equal(cmd,[0,0,1]);self.assertEqual(state['turn_assist'],'no_observed_safe_arc')
    def test_staging_reserves_space_for_the_actual_turning_arc(self):
        from grutopia_extension.interactive_navigation.go2_turn_response import stepping_arc_clear
        from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear,body_translation_recovery_clear
        from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver
        p,_,_=self.make();rows,cols=np.indices(p.free.shape)
        x=-3+(cols+.5)*.05;y=-3+(rows+.5)*.05
        p.free[:]=((x>-.45)&(x<.65))|(y>.3)
        position=np.array([0.,0.])
        self.assertTrue(body_rotation_clear(p,[],position,0.,-.6,margin=.025))
        self.assertFalse(stepping_arc_clear(p,[],position,0.,-1.))
        target=LocalTurnManeuver(-.6)._select_stage(p,[],position,0.,require_stepping_arc=True)
        self.assertIsNotNone(target);self.assertTrue(stepping_arc_clear(p,[],target,0.,-1.))
        self.assertTrue(body_translation_recovery_clear(p,[],position,0.,target))

if __name__=='__main__':unittest.main()
