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
    def test_staging_can_keep_the_gait_until_braking_without_exceeding_translation_bound(self):
        p,limiter,assist=self.make()
        assist.update([0,0,1],[0,0],0,p,[],limiter,0.,maintain_arc=True)
        assist.update([0,0,1],[0,0],0,p,[],limiter,1.,maintain_arc=True)
        cmd,state=assist.update([0,0,1],[-.05,0],.25,p,[],limiter,1.3,maintain_arc=True)
        self.assertLess(cmd[0],0.)
        cmd,state=assist.update([0,0,1],[-.21,0],.9,p,[],limiter,2.,maintain_arc=True)
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
    def test_last_centimetres_keep_a_gait_command_then_brake_on_actual_motion(self):
        from unittest.mock import patch
        from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
        p,limiter,_=self.make();stage=LocalStagingControl([0,0],0,-.6,p,[],0.)
        target=np.array([.04,0.]);stage.target=target
        module='grutopia_extension.interactive_navigation.local_staging_control.'
        with patch(module+'body_rotation_clear',return_value=False),patch(module+'stepping_arc_clear',return_value=False),patch.object(stage.selector,'_select_stage',return_value=target):
            command,_=stage.update([0,0],0,p,[],limiter,.1)
            self.assertAlmostEqual(command[0],.2)
            limiter.measured=np.array([.2,0,0])
            command,_=stage.update([0,0],0,p,[],limiter,.2)
            np.testing.assert_array_equal(command,[0,0,0])

class StagingYawRateTests(unittest.TestCase):
    def make(self):
        from grutopia_extension.interactive_navigation.local_staging_control import StagingYawRate
        return StagingYawRate()

    def test_requires_sustained_directional_motion_before_cruising(self):
        rate=self.make()
        self.assertEqual(rate.request(2.8,0.,0.),1.)
        self.assertEqual(rate.request(2.8,.5,.1),1.)
        self.assertEqual(rate.request(2.8,.5,.20),1.)
        self.assertEqual(rate.request(2.8,.5,.23),.6)
        self.assertEqual(rate.request(2.8,.5,1.),.6)

    def test_noise_or_rotation_in_wrong_direction_does_not_confirm_motion(self):
        rate=self.make()
        for now,measured in [(0.,-.7),(.2,-.7),(.3,.5),(.4,.1),(.5,.5),(.6,.5)]:
            self.assertEqual(rate.request(2.,measured,now),1.)

    def test_short_response_dip_keeps_cruise_but_sustained_stall_rearms(self):
        rate=self.make()
        rate.request(2.,.5,0.)
        self.assertEqual(rate.request(2.,.5,.13),.6)
        self.assertEqual(rate.request(2.,.1,.2),.6)
        self.assertEqual(rate.request(2.,.5,.8),.6)
        self.assertEqual(rate.request(2.,.1,.9),.6)
        self.assertEqual(rate.request(2.,.1,1.6),.6)
        self.assertEqual(rate.request(2.,.1,1.71),1.)

    def test_reverse_turn_and_new_maneuver_require_new_confirmation(self):
        rate=self.make()
        rate.request(-2.,-.5,0.)
        self.assertEqual(rate.request(-2.,-.5,.13),-.6)
        self.assertEqual(rate.request(2.,-.5,.3),1.)
        rate.request(2.,.5,.4)
        self.assertEqual(rate.request(2.,.5,.53),.6)
        rate.reset()
        self.assertEqual(rate.request(2.,0.,.6),1.)

    def test_actual_stage_uses_measured_cruise_and_retains_predictive_braking(self):
        from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
        p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22
        limiter=VelocitySlewLimiter()
        stage=LocalStagingControl([0,0],0,2.,p,[],0.);stage.phase='staging_turn'
        limiter.measured=np.array([0.,0.,.5]);limiter.command=np.array([0.,0.,1.])
        command,_=stage.update([0,0],0.,p,[],limiter,.1)
        self.assertEqual(command[2],1.)
        command,_=stage.update([0,0],.1,p,[],limiter,.23)
        self.assertEqual(command[2],.6)
        command,state=stage.update([0,0],1.9,p,[],limiter,.3)
        np.testing.assert_array_equal(command,[0.,0.,0.])
        self.assertEqual(state['staging_phase'],'staging_turn_brake')


if __name__=='__main__':unittest.main()
