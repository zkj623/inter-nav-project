import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,SmoothExplorationTracker,smooth_recovery_action
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear,body_rotation_clear
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl


class SmoothControlTests(unittest.TestCase):
    def test_sharp_turn_brakes_before_combining_reverse_strafe_and_yaw(self):
        p=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
        p.navigation_half_width=.22;p.control_time=.01
        limiter=VelocitySlewLimiter();limiter.accept([.4,0,0],0)
        limiter.measured=np.array([.4,0,0])
        tracker=SmoothExplorationTracker([0,0],[[-1,1]],limiter)
        action,state=tracker.action(np.zeros(2),0,p,[])
        self.assertTrue(state['local_staging'])
        self.assertEqual(state['staging_phase'],'staging_brake')
        np.testing.assert_array_equal(state['desired_command'],[0,0,0])
        self.assertGreater(action['move_by_speed'][0],0)
        self.assertLess(action['move_by_speed'][0],.4)
        self.assertEqual(action['move_by_speed'][1:],[0.,0.])

    def test_sharp_turn_keeps_heading_until_settled_instead_of_chasing_new_carrot(self):
        p=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
        p.navigation_half_width=.22;p.control_time=.01
        limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0)
        tracker=SmoothExplorationTracker([0,0],[[0,2]],limiter)
        tracker.action(np.zeros(2),0,p,[])
        original=limiter.staging
        replacement=SmoothExplorationTracker([0,0],[[2,0]],limiter)
        p.control_time=.02
        action,state=replacement.action(np.zeros(2),0,p,[])
        self.assertIs(limiter.staging,original)
        self.assertAlmostEqual(state['target_heading_rad'],np.pi/2)

    def test_stalled_short_carrot_aligns_to_the_outgoing_corner_segment(self):
        p=self.corner();p.free[:]=True;p.control_time=2.
        x=-2.5+(np.arange(201)+.5)*p.resolution;xx,yy=np.meshgrid(x,x)
        p.free[((xx-1.20)**2+(yy-.10)**2<=.13**2)|((xx>=-.6)&(xx<=.6)&(yy>=-.9)&(yy<=-.3))]=False
        start=np.array([.55,0.])
        limiter=VelocitySlewLimiter();limiter.accept([0,0,0],1.99)
        limiter.progress_position=start.copy();limiter.progress_time=0.
        tracker=SmoothExplorationTracker(start,[[.75,0],[1.15,-.75],[1.9,-1.5]],limiter)
        action,state=tracker.action(start,0,p,[])
        self.assertTrue(state['local_staging'])
        self.assertAlmostEqual(state['target_heading_rad'],np.arctan2(-.75,.4))
        self.assertTrue(body_translation_recovery_clear(p,[],start,0,limiter.staging.target))
    def test_staging_completion_replans_before_resuming_the_old_carrot(self):
        from types import SimpleNamespace
        p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
        p.control_time=1.;limiter=VelocitySlewLimiter();limiter.accept([0,0,0],.99)
        limiter.staging=SimpleNamespace(update=lambda *args:None)
        tracker=SmoothExplorationTracker([0,0],[[2,0]],limiter)
        action,state=tracker.action(np.array([.8,.3]),0,p,[])
        self.assertTrue(state['staging_completed']);self.assertTrue(state['needs_replan'])
        np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])
        self.assertIsNone(limiter.staging)
    def corner(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        x=-2.5+(np.arange(201)+.5)*p.resolution;xx,yy=np.meshgrid(x,x)
        p.free[((xx-.65)**2+(yy-.10)**2<=.13**2)|((xx>=-.6)&(xx<=.08)&(yy>=-.9)&(yy<=-.45))]=False
        p.navigation_half_width=.22;p.full_navigation_footprint=True
        return p
    def test_planar_norm_and_yaw_acceleration_are_bounded(self):
        s=VelocitySlewLimiter();s.accept([0,0,0],0)
        v=s.preview([.5,.18,1.2],.02)
        self.assertLessEqual(np.linalg.norm(v[:2]),.65*.02+1e-10)
        self.assertLessEqual(abs(v[2]),1.8*.02+1e-10)
    def test_direction_reversal_and_variable_physics_steps_are_continuous(self):
        s=VelocitySlewLimiter();s.accept([.4,0,.5],0)
        t=0
        for dt in [.005,.005,.020,.005]:
            t+=dt;old=s.command.copy();v=s.preview([-.3,0,-1],t)
            self.assertLessEqual(np.linalg.norm(v[:2]-old[:2]),.65*dt+1e-10)
            self.assertLessEqual(abs(v[2]-old[2]),1.8*dt+1e-10);s.accept(v,t)
    def test_same_limiter_survives_path_replacement(self):
        s=VelocitySlewLimiter();s.accept([.3,.02,.1],1.)
        tracker=SmoothExplorationTracker((0,0),[(1,0),(2,0)],s)
        self.assertIs(tracker.limiter,s)
        np.testing.assert_array_equal(tracker.limiter.command,[.3,.02,.1])
    def test_straight_navigation_reaches_baseline_speed_without_periodic_stops(self):
        p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22;p.full_navigation_footprint=True
        s=VelocitySlewLimiter();s.accept([0,0,0],0);f=SmoothExplorationTracker((0,0),[(2,0)],s)
        values=[]
        for i in range(1,101):
            p.control_time=i*.01;a,state=f.action(np.array([0.,0.]),0,p,[]);values.append(a['move_by_speed'][0])
        self.assertGreater(values[-1],.49)
        self.assertTrue(all(x>0 for x in values));self.assertLess(max(np.abs(np.diff(values))),.0066)

    def test_blind_band_triggers_safe_observation_then_resumes_after_scan(self):
        p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22;p.full_navigation_footprint=True
        p.planning_observed=np.ones_like(p.free)
        hole=p.cell((.7,.05));p.free[hole]=False;p.planning_observed[hole]=False
        p.control_time=.01;s=VelocitySlewLimiter();s.accept([0,0,0],0)
        tracker=SmoothExplorationTracker((0,0),[(1.5,0)],s)
        a,state=tracker.action(np.array([0.,0.]),0,p,[])
        self.assertTrue(state['observation_motion']);self.assertFalse(state['needs_replan'])
        self.assertEqual(a['move_by_speed'][:2],[0.,0.]);self.assertGreater(abs(a['move_by_speed'][2]),0.)
        self.assertFalse(p.free[hole]);self.assertFalse(p.planning_observed[hole])
        p.free[hole]=True;p.planning_observed[hole]=True;p.control_time=.03
        a,state=tracker.action(np.array([0.,0.]),0,p,[])
        self.assertFalse(state.get('observation_motion',False));self.assertGreater(a['move_by_speed'][0],0.)

    def test_velocity_reference_matches_original_exploration_with_same_limits(self):
        p=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
        p.navigation_half_width=.22;p.full_navigation_footprint=True;p.control_time=.02
        original=ExplorationPathTracker((0,0),[(2,1)],max_forward_speed=.5,max_lateral_speed=.18)
        expected,_=original.action(np.array([0.,0.]),0,p,[])
        limiter=VelocitySlewLimiter(linear_acceleration=1e5,yaw_acceleration=1e5);limiter.accept([0,0,0],0)
        tracker=SmoothExplorationTracker((0,0),[(2,1)],limiter)
        actual,state=tracker.action(np.array([0.,0.]),0,p,[])
        np.testing.assert_allclose(state['desired_command'],expected['move_by_speed'])
        np.testing.assert_allclose(actual['move_by_speed'],expected['move_by_speed'])

    def test_observation_timeout_requests_replan_instead_of_standing_forever(self):
        p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22;p.full_navigation_footprint=True;p.control_time=10.
        p.planning_observed=np.ones_like(p.free);hole=p.cell((.7,.05))
        p.free[hole]=False;p.planning_observed[hole]=False
        s=VelocitySlewLimiter();s.accept([0,0,0],9.99)
        tracker=SmoothExplorationTracker((0,0),[(1.5,0)],s);tracker._observation_started=0.
        action,state=tracker.action(np.array([0.,0.]),0,p,[])
        self.assertTrue(state['needs_replan']);self.assertTrue(state['observation_timeout_or_blocked'])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])

    def test_partial_escape_can_reduce_existing_overlap_without_new_cells(self):
        xx=-3+(np.arange(120)+.5)*.05
        free=np.broadcast_to(xx<.45,(120,120)).copy()
        p=SidePushPlanner(free,[[-3,3],[-3,3]],resolution=.05);p.navigation_half_width=.22
        self.assertFalse(body_translation_recovery_clear(p,[],(0,0),0,(-.03,0)))
        self.assertTrue(body_translation_recovery_clear(p,[],(0,0),0,(-.03,0),require_clear_end=False))
        self.assertTrue(body_translation_recovery_clear(p,[],(0,0),0,(-.15,0)))
        p.free[p.cell((-.4,.1))]=False
        self.assertFalse(body_translation_recovery_clear(p,[],(0,0),0,(-.15,0)))

    def test_local_overlap_does_not_cancel_small_safe_recovery_step(self):
        xx=-3+(np.arange(120)+.5)*.05
        p=SidePushPlanner(np.broadcast_to(xx<.45,(120,120)).copy(),[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22;p.full_navigation_footprint=True;p.control_time=.01
        s=VelocitySlewLimiter();s.accept([0,0,0],0);s.observe([-.15,0,0],0)
        tracker=SmoothExplorationTracker((0,0),[(0,-2)],s)
        action,state=tracker.action(np.zeros(2),0,p,[])
        self.assertLess(action['move_by_speed'][0],0)
        self.assertTrue(state['overlap_recovery']);self.assertTrue(state['body_motion_safe'])

    def test_recovery_smoothing_preserves_certified_yaw_and_brakes_translation(self):
        p=self.corner();s=VelocitySlewLimiter();s.accept([0,0,0],0)
        action=smooth_recovery_action({'move_by_speed':[0,0,.8]},(0,0),0,p,[],s,.02)
        self.assertGreater(action['move_by_speed'][2],0)
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        s.measured=np.array([.2,0,0])
        self.assertEqual(smooth_recovery_action({'move_by_speed':[0,0,.8]},(0,0),0,p,[],s,.02)['move_by_speed'],[0.,0.,0.])

    def test_stalled_corner_selects_a_full_body_safe_turning_pose(self):
        p=self.corner();p.control_time=2.
        s=VelocitySlewLimiter();s.accept([0,0,0],1.99);s.progress_time=0.;s.progress_position=np.zeros(2)
        tracker=SmoothExplorationTracker((0,0),[(2,-1.7)],s)
        action,state=tracker.action(np.zeros(2),0,p,[])
        self.assertTrue(state['local_staging']);self.assertTrue(tracker.staging_active)
        self.assertTrue(body_translation_recovery_clear(p,[],(0,0),0,s.staging.target))
        self.assertTrue(body_rotation_clear(p,[],s.staging.target,0,s.staging.goal_heading,margin=.025))
        self.assertEqual(action['move_by_speed'][2],0.)

    def test_turn_waits_for_actual_staging_translation_to_settle(self):
        p=self.corner();recovery=LocalStagingControl((0,0),0,-.7,p,[],0.)
        s=VelocitySlewLimiter();s.command=np.array([.2,0,0]);s.measured=np.array([.2,0,0])
        action,state=recovery.update(recovery.target,0,p,[],s,.1)
        np.testing.assert_array_equal(action,[0,0,0]);self.assertEqual(state['staging_phase'],'staging_brake')
        s.command[:]=0;s.measured[:]=0
        recovery.update(recovery.target,0,p,[],s,.2)
        recovery.update(recovery.target,0,p,[],s,.45)
        action,state=recovery.update(recovery.target,0,p,[],s,.46)
        self.assertLess(action[2],0);self.assertEqual(action[:2].tolist(),[0.,0.])


if __name__=='__main__':unittest.main()
