import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,SmoothExplorationTracker


class SmoothControlTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
