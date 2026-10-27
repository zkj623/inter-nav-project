import math
import unittest

from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery
from grutopia_extension.interactive_navigation.go2_rollout_health import Go2RolloutHealth


def quaternion(tilt=0.,yaw=0.):
    a=math.radians(tilt)/2;b=yaw/2
    return [math.cos(a)*math.cos(b),-math.sin(a)*math.sin(b),math.sin(a)*math.cos(b),math.cos(a)*math.sin(b)]


class AttitudeRecoveryTests(unittest.TestCase):
    def make(self):
        return NavigationAttitudeRecovery(minimum_hold_ticks=6,stable_ticks=3,maximum_hold_ticks=15)

    def test_normal_gait_and_disabled_contact_do_not_trigger(self):
        s=self.make()
        self.assertEqual(s.update([0,0,.4],quaternion(17.))['state'],'idle')
        self.assertEqual(s.update([0,0,.4],quaternion(25.),enabled=False)['state'],'idle')

    def test_hold_has_hysteresis_and_requires_settled_window(self):
        s=self.make();r=s.update([0,0,.4],quaternion(20.))
        self.assertTrue(r['started']);self.assertEqual(r['state'],'hold')
        for tilt in [15.,9.,12.,9.,9.]:
            self.assertEqual(s.update([0,0,.4],quaternion(tilt))['state'],'hold')
        self.assertEqual(s.update([0,0,.4],quaternion(9.))['state'],'released')
        self.assertFalse(s.active)

    def test_level_but_still_sliding_does_not_resume(self):
        s=self.make();s.update([0,0,.4],quaternion(20.))
        for i in range(1,10):
            self.assertEqual(s.update([i*.025,0,.4],quaternion())['state'],'hold')

    def test_level_but_rotating_does_not_resume(self):
        s=self.make();s.update([0,0,.4],quaternion(20.))
        for i in range(1,10):
            self.assertEqual(s.update([0,0,.4],quaternion(yaw=i*.04))['state'],'hold')

    def test_yaw_wrap_is_not_false_motion_and_minimum_hold_is_enforced(self):
        s=self.make();s.update([0,0,.4],quaternion(20.))
        for i in range(4):
            self.assertEqual(s.update([0,0,.4],quaternion(yaw=(-1)**i*(math.pi-.002)))['state'],'hold')
        self.assertEqual(s.update([0,0,.4],quaternion(yaw=math.pi))['state'],'released')

    def test_unsettled_timeout_is_terminal(self):
        s=self.make()
        for _ in range(14):self.assertEqual(s.update([0,0,.4],quaternion(20.))['state'],'hold')
        self.assertEqual(s.update([0,0,.4],quaternion(20.))['state'],'failed')
        self.assertEqual(s.update([0,0,.4],quaternion())['state'],'failed')

    def test_a_new_tilt_episode_requires_a_new_hold(self):
        s=self.make();s.update([0,0,.4],quaternion(20.))
        for _ in range(5):r=s.update([0,0,.4],quaternion())
        self.assertEqual(r['state'],'released')
        self.assertTrue(s.update([0,0,.4],quaternion(20.))['started'])
        self.assertEqual(s.report()['episodes'],2)

    def test_supervisor_does_not_erase_rollout_health_failure(self):
        s=self.make();health=Go2RolloutHealth(.01)
        for tilt in [20.,31.,0.]:
            health.observe([0,0,.4],quaternion(tilt));s.update([0,0,.4],quaternion(tilt))
        self.assertFalse(health.healthy)
        self.assertAlmostEqual(health.max_tilt,31.)


if __name__=='__main__':unittest.main()
