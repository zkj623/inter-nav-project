import math,unittest
from grutopia_extension.interactive_navigation.go2_rollout_health import Go2RolloutHealth


class RolloutHealthTests(unittest.TestCase):
    def test_a_transient_violation_during_settling_cannot_be_hidden_by_recovery(self):
        health=Go2RolloutHealth(.01)
        health.observe([0,0,.41],[1,0,0,0])
        a=math.radians(35)/2
        health.observe([0,0,.35],[math.cos(a),math.sin(a),0,0])
        health.observe([0,0,.41],[1,0,0,0])
        self.assertFalse(health.healthy)
        self.assertAlmostEqual(health.max_tilt,35.)
        self.assertEqual(health.report()['first_violation']['sample'],2)

    def test_height_is_relative_to_the_one_cm_floor(self):
        health=Go2RolloutHealth(.01)
        health.observe([0,0,.205],[1,0,0,0])
        self.assertFalse(health.healthy)
        self.assertEqual(health.violation['reason'],'base_too_low')


if __name__=='__main__':unittest.main()
