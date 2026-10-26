import math
import unittest
import numpy as np

from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear, body_translation_recovery_clear
from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver


class LocalTurnManeuverTests(unittest.TestCase):
    def planner(self):
        p = SidePushPlanner(np.ones((201, 201), bool), [[-2.5, 2.5], [-2.5, 2.5]])
        x = -2.5 + (np.arange(201) + .5) * p.resolution
        xx, yy = np.meshgrid(x, x)
        obstacle = (xx-.65)**2 + (yy-.10)**2 <= .13**2
        obstacle |= (xx >= -.6) & (xx <= .08) & (yy >= -.9) & (yy <= -.45)
        p.free[obstacle] = False
        return p

    def test_back_up_then_turn_around_corner_without_crossing_geometry(self):
        p = self.planner()
        maneuver = LocalTurnManeuver(-.7)
        self.assertFalse(body_rotation_clear(p, [], (0., 0.), 0., -.7))
        action, state = maneuver.action((0., 0.), 0., p, [])
        self.assertEqual(state['controller'], 'local_turn_staging')
        self.assertFalse(state['needs_replan'])
        self.assertTrue(body_rotation_clear(p, [], maneuver.stage_target, 0., -.7, margin=.025))
        self.assertTrue(body_translation_recovery_clear(p, [], (0., 0.), 0., maneuver.stage_target))
        self.assertEqual(action['move_by_speed'][2], 0.)
        position = np.zeros(2)
        for _ in range(80):
            action, state = maneuver.action(position, 0., p, [])
            if state['controller'] == 'local_align_to_path':
                break
            delta = np.array(action['move_by_speed'][:2]) * .05
            self.assertTrue(body_translation_recovery_clear(p, [], position, 0., position+delta))
            position += delta
        self.assertEqual(state['controller'], 'local_align_to_path')
        self.assertLess(action['move_by_speed'][2], 0.)
        self.assertIsNone(maneuver.action(position, -.7, p, []))

    def test_closed_space_stops_without_unsafe_motion(self):
        p = self.planner()
        p.free[:] = False
        action, state = LocalTurnManeuver(math.pi/2).action((0., 0.), 0., p, [])
        self.assertEqual(action['move_by_speed'], [0., 0., 0.])
        self.assertTrue(state['needs_replan'])

    def test_yaw_wrap_uses_short_rotation(self):
        p = self.planner()
        p.free[:] = True
        action, state = LocalTurnManeuver(-3.).action((0., 0.), 3., p, [])
        self.assertGreater(action['move_by_speed'][2], 0.)
        self.assertEqual(state['controller'], 'local_align_to_path')


if __name__ == '__main__':
    unittest.main()
