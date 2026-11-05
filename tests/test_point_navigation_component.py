import tempfile
import unittest
from pathlib import Path

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig
from grutopia_extension.interactive_navigation.point_navigation import (
    PointNavigationComponent,
    PointNavigationConfig,
    PointNavigationStatus,
)
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    load_point_navigation_profile,
    programmatic_g1_profile,
    save_point_navigation_profile,
)


def observation(position=(0.0, 0.0, 0.8)):
    return {
        'position': np.asarray(position, dtype=np.float32),
        'orientation': np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        'sensors': {},
    }


class PointNavigationComponentTest(unittest.TestCase):
    def config(self, **overrides):
        values = {
            'goal': (2.0, 0.0, 0.8),
            'mapping': MappingConfig(
                x_limits=(-1.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            'max_steps': 10,
            'use_scene_graph': False,
        }
        values.update(overrides)
        return PointNavigationConfig(**values)

    def test_point_navigation_action_stops_until_lidar_map_exists(self):
        component = PointNavigationComponent(self.config())

        action = component.action(0, observation())

        self.assertEqual(action, {'move_by_speed': [0.0, 0.0, 0.0]})

    def test_live_path_check_replans_only_when_route_becomes_blocked(self):
        component = PointNavigationComponent(self.config(replan_interval_steps=2))
        component.mapping.map.lidar_frames = 1
        component.mapping.map.occupancy.mark_free((0.0, 0.0), radius=0.5)

        component.action(0, observation())
        component.action(2, observation())
        self.assertEqual(component.mapping.replan_count, 0)

        component.mapping.map.occupancy.mark_occupied((1.0, 0.0), radius=0.2, evidence=4.0)
        component.action(4, observation())
        self.assertEqual(component.mapping.replan_count, 1)
        self.assertEqual(component.mapping.replan_reasons, {'blocked': 1})

    def test_changed_goal_replans_even_when_navigation_state_is_unchanged(self):
        component = PointNavigationComponent(self.config())
        component.mapping.map.lidar_frames = 1
        component.mapping.map.occupancy.observed.fill(True)
        robot = observation()

        component.mapping.point_navigation_action((2.0, 0.0, 0.8), robot, step=0)
        component.mapping.point_navigation_action((3.0, 1.0, 0.8), robot, step=1)

        self.assertEqual(component.mapping.map.plan_count, 2)
        self.assertEqual(component.mapping.replan_reasons, {'goal_changed': 1})
        np.testing.assert_allclose(component.mapping._planned_path[-1], (3.0, 1.0, 0.8))
        self.assertEqual(component.mapping._plan_history[-1]['reason'], 'goal_changed')

    def test_zero_length_path_segment_does_not_hide_later_blockage(self):
        component = PointNavigationComponent(self.config())
        runtime = component.mapping
        runtime.map.occupancy.observed.fill(True)
        start = (0.0, 0.0, 0.8)
        runtime._planned_path = (start, (1.0, 0.0, 0.8))
        runtime.map.occupancy.mark_occupied((1.0, 0.0), radius=0.1, evidence=4.0)

        self.assertTrue(runtime._remaining_path_blocked(start))
        self.assertEqual(runtime._last_path_blockage['reason'], 'inflated_occupancy')

    def test_success_fall_and_timeout_are_distinct_terminal_results(self):
        success = PointNavigationComponent(self.config()).evaluate(0, observation((1.8, 0.0, 0.8)))
        fallen = PointNavigationComponent(self.config()).evaluate(0, observation((0.0, 0.0, 0.2)))
        timeout = PointNavigationComponent(self.config()).evaluate(9, observation())

        self.assertEqual(success.status, PointNavigationStatus.SUCCEEDED)
        self.assertEqual(fallen.failure_reason, 'robot_fell')
        self.assertEqual(timeout.failure_reason, 'global_step_limit')
        self.assertTrue(success.as_event()['success'])

    def test_update_tracks_trajectory_and_mapped_clearance(self):
        component = PointNavigationComponent(self.config())
        component.mapping.map.occupancy.mark_occupied((1.0, 0.0), radius=0.05)

        component.update(0, observation((0.0, 0.0, 0.8)))
        component.update(1, observation((0.5, 0.0, 0.8)))
        stats = component.statistics()

        self.assertEqual(stats['trajectory_points'], 2)
        self.assertAlmostEqual(stats['trajectory_length'], 0.5)
        self.assertLessEqual(stats['minimum_mapped_obstacle_clearance'], 0.55)

    def test_static_furniture_intersection_is_reported(self):
        component = PointNavigationComponent(self.config())
        component.seed_static_obstacles(
            [
                {
                    'label': 'chair/example',
                    'minimum_xy': (1.0, -0.3),
                    'maximum_xy': (1.5, 0.3),
                }
            ]
        )

        component.update(0, observation((1.2, 0.0, 0.8)))
        stats = component.statistics()

        self.assertEqual(stats['static_obstacle_boxes'], 1)
        self.assertEqual(stats['trajectory_static_obstacle_violations'], 1)
        self.assertEqual(stats['violated_static_obstacle_labels'], ['chair/example'])

    def test_programmatic_profile_round_trips_json(self):
        profile = programmatic_g1_profile()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'profile.json'
            save_point_navigation_profile(profile, str(output))
            loaded = load_point_navigation_profile(str(output))

        self.assertEqual(loaded.name, profile.name)
        self.assertEqual(loaded.start, profile.start)
        self.assertEqual(loaded.goals, profile.goals)
        self.assertEqual(loaded.mapping, profile.mapping)

    def test_profile_rejects_invalid_scale(self):
        profile = programmatic_g1_profile()
        payload = profile.to_dict()
        payload['scene_scale'] = [0.01, 0.0, 0.01]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'bad.json'
            output.write_text(__import__('json').dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'scene_scale'):
                load_point_navigation_profile(str(output))


if __name__ == '__main__':
    unittest.main()
