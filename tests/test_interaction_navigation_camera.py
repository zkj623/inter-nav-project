import unittest
from pathlib import Path

import numpy as np

from grutopia_extension.interactive_navigation import (
    G1_FIRST_PERSON_CLIPPING_RANGE,
    G1_FIRST_PERSON_OFFSET,
    G1_FIRST_PERSON_PITCH_DEGREES,
)
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    load_point_navigation_profile,
)
from grutopia_extension.interactive_navigation.visualization import (
    following_camera_world_pose,
)


TABLE_FRONT_X = 5.800057285961181
STANDOFF = (5.68, -2.607, 0.8)
BOWL = (5.9960737228393555, -2.8858540058135986, 0.7983754873275757)


class FirstPersonCameraPlacementTest(unittest.TestCase):
    def test_previous_forward_mount_crossed_the_table_front(self):
        position, _ = following_camera_world_pose(
            STANDOFF,
            yaw=0.0,
            offset=(0.14, 0.0, 0.48),
            pitch_degrees=3.0,
        )

        self.assertGreater(position[0], TABLE_FRONT_X)

    def test_g1_first_person_camera_looks_forward(self):
        position, orientation = following_camera_world_pose(
            STANDOFF,
            yaw=0.0,
            offset=G1_FIRST_PERSON_OFFSET,
            pitch_degrees=G1_FIRST_PERSON_PITCH_DEGREES,
        )
        look = _forward_axis(orientation)

        self.assertEqual(tuple(G1_FIRST_PERSON_OFFSET), (0.14, 0.0, 0.48))
        self.assertEqual(G1_FIRST_PERSON_PITCH_DEGREES, 3.0)
        self.assertGreater(position[2], 1.20)
        self.assertGreater(look[0], 0.95)
        self.assertLess(abs(look[2]), 0.10)

    def test_near_clip_is_below_the_table_standoff(self):
        near, far = G1_FIRST_PERSON_CLIPPING_RANGE

        self.assertLess(near, 0.05)
        self.assertLess(near, TABLE_FRONT_X - STANDOFF[0])
        self.assertGreater(far, 10.0)


class ThirdPersonCameraPlacementTest(unittest.TestCase):
    def test_g1_third_person_is_close_and_elevated(self):
        profile = load_point_navigation_profile(
            str(
                Path(__file__).resolve().parents[1]
                / 'grutopia'
                / 'demo'
                / 'profiles'
                / 'g1_grscene_mv7_navigation.json'
            )
        )
        offset = profile.third_person_offset
        position, orientation = following_camera_world_pose(
            STANDOFF,
            yaw=0.0,
            offset=offset,
            pitch_degrees=profile.third_person_pitch_degrees,
        )
        look = _forward_axis(orientation)

        to_robot = np.asarray(STANDOFF, dtype=np.float64) - position
        to_robot /= np.linalg.norm(to_robot)
        robot_in_view_degrees = np.degrees(
            np.arccos(np.clip(np.dot(look, to_robot), -1.0, 1.0))
        )

        self.assertGreater(offset[0], -1.0)
        self.assertLess(offset[0], 0.0)
        self.assertGreater(offset[2], 1.4)
        self.assertGreater(profile.third_person_pitch_degrees, 40.0)
        self.assertGreater(position[0], STANDOFF[0] - 1.0)
        self.assertGreater(position[2], 2.2)
        self.assertLess(position[2], 2.55)
        self.assertLess(look[2], -0.70)
        self.assertLess(robot_in_view_degrees, 18.0)


def _forward_axis(orientation):
    w, x, y, z = orientation
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )


if __name__ == '__main__':
    unittest.main()
