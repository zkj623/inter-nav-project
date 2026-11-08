from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from grutopia_extension.interaction_clearance.environment import ClearanceEnvironment
from grutopia_extension.interaction_clearance.primitives import FrozenBaseCollisionChecker, check_configuration
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox
from grutopia_extension.manipulation.r1lite_bimanual_lift_sim import R1LiteBimanualLift


@unittest.skipUnless(importlib.util.find_spec('fcl'), 'requires python-fcl')
class ExternalMarginTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / 'robot.urdf'
        path.write_text('<robot name="fixture"><link name="base_link"><visual><geometry>'
                        '<box size="0.4 0.4 0.4"/></geometry></visual></link></robot>')
        model = RobotCollisionModel(path)
        task = ClearanceTask(base_obstacle_clearance_m=0.08, arm_obstacle_clearance_m=0.08,
                             execution_obstacle_clearance_m=0.03)
        self.environment = SimpleNamespace(task=task, collision_model=model, box=BimanualBox(),
            bimanual=SimpleNamespace(names=[]), robot=SimpleNamespace(dof_names=[], get_joint_positions=lambda: []),
            base=SimpleNamespace(get_world_pose=lambda: (np.array((0.0, 0.0, 0.2)), np.array((1.0, 0.0, 0.0, 0.0)))),
            object=SimpleNamespace(get_world_pose=lambda: (np.array((-1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0, 0.0)))),
            rotation=lambda quaternion: np.eye(3))
        self.wall(0.5)

    def tearDown(self):
        self.directory.cleanup()

    def wall(self, horizontal):
        model = self.environment.collision_model
        self.environment.wall_objects = {'Obstacle': model.fcl.CollisionObject(model.fcl.Box(0.1, 2.0, 1.2),
            model.fcl.Transform(np.eye(3), (horizontal, 0.0, 0.6)))}

    def test_base_clearance_checks_entire_robot(self):
        pose = np.array((0.20, 0.0, 0.0))
        result = FrozenBaseCollisionChecker(self.environment, False).details(pose)
        self.assertAlmostEqual(result['minimum_wall_clearance_m'], 0.05)
        self.assertFalse(result['accepted'])
        self.environment.task = replace(self.environment.task, base_obstacle_clearance_m=0.04)
        self.assertTrue(FrozenBaseCollisionChecker(self.environment, False).free(pose))

    def test_held_object_also_has_eight_centimeter_margin(self):
        self.wall(-1.30)
        result = FrozenBaseCollisionChecker(self.environment, True).details(np.zeros(3))
        self.assertAlmostEqual(result['minimum_wall_clearance_m'], 0.05)
        self.assertFalse(result['accepted'])

    def test_arm_interpolation_uses_the_same_external_margin(self):
        base, box = np.eye(4), np.eye(4)
        base[:3, 3] = (0.20, 0.0, 0.2)
        box[:3, 3] = (-1.0, 0.0, 0.14)
        with self.assertRaisesRegex(RuntimeError, 'PLANNED_COLLISION'):
            check_configuration(self.environment, np.zeros(15), base, box)
        base[0, 3] = 0.10
        report = check_configuration(self.environment, np.zeros(15), base, box)
        self.assertGreater(report['minimum_wall_clearance_m'], 0.08)

    def test_runtime_stops_below_three_centimeters(self):
        environment = ClearanceEnvironment.__new__(ClearanceEnvironment)
        for name, value in vars(self.environment).items():
            setattr(environment, name, value)
        environment.collision_link_names = ['base_link']
        environment.safety_ok = True
        sample = {'link_positions': [[0.219, 0.0, 0.2]], 'link_quaternions_wxyz': [[1.0, 0.0, 0.0, 0.0]],
                  'object_position': [-1.0, 0.0, 0.0], 'object_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0]}
        with patch.object(R1LiteBimanualLift, 'collision_sample', return_value=sample):
            self.assertAlmostEqual(environment.collision_sample()['minimum_wall_clearance_m'], 0.031)
            sample['link_positions'][0][0] = 0.221
            with self.assertRaisesRegex(RuntimeError, 'Wall clearance violation'):
                environment.collision_sample()
        self.assertFalse(environment.safety_ok)


if __name__ == '__main__':
    unittest.main()
