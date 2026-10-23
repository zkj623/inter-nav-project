from pathlib import Path
import unittest

import numpy as np

from grutopia_extension.interaction_clearance.skill_runtime import Outcome, SkillContext, load_skill, validate_skill_source
from grutopia_extension.interaction_clearance.task import ClearanceTask, evaluate_clearance


class ClearanceTaskTests(unittest.TestCase):
    def test_default_task_blocks_the_passage(self):
        task = ClearanceTask()
        task.validate()
        result = evaluate_clearance(task, (*task.object_xy, 0), np.eye(3), 0, 0, 0, True, 2)
        self.assertFalse(result['road_clear'])
        self.assertFalse(result['success'])

    def test_clearance_requires_supported_released_stable_object(self):
        task = ClearanceTask()
        result = evaluate_clearance(task, (*task.place_xy, 0.0001), np.eye(3), 0, 0, 0, True, 1.5)
        self.assertTrue(result['success'])
        for bottom, force, stable, safe in ((0.2, 0, 2, True), (0, 10, 2, True), (0, 0, 0.5, True), (0, 0, 2, False)):
            result = evaluate_clearance(task, (*task.place_xy, bottom), np.eye(3), 0, 0, force, safe, stable)
            self.assertFalse(result['success'])

    def test_task_round_trip(self):
        task = ClearanceTask.from_dict(ClearanceTask().to_dict())
        self.assertEqual(task.object_xy, (0.65, 0.0))
        self.assertEqual(task.placement.lower, (0.35, 0.40))

    def test_already_clear_upstream_task_is_rejected(self):
        with self.assertRaises(ValueError):
            ClearanceTask(object_xy=(0.65, 0.70)).validate()

    def test_moving_object_is_not_complete(self):
        result = evaluate_clearance(ClearanceTask(), (0.65, 0.70, 0), np.eye(3), 0.1, 0, 0, True, 2)
        self.assertFalse(result['success'])

    def test_rotated_footprint_can_violate_clearance(self):
        cosine = np.cos(np.pi / 4)
        sine = np.sin(np.pi / 4)
        rotated = np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
        task = ClearanceTask()
        unrotated = evaluate_clearance(task, (0.65, 0.68, 0), np.eye(3), 0, 0, 0, True, 2)
        result = evaluate_clearance(task, (0.65, 0.68, 0), rotated, 0, 0, 0, True, 2)
        self.assertTrue(unrotated['road_clear'])
        self.assertFalse(result['road_clear'])


class SkillBoundaryTests(unittest.TestCase):
    def test_unsafe_skill_code_is_rejected(self):
        sources = ('import os\ndef run(ctx): return True',
                   'def run(ctx):\n while True: pass',
                   'def run(ctx): return ctx._policies',
                   'def run(ctx): return open("data")',
                   'def run(ctx):\n ctx.task.mass_kg = 0',
                   'def run(ctx):\n for ctx.task in range(1): pass',
                   'def run(ctx):\n for ctx.observe()["position"] in range(1): pass',
                   'def run(ctx):\n for index in range(100): pass')
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                validate_skill_source(source)

    def test_default_skill_calls_policy_then_independent_verification(self):
        calls = []

        class FakePolicy:
            def execute(self, task):
                calls.append('policy')
                return Outcome(True, 'DONE')

        def verify():
            calls.append('verify')
            return Outcome(True, 'ROAD_CLEARED')

        task = ClearanceTask()
        context = SkillContext(task, {'pick_place': FakePolicy()}, verify, lambda: {}, lambda *args: None)
        skill_path = Path(__file__).parents[1] / 'grutopia_extension/interaction_clearance/skills/clear_blockage.py'
        skill, digest = load_skill(skill_path)
        self.assertTrue(skill(context).success)
        self.assertEqual(calls, ['policy', 'verify'])
        self.assertEqual(len(digest), 64)

    def test_skill_cannot_override_upstream_target(self):
        context = SkillContext(ClearanceTask(), {'pick_place': None}, lambda: None, lambda: {}, lambda *args: None)
        outcome = context.execute('pick_place', object_id='other', place_xy=(0.65, 0.70))
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.reason, 'UPSTREAM_TARGET_MISMATCH')

    def test_unknown_policy_is_structured_failure(self):
        context = SkillContext(ClearanceTask(), {}, lambda: None, lambda: {}, lambda *args: None)
        outcome = context.execute('push', object_id='box_0', place_xy=(0.65, 0.70))
        self.assertEqual(outcome.reason, 'UNKNOWN_POLICY')

    def test_failure_does_not_request_success_verification(self):
        class FailedPolicy:
            def execute(self, task):
                return Outcome(False, 'GRASP_LOST')

        def unexpected_verify():
            raise AssertionError('Failure must be returned to upstream')

        context = SkillContext(ClearanceTask(), {'pick_place': FailedPolicy()}, unexpected_verify,
                               lambda: {}, lambda *args: None)
        path = Path(__file__).parents[1] / 'grutopia_extension/interaction_clearance/skills/clear_blockage.py'
        skill, _ = load_skill(path)
        self.assertEqual(skill(context).reason, 'GRASP_LOST')

    def test_policy_exceptions_produce_structured_feedback(self):
        events = []

        class BrokenPolicy:
            def execute(self, task):
                raise ValueError('unreachable target')

        context = SkillContext(ClearanceTask(), {'pick_place': BrokenPolicy()}, lambda: None, lambda: {},
                               lambda kind, details: events.append((kind, details)))
        outcome = context.execute('pick_place', object_id='box_0', place_xy=(0.65, 0.70))
        self.assertEqual(outcome.reason, 'POLICY_EXCEPTION')
        self.assertEqual(events[-1][0], 'policy_end')

    def test_policy_call_budget_is_enforced(self):
        class CompletedPolicy:
            def execute(self, task):
                return Outcome(True, 'DONE')

        context = SkillContext(ClearanceTask(), {'pick_place': CompletedPolicy()}, lambda: None, lambda: {},
                               lambda *args: None, maximum_calls=1)
        self.assertTrue(context.execute('pick_place', object_id='box_0', place_xy=(0.65, 0.70)).success)
        self.assertEqual(context.execute('pick_place', object_id='box_0', place_xy=(0.65, 0.70)).reason,
                         'POLICY_CALL_BUDGET_EXCEEDED')


if __name__ == '__main__':
    unittest.main()
