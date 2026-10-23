from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from grutopia_extension.interaction_clearance.benchmark import aggregate_results, default_suite, digest, select_best, suite_cases, validate_suite
from grutopia_extension.interaction_clearance.skill_runtime import Outcome, SkillContext, validate_skill_source
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia.demo.evolve_r1lite_clearance_skill import generate_astra, generation_prompt, training_feedback


class LayoutSuiteTests(unittest.TestCase):
    def test_defaults_have_seven_real_layouts_and_disjoint_splits(self):
        suite = validate_suite(default_suite())
        self.assertEqual([len(suite_cases(suite, split)) for split in ('train', 'validation', 'heldout')], [3, 2, 2])
        walls = {digest(item['task']['scene_obstacles']) for item in suite['layouts']}
        self.assertGreaterEqual(len(walls), 5)
        targets = [case['task']['place_xy'][1] for case in suite_cases(suite, 'train')]
        self.assertTrue(min(targets) < 0 < max(targets))

    def test_geometry_leakage_across_splits_is_rejected(self):
        suite = default_suite()
        suite['layouts'][3]['task'] = deepcopy(suite['layouts'][0]['task'])
        suite['layouts'][3]['task']['seed'] += 100
        with self.assertRaises(ValueError):
            validate_suite(suite)

    def test_all_candidates_get_identical_paired_cases(self):
        suite = default_suite()
        suite['repeats'] = 2
        self.assertEqual(suite_cases(suite, 'train'), suite_cases(suite, 'train'))
        seeds = [item['task']['seed'] for item in suite_cases(suite, 'train')]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_static_blockage_cannot_be_a_clearance_task(self):
        data = ClearanceTask().to_dict()
        data['scene_obstacles'][0]['position'] = [0.7, 0.0, 0.4]
        with self.assertRaises(ValueError):
            ClearanceTask.from_dict(data)

    def test_missing_result_disqualifies_candidate(self):
        cases = suite_cases(default_suite(), 'train')
        with self.assertRaises(ValueError):
            aggregate_results(cases, [])

    def test_safety_ranks_ahead_of_raw_success(self):
        cases = suite_cases(default_suite(), 'train')
        safe = [{'case_id': item['case_id'], 'success': index != 0, 'reason': 'DONE',
                 'safety_failure': False, 'simulated_seconds': 40, 'policy_calls': 1} for index, item in enumerate(cases)]
        unsafe = [dict(item, success=True, safety_failure=index == 0) for index, item in enumerate(safe)]
        self.assertGreater(tuple(aggregate_results(cases, safe)['selection_score']),
                           tuple(aggregate_results(cases, unsafe)['selection_score']))

    def test_macro_average_does_not_overweight_large_layout_groups(self):
        cases = [{'case_id': f'case_{index}', 'layout_id': 'large' if index < 3 else 'small'} for index in range(4)]
        results = [{'case_id': item['case_id'], 'success': index < 3, 'reason': 'DONE',
                    'safety_failure': False, 'simulated_seconds': 40} for index, item in enumerate(cases)]
        summary = aggregate_results(cases, results)
        self.assertEqual(summary['macro_success_rate'], 0.5)
        self.assertEqual(summary['worst_layout_success_rate'], 0.0)

    def test_mismatched_runtime_or_split_cannot_be_compared(self):
        evaluation = {'suite_sha256': 'same', 'split': 'train', 'runtime_sha256': 'first',
                      'metrics': {'selection_score': [1, 0, 1, 1, -40, -1]}}
        with self.assertRaises(ValueError):
            select_best([evaluation, dict(evaluation, runtime_sha256='different')])
        with self.assertRaises(ValueError):
            select_best([evaluation, dict(evaluation, split='heldout')])

    def test_generator_receives_no_validation_or_heldout_feedback(self):
        suite = default_suite()
        prompt = generation_prompt(suite, [], 1)
        for layout in suite['layouts']:
            if layout['split'] != 'train':
                self.assertNotIn(layout['layout_id'], prompt)
        candidate = {'candidate_id': 'example', 'source': 'def run(ctx): return ctx.verify_clearance()',
                     'train': {'metrics': {}, 'results': []}, 'validation': {'secret': 'DO_NOT_LEAK'}}
        self.assertNotIn('DO_NOT_LEAK', str(training_feedback([candidate])))


class SkillWaitTests(unittest.TestCase):
    def context(self, wait=None, policies=None):
        return SkillContext(ClearanceTask(), policies or {}, lambda: Outcome(False, 'BLOCKED'), lambda: {},
                            lambda *args: None, wait=wait)

    def test_wait_api_is_allowed_and_budget_is_enforced(self):
        validate_skill_source('def run(ctx):\n outcome = ctx.wait(1.0)\n return outcome')
        context = self.context(wait=lambda seconds: Outcome(True, 'WAIT_COMPLETE'))
        for _ in range(3):
            self.assertTrue(context.wait(2).success)
        self.assertEqual(context.wait(1).reason, 'WAIT_BUDGET_EXCEEDED')
        for value in (-1, 3, float('nan'), '2'):
            self.assertEqual(self.context().wait(value).reason, 'INVALID_WAIT_DURATION')

    def test_failed_policy_prevents_unsafe_wait_and_retry(self):
        class FailedPolicy:
            def execute(self, task):
                return Outcome(False, 'GRASP_LOST')

        context = self.context(wait=lambda seconds: Outcome(True, 'DONE'), policies={'pick_place': FailedPolicy()})
        self.assertEqual(context.execute('pick_place', object_id='box_0', place_xy=(0.65, 0.7)).reason, 'GRASP_LOST')
        self.assertEqual(context.wait(1).reason, 'SAFETY_STOPPED')
        self.assertEqual(context.execute('pick_place', object_id='box_0', place_xy=(0.65, 0.7)).reason, 'UNSAFE_RETRY_REJECTED')

    def test_wait_exceptions_stop_execution(self):
        def broken_wait(seconds):
            raise RuntimeError('collision')

        context = self.context(wait=broken_wait)
        self.assertEqual(context.wait(1).reason, 'WAIT_EXCEPTION')
        self.assertEqual(context.wait(0).reason, 'SAFETY_STOPPED')


class GeneratorContractTests(unittest.TestCase):
    def response(self, source, tool=False):
        payload = {'source': source, 'rationale': 'Generic orchestration; policy remains frozen.'}
        events = [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(payload)}},
                  {'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}}]
        if tool:
            events.insert(0, {'type': 'item.completed', 'item': {'type': 'command_execution'}})
        return subprocess.CompletedProcess([], 0, '\n'.join(json.dumps(event) for event in events), '')

    def test_generation_inherits_config_without_credentials_and_checks_source(self):
        source = 'def run(ctx):\n return ctx.verify_clearance()\n'
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.run', return_value=self.response(source)) as run:
            target = Path(folder) / 'round'
            result = generate_astra('codex', 'gpt-6-astra', Path(folder) / 'schema.json', 'prompt', target, 30)
            self.assertEqual(result['source'], source)
            self.assertEqual((target / 'candidate.py').read_text(), source)
            command = run.call_args.args[0]
            self.assertIn('gpt-6-astra', command)
            self.assertIn('read-only', command)
            self.assertNotIn('--ignore-user-config', command)
            metadata = json.loads((target / 'generation_metadata.json').read_text())
            self.assertFalse(metadata['credential_exported'])

    def test_invalid_source_is_preserved_but_not_uploaded_as_candidate(self):
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.run', return_value=self.response('import os')):
            target = Path(folder) / 'round'
            with self.assertRaises(ValueError):
                generate_astra('codex', 'gpt-6-astra', Path(folder) / 'schema', 'prompt', target, 30)
            self.assertTrue((target / 'generation.json').exists())
            self.assertFalse((target / 'candidate.py').exists())

    def test_unexpected_generator_tool_use_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.run', return_value=self.response('def run(ctx): return ctx.verify_clearance()', True)):
            with self.assertRaises(ValueError):
                generate_astra('codex', 'gpt-6-astra', Path(folder) / 'schema', 'prompt', Path(folder) / 'round', 30)

    def test_source_rejection_is_sent_back_without_heldout_feedback(self):
        prompt = generation_prompt(default_suite(), [], 2, [{'round': 1, 'error': 'Forbidden skill syntax: Import'}])
        self.assertIn('Forbidden skill syntax: Import', prompt)
        self.assertNotIn('right_long', prompt)


if __name__ == '__main__':
    unittest.main()
