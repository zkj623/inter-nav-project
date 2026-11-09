import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from grutopia_extension.interactive_navigation.semantic_exploration import (
    AdaptiveExplorationPlanner,
    ExplorationConfig,
    ExplorationMode,
    FrontierCandidate,
    Qwen3Scorer,
    Qwen3WorkerGenerator,
    serialize_local_semantic_voronoi,
)


class SemanticExplorationTest(unittest.TestCase):
    def setUp(self):
        self.frontiers = [
            {
                'id': 'short',
                'position': (1.0, 0.0),
                'information_gain': 0.2,
                'clearance': 0.5,
                'degree': 1,
                'extensibility': 0.5,
            },
            {
                'id': 'useful',
                'position': (2.0, 0.0),
                'information_gain': 1.0,
                'clearance': 1.0,
                'degree': 3,
                'extensibility': 2.0,
            },
        ]

    def test_geometric_frontier_ranking_uses_combined_features(self):
        planner = AdaptiveExplorationPlanner()
        decision, goal = planner.select_goal((0.0, 0.0), self.frontiers)

        self.assertEqual(decision.mode, ExplorationMode.GEOMETRIC)
        self.assertEqual(decision.selected_frontier_id, 'useful')
        self.assertEqual(goal, (2.0, 0.0))
        self.assertGreater(decision.candidates[0].score, decision.candidates[1].score)

    def test_semantic_mode_switches_on_evidence_and_model_availability(self):
        scorer = Qwen3Scorer(generator=lambda *_args, **_kwargs: '{"short":1.0,"useful":0.0}')
        planner = AdaptiveExplorationPlanner(
            ExplorationConfig(semantic_weight=5.0, semantic_evidence_threshold=0.3),
            scorer,
        )

        decision, goal = planner.select_goal(
            (0.0, 0.0),
            self.frontiers,
            semantic_voronoi={'semantic_evidence': 0.8},
        )

        self.assertEqual(decision.mode, ExplorationMode.SEMANTIC)
        self.assertTrue(decision.used_model)
        self.assertEqual(goal, (1.0, 0.0))

    def test_repeated_selection_triggers_geometric_deadlock_recovery(self):
        scorer = Qwen3Scorer(generator=lambda *_args, **_kwargs: '{"short":1.0}')
        planner = AdaptiveExplorationPlanner(
            ExplorationConfig(deadlock_window=2, semantic_evidence_threshold=0.1),
            scorer,
        )
        frontier = [self.frontiers[0]]
        planner.select_goal((0.0, 0.0), frontier, {'semantic_evidence': 1.0})
        planner.select_goal((0.0, 0.0), frontier, {'semantic_evidence': 1.0})

        decision, _ = planner.select_goal((0.0, 0.0), frontier, {'semantic_evidence': 1.0})

        self.assertEqual(decision.mode, ExplorationMode.GEOMETRIC)
        self.assertEqual(decision.reason, 'deadlock_geometric_recovery')

    def test_failures_blacklist_frontier_and_arrival_penalizes_revisit(self):
        planner = AdaptiveExplorationPlanner(ExplorationConfig(blacklist_failure_threshold=2))
        planner.record_failure('short')
        planner.record_failure('short')

        decision, goal = planner.select_goal((0.0, 0.0), self.frontiers)

        self.assertIn('short', planner.blacklist)
        self.assertEqual(decision.selected_frontier_id, 'useful')
        self.assertEqual(goal, (2.0, 0.0))
        planner.record_arrival('useful')
        self.assertEqual(planner.visit_counts['useful'], 1)

    def test_qwen_strict_json_and_error_fallback(self):
        candidates = [
            FrontierCandidate('a', (1.0, 0.0)),
            FrontierCandidate('b', (2.0, 0.0)),
        ]
        valid = Qwen3Scorer(generator=lambda *_args, **_kwargs: '{"a":0.25,"b":0.75}')
        invalid = Qwen3Scorer(generator=lambda *_args, **_kwargs: '```json\n{"a":1,"b":0}\n```')

        self.assertEqual(valid.score(candidates, '{}'), {'a': 0.25, 'b': 0.75})
        self.assertIsNone(invalid.score(candidates, '{}'))
        self.assertIsNotNone(invalid.last_error)

    def test_qwen_exception_and_timeout_fall_back_to_heuristic(self):
        def fail(*_args, **_kwargs):
            raise RuntimeError('offline')

        planner = AdaptiveExplorationPlanner(
            ExplorationConfig(semantic_evidence_threshold=0.1),
            Qwen3Scorer(generator=fail),
        )
        decision, goal = planner.select_goal(
            (0.0, 0.0),
            self.frontiers,
            {'semantic_evidence': 1.0},
        )

        self.assertEqual(decision.mode, ExplorationMode.GEOMETRIC)
        self.assertTrue(decision.fell_back)
        self.assertEqual(goal, (2.0, 0.0))

        slow = Qwen3Scorer(
            generator=lambda *_args, **_kwargs: (time.sleep(0.05) or '{"a":1.0}'),
            timeout_seconds=0.001,
        )
        self.assertIsNone(slow.score([FrontierCandidate('a', (0.0, 0.0))], '{}'))

    def test_graph_json_has_hard_size_limit(self):
        snapshot = {
            'nodes': [
                {'id': f'n{i}', 'position': (float(i), 0.0), 'kind': 'object', 'label': 'chair'}
                for i in range(100)
            ],
            'edges': [],
        }
        encoded = serialize_local_semantic_voronoi(snapshot, self.frontiers, max_bytes=256)

        self.assertLessEqual(len(encoded.encode('utf-8')), 256)
        self.assertTrue(json.loads(encoded)['truncated'])

    def test_graph_json_includes_region_hierarchy(self):
        snapshot = {
            'nodes': [{'id': 'n0', 'position': (0.0, 0.0), 'kind': 'junction', 'label': ''}],
            'edges': [],
            'semantics': [],
            'regions': [
                {
                    'region_id': 'region:1:1',
                    'node_ids': ('n0',),
                    'area': 12.345,
                    'labels': (('bed', 2), ('lamp', 1)),
                    'adjacent': ('region:9:9',),
                    'frontier_count': 1,
                },
            ],
            'doorways': [
                {
                    'doorway_id': 'doorway:5:5',
                    'edge_id': 'edge:a:b:5:5',
                    'position': (1.234, 5.678),
                    'width': 0.82,
                    'regions': ('region:1:1', 'region:9:9'),
                },
            ],
        }
        frontiers = [{'frontier_id': 'f0', 'position': (2.0, 0.0), 'region_id': 'region:1:1'}]

        parsed = json.loads(serialize_local_semantic_voronoi(snapshot, frontiers))

        self.assertEqual(parsed['regions'][0]['id'], 'region:1:1')
        self.assertEqual(parsed['regions'][0]['labels'], [['bed', 2], ['lamp', 1]])
        self.assertEqual(parsed['regions'][0]['adjacent'], ['region:9:9'])
        self.assertEqual(parsed['regions'][0]['area'], 12.35)
        self.assertEqual(parsed['doorways'][0]['width'], 0.82)
        self.assertEqual(parsed['doorways'][0]['regions'], ['region:1:1', 'region:9:9'])
        self.assertEqual(parsed['frontiers'][0]['region'], 'region:1:1')

    def test_graph_json_truncation_drops_regions_last(self):
        snapshot = {
            'nodes': [
                {'id': f'n{i}', 'position': (float(i), 0.0), 'kind': 'corridor', 'label': ''}
                for i in range(50)
            ],
            'edges': [
                {'source': f'n{i}', 'target': f'n{i + 1}', 'length': 1.0}
                for i in range(49)
            ],
            'regions': [
                {
                    'region_id': 'region:0:0',
                    'node_ids': tuple(f'n{i}' for i in range(50)),
                    'area': 40.0,
                    'labels': (),
                    'adjacent': (),
                    'frontier_count': 1,
                },
            ],
        }
        encoded = serialize_local_semantic_voronoi(snapshot, self.frontiers, max_bytes=700)
        parsed = json.loads(encoded)

        self.assertLessEqual(len(encoded.encode('utf-8')), 700)
        self.assertTrue(parsed['truncated'])
        self.assertEqual(len(parsed['regions']), 1)
        self.assertEqual(parsed['edges'], [])

    def test_qwen_worker_generator_uses_isolated_json_lines_process(self):
        source = """
import json
import sys
print(json.dumps({"event": "qwen3_ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"request_id": request["request_id"], "text": "{\\"f\\":0.9}"}), flush=True)
"""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'worker.py'
            script.write_text(source, encoding='utf-8')
            worker = Qwen3WorkerGenerator(
                python_executable=sys.executable,
                worker_script=str(script),
                startup_timeout=2.0,
            )
            try:
                worker.start()
                self.assertEqual(worker.generate_prompt('{}', timeout=2.0), '{"f":0.9}')
            finally:
                worker.close()


if __name__ == '__main__':
    unittest.main()
