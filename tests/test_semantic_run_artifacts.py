import csv
import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from grutopia_extension.interactive_navigation.semantic_run_artifacts import SemanticRunArtifacts


@dataclass
class _Run:
    target_query: str = 'refrigerator'
    record_every: int = 20


@dataclass
class _Config:
    target_query: str = 'refrigerator'
    target_distance: float = 1.1


class _Profile:
    def to_dict(self):
        return {'name': 'test-scene', 'goals': [[10.0, -1.5, 0.5]]}


class _Component:
    def __init__(self):
        self.config = _Config()
        self.target_node = None
        self.target_confirmed = False
        self.current_goal = None
        self.state = 'explore'
        self.mapping = SimpleNamespace(_plan_history=[])
        self.support = (0, 0, 0)

    def _node_label_evidence(self, _node):
        return self.support


class SemanticRunArtifactsTest(unittest.TestCase):
    def test_cue_records_remain_separate_from_confirmed_target(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SemanticRunArtifacts(directory, _Run(), _Profile())
            component = _Component()
            component.target_cue = SimpleNamespace(cue_id='cue:1')
            component.current_goal = (3, 0, .5)
            component.cue_history = [{'step': 24, 'event': 'selected', 'cue_id': 'cue:1'}]
            recorder.observe(24, component, {'position': (0, 0, .5)}, sample=True, heartbeat=True)
            recorder.observe(25, component, {'position': (0, 0, .5)})
            recorder.finish(None, None, None)
            root = Path(directory)
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events], ['target_cue_selected', 'goal_changed'])
            with (root / 'trace.csv').open(newline='') as file:
                row = list(csv.DictReader(file))[0]
            self.assertEqual((row['goal_kind'], row['target_cue_id'], row['target_node_id']), ('target_cue', 'cue:1', ''))
            self.assertEqual(row['target_label_support'], '0')
            self.assertEqual(row['target_confirmed'], 'False')

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'run_summary.json').write_text('{"status":"succeeded"}')
            with self.assertRaises(FileExistsError):
                SemanticRunArtifacts(directory, _Run(), _Profile())

    def test_interrupt_safe_trace_and_target_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SemanticRunArtifacts(directory, _Run(), _Profile())
            component = _Component()
            initial = {'position': (0.0, 0.0, 0.5)}
            recorder.set_runtime_details(component, services={'grounding_dino': {'model': 'test'}})
            recorder.observe(0, component, initial, sample=True, heartbeat=True)

            component.target_node = SimpleNamespace(node_id='object:door:4', label='door')
            component.current_goal = (10.0, -1.5, 0.5)
            component.support = (7, 30, 7)
            recorder.observe(20, component, {'position': (2.0, 0.0, 0.5)}, sample=True)
            component.target_confirmed = True
            component.support = (8, 31, 8)
            component.mapping._plan_history.append({
                'step': 21, 'state': 'navigate_to_goal', 'reason': 'goal_changed',
                'goal': [10.0, -1.5, 0.5], 'path': [[2.0, 0.0, 0.5], [10.0, -1.5, 0.5]],
            })
            final_observation = {
                'position': (2.2, 0.0, 0.5),
                'orientation': (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            }
            recorder.observe(21, component, final_observation)
            recorder.observe_plans(21, component)

            # These records survive before normal shutdown or final_map.save().
            root = Path(directory)
            progress = json.loads((root / 'progress.json').read_text())
            self.assertEqual(progress['last_step'], 0)
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['random_seed'], None)
            self.assertIn('semantic_config', json.loads((root / 'manifest.json').read_text()))
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([event['event'] for event in events], [
                'target_changed', 'goal_changed', 'target_confirmation_changed', 'plan_changed',
            ])
            self.assertEqual(events[2]['matching_labels'], 8)
            self.assertEqual(events[3]['waypoints'], 2)

            recorder.finish(21, component, final_observation)
            self.assertEqual(json.loads((root / 'progress.json').read_text())['last_step'], 21)
            with (root / 'trace.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row['step'] for row in rows], ['0', '20', '21'])
            self.assertEqual(rows[-1]['target_confirmed'], 'True')
            self.assertEqual(rows[-1]['target_label_support'], '8')
            self.assertAlmostEqual(float(rows[-1]['yaw_rad']), math.pi / 2)


if __name__ == '__main__':
    unittest.main()
