import json
import tempfile
import unittest
from pathlib import Path

from grutopia.demo.summarize_semantic_runs import collect_runs


class SummarizeSemanticRunsTest(unittest.TestCase):
    def test_completed_and_unfinished_runs_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = root / 'completed'
            incomplete = root / 'interrupted'
            completed.mkdir()
            incomplete.mkdir()
            (completed / 'manifest.json').write_text(json.dumps({
                'run_id': 'completed', 'git': {'commit': 'abc123', 'dirty': False},
                'scene_profile': {'name': 'scene-a'},
                'run_config': {'semantic_detection_mode': 'open_vocab'},
            }))
            (completed / 'run_summary.json').write_text(json.dumps({
                'status': 'succeeded', 'step': 2365, 'target_confirmed': True,
                'perception': {'open_vocabulary_frames': 99},
            }))
            (incomplete / 'run_summary.json').write_text(json.dumps({
                'status': 'running', 'target_query': 'refrigerator',
            }))
            (incomplete / 'progress.json').write_text(json.dumps({'last_step': 1800}))

            rows = {row['run_id']: row for row in collect_runs(root)}
            self.assertEqual(rows['completed']['status'], 'succeeded')
            self.assertEqual(rows['completed']['open_vocabulary_frames'], 99)
            self.assertEqual(rows['interrupted']['status'], 'unfinished_record')
            self.assertEqual(rows['interrupted']['last_recorded_step'], 1800)
            self.assertEqual(rows['interrupted']['last_recorded_source'], 'progress')


if __name__ == '__main__':
    unittest.main()
