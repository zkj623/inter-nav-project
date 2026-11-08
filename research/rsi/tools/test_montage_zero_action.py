import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.rsi.tools.export_verified_cases import archive_no_action_case


class NoActionExportTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'validation' / 'sim_data' / 'case'
        videos = root / 'validation' / 'videos'
        run.mkdir(parents=True)
        videos.mkdir()
        source = run / 'source' / 'runtime.py'
        source.parent.mkdir()
        source.write_text('runtime')
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        (root / 'runtime_manifest.json').write_text(json.dumps({'files': {'runtime.py': checksum}}))
        (run / 'source_manifest.json').write_text(json.dumps({'runtime.py': checksum}))
        (run / 'result.json').write_text(json.dumps({'frames': 0, 'success': False}))
        (run / 'transitions.jsonl').write_text('')
        (run / 'finish_layout.jpg').write_bytes(b'actual_scene_screenshot')
        np.savez(run / 'demonstration.npz', observations=np.zeros((0, 53)), actions=np.zeros((0, 22)))
        row = {'case_id': 'case', 'success': False, 'identity_matches': True, 'reason': 'BASE_PLAN_REJECTED'}
        return run, videos, row

    def test_scene_still_is_not_claimed_as_rollout_or_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, videos, row = self.fixture(Path(temporary))
            self.assertTrue(archive_no_action_case(run, row, videos))
            metadata = json.loads((videos / 'case.no_action.json').read_text())
            self.assertEqual(metadata['presentation'], 'actual_scene_still_no_actions')
            self.assertFalse(metadata['episode_success'])
            self.assertFalse(metadata['video_exported'])
            self.assertFalse(metadata['offline_audit_available'])
            self.assertFalse(metadata['offline_audit_passed'])

    def test_success_cannot_use_scene_still(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, videos, row = self.fixture(Path(temporary))
            row['success'] = True
            with self.assertRaises(ValueError):
                archive_no_action_case(run, row, videos)

    def test_changed_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, videos, row = self.fixture(Path(temporary))
            (run / 'source' / 'runtime.py').write_text('modified')
            with self.assertRaises(ValueError):
                archive_no_action_case(run, row, videos)

    def test_nonzero_rollout_uses_existing_video_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, videos, row = self.fixture(Path(temporary))
            (run / 'result.json').write_text(json.dumps({'frames': 10, 'success': False}))
            self.assertFalse(archive_no_action_case(run, row, videos))
            self.assertFalse((videos / 'case.no_action.json').exists())


if __name__ == '__main__':
    unittest.main()
