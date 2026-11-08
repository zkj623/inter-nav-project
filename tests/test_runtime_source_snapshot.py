import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grutopia.demo.r1lite_interaction_clearance import snapshot_runtime_sources


class RuntimeSourceSnapshotTests(unittest.TestCase):
    def test_new_manifest_dependency_is_automatically_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / 'runtime', Path(temporary) / 'episode'
            dependency = 'grutopia_extension/manipulation/physics_step.py'
            source = root / dependency
            source.parent.mkdir(parents=True)
            source.write_text('native_clock = True\n')
            expected = {dependency: hashlib.sha256(source.read_bytes()).hexdigest()}
            with patch('grutopia_extension.interaction_clearance.benchmark.runtime_fingerprint', return_value={'files': expected}):
                snapshot = snapshot_runtime_sources(root, output)
            self.assertEqual(snapshot, expected)
            self.assertEqual((output / 'source' / dependency).read_bytes(), source.read_bytes())
            self.assertEqual(json.loads((output / 'source_manifest.json').read_text()), expected)

    def test_changed_dependency_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / 'runtime', Path(temporary) / 'episode'
            root.mkdir()
            (root / 'dependency.py').write_text('changed\n')
            with patch('grutopia_extension.interaction_clearance.benchmark.runtime_fingerprint',
                       return_value={'files': {'dependency.py': 'incorrect_digest'}}):
                with self.assertRaises(RuntimeError):
                    snapshot_runtime_sources(root, output)
            self.assertFalse((output / 'source_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
