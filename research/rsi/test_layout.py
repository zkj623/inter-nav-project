import os
from pathlib import Path
import unittest
from unittest.mock import patch

from grutopia.demo import evolve_r1lite_clearance_skill as legacy
from research.rsi.evolution import skill_evolution
from research.rsi import paths


class ResearchLayoutTests(unittest.TestCase):
    def test_legacy_entry_uses_same_implementation(self):
        self.assertIs(legacy.main, skill_evolution.main)
        self.assertIs(legacy.generation_prompt, skill_evolution.generation_prompt)
        self.assertIs(legacy.generate_astra, skill_evolution.generate_astra)

    def test_configured_results_live_outside_source_tree(self):
        with patch.dict(os.environ, {'R1LITE_RSI_RESULTS_ROOT': '/external/experiments'}):
            self.assertEqual(paths.default_results_root(), Path('/external/experiments'))

    def test_default_results_directory_is_gitignored(self):
        with patch.dict(os.environ, {}, clear=True):
            root = paths.default_results_root()
            self.assertEqual(root, Path(paths.__file__).resolve().parent / 'outputs')
            self.assertIn('/outputs/', (root.parent / '.gitignore').read_text())


if __name__ == '__main__':
    unittest.main()
