import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from research.rsi.household_catalog.catalog_support import safe_child, verify_catalog


class CatalogSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'model.usda').write_text('#usda 1.0')
        (self.root / 'source.usd').write_bytes(b'fixture-source')
        self.sha = hashlib.sha256(b'fixture-source').hexdigest()
        self.asset = dict(id='fixture', usd='model.usda', source_sha256=self.sha)
        self.expected = dict(assets=[dict(id='fixture', source_sha256=self.sha,
            dependencies=[dict(path='source.usd', sha256=self.sha)])])
        self.write([self.asset])

    def write(self, assets):
        (self.root / 'asset_manifest.json').write_text(json.dumps(dict(assets=assets)))

    def test_complete_catalog(self):
        self.assertEqual(verify_catalog(self.root, self.expected), dict(passed=True, assets=1, dependencies=1))

    def test_missing_or_changed_dependency_is_rejected(self):
        (self.root / 'source.usd').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verify_catalog(self.root, self.expected)
        (self.root / 'source.usd').unlink()
        with self.assertRaises(FileNotFoundError):
            verify_catalog(self.root, self.expected)

    def test_wrong_model_or_duplicate_id_is_rejected(self):
        self.write([self.asset, self.asset])
        with self.assertRaises(ValueError):
            verify_catalog(self.root, self.expected)
        self.write([dict(self.asset, source_sha256='0' * 64)])
        with self.assertRaises(ValueError):
            verify_catalog(self.root, self.expected)

    def test_paths_cannot_escape_catalog(self):
        with self.assertRaises(ValueError):
            safe_child(self.root, '../outside.usd')
        self.write([dict(self.asset, usd='../outside.usd')])
        with self.assertRaises(ValueError):
            verify_catalog(self.root, self.expected)

if __name__ == '__main__':
    unittest.main()
