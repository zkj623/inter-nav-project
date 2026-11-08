"""Run the controller snapshot regressions and catalog tests without a rollout."""
import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from catalog_support import PACKAGE, REPOSITORY

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--collision-deps', type=Path)
parser.add_argument('--result', type=Path)
parser.add_argument('--robot-usd', type=Path, default=os.environ.get('R1LITE_USD_PATH'))
args = parser.parse_args()
if args.robot_usd is None or not args.robot_usd.with_suffix('.urdf').is_file():
    parser.error('Set --robot-usd or R1LITE_USD_PATH to a USD with its sibling URDF and meshes')
os.environ['R1LITE_USD_PATH'] = str(args.robot_usd.resolve())
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / 'tests'), str(PACKAGE)]
if args.collision_deps:
    sys.path.insert(0, str(args.collision_deps))
names = json.loads((PACKAGE / 'r35_test_modules.json').read_text())
names += ['test_geometry_core', 'test_household_catalog_support']
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skipped=len(result.skipped), passed=result.wasSuccessful(), physics_executed=False)
if args.result:
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
raise SystemExit(0 if result.wasSuccessful() else 1)
