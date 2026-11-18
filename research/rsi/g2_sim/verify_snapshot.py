"""CPU-only checks for the published snapshot and its result accounting."""
import ast
import json
from pathlib import Path
from collections import Counter
from workspace import verify_snapshot

root = Path(__file__).resolve().parent
manifest = verify_snapshot()
for path in (root / 'snapshot').glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
summary = json.loads((root / 'fixtures/strategy15_summary.json').read_text())
assert len(summary['objects']) == 15
assert sum(r['trial_count'] for r in summary['objects']) == 63
assert sum(r['invalid_or_cancelled_count'] for r in summary['objects']) == 1
assert Counter(r['status'] for r in summary['objects']) == {'repeat_pass': 7, 'tested_not_passed': 6, 'no_admissible_plan': 2}
baseline = json.loads((root / 'baselines.json').read_text())
for row in summary['objects']:
    if row['status'] != 'repeat_pass':
        assert row['asset'] not in baseline
        continue
    config = baseline[row['asset']]
    trials = [t for t in row['trials'] if t['success'] and t['candidate'] == config['candidate'] and Path(t['plan_path']).parent.name == config['plan_dir']]
    assert {t['repeat'] for t in trials} == {1, 2, 3}
    for trial in trials:
        result = json.loads((root / 'results' / trial['variant'] / row['asset'] / Path(trial['path']).parent.name / 'result.json').read_text())
        assert result['success'] and not result['object_attachment'] and not result['robot_fixed_to_world']
        assert result['min_hold_lift_m'] >= .15 and result['max_hold_tilt_deg'] <= 20
print(f'Checked {len(manifest)} snapshot files, 15 objects, 63 valid trials and 7 three-repeat baselines')
