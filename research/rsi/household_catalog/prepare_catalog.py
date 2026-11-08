"""Prepare the selected 20 assets from an existing GRScenes installation."""
import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from catalog_support import PACKAGE, verify_catalog

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True, help='Dataset installation containing scenes/GRScenes-100 and benchmark/meta')
parser.add_argument('--output', type=Path, required=True, help='New output directory; an existing directory is never overwritten')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
if args.output.exists():
    parser.error('Choose a new output directory; existing catalogs are not overwritten')
if not (args.source / 'scenes/GRScenes-100/home_scenes/models/object').is_dir():
    parser.error('Source must contain scenes/GRScenes-100/home_scenes/models/object')
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': args.gpu, 'multi_gpu': False})
exit_status = 0
try:
    import omni.physx
    steps = []
    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(lambda dt: steps.append(float(dt)))
    spec = importlib.util.spec_from_file_location('catalog_importer', PACKAGE / 'import_grscenes.py')
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    sys.argv = ['import_grscenes.py', '--source', str(args.source), '--selection', str(PACKAGE / 'selection.json'), '--output', str(args.output)]
    importer.main()
    result = verify_catalog(args.output)
    result['physics_steps'] = len(steps)
    assert not steps, 'Catalog preparation unexpectedly advanced physics'
    (args.output / 'prepare_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
except Exception as error:
    exit_status = 1
    detail = dict(passed=False, error=str(error), traceback=traceback.format_exc())
    if args.output.exists():
        (args.output / 'prepare_error.json').write_text(json.dumps(detail, indent=2))
    print(json.dumps(detail), file=sys.stderr, flush=True)
finally:
    app.close()
raise SystemExit(exit_status)
