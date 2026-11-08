"""Portable catalog configuration and dependency verification; no simulator imports."""
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[2]
DEFAULT_CATALOG = Path(os.environ.get('R1LITE_CATALOG_ROOT', REPOSITORY / 'assets/household_catalog20'))


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Catalog path escapes its root: {relative}')
    return path


def verify_catalog(root, expected=None):
    root = Path(root).resolve()
    if expected is None:
        expected = json.loads((PACKAGE / 'expected_catalog.json').read_text(encoding='utf-8'))
    actual = json.loads((root / 'asset_manifest.json').read_text(encoding='utf-8'))
    wanted = {a['id']: a for a in expected['assets']}
    assets = {a['id']: a for a in actual['assets']}
    if set(assets) != set(wanted) or len(assets) != len(actual['assets']):
        raise ValueError('Catalog IDs differ from the selected assets or contain duplicates')
    dependencies = {}
    for aid, asset in assets.items():
        if asset['source_sha256'] != wanted[aid]['source_sha256']:
            raise ValueError(f'Source revision differs for {aid}')
        if not safe_child(root, asset['usd']).is_file():
            raise FileNotFoundError(f'Missing USD entry for {aid}')
        for dep in wanted[aid]['dependencies']:
            dependencies[dep['path']] = dep['sha256']
    for name, checksum in dependencies.items():
        path = safe_child(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Catalog dependency checksum mismatch: {name}')
    return dict(passed=True, assets=len(assets), dependencies=len(dependencies))


def input_fingerprint(runtime, catalog, robot_usd):
    files = {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() for name in (
        'lift_probe.py', 'cooked_geometry.py', 'geometry_adapter.py', 'bake_collision_meshes.py',
        'load_asset.py', 'catalog_support.py')}
    files['catalog_manifest'] = hashlib.sha256((Path(catalog) / 'asset_manifest.json').read_bytes()).hexdigest()
    runtime_manifest = json.loads((PACKAGE / 'r35_runtime_manifest.json').read_text())['files']
    for name in runtime_manifest:
        files['runtime:' + name] = hashlib.sha256((Path(runtime) / name).read_bytes()).hexdigest()
    for path in (Path(robot_usd), Path(robot_usd).with_suffix('.urdf')):
        files['robot:' + path.suffix] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify the selected catalog and every source dependency without starting Isaac Sim')
    parser.add_argument('--root', type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    print(json.dumps(verify_catalog(args.root)))
