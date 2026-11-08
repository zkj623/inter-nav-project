"""Resolve external robot assets for the exported snapshot regressions."""
import os
from pathlib import Path


def robot_urdf():
    usd = os.environ.get('R1LITE_USD_PATH')
    if not usd:
        raise RuntimeError('Set R1LITE_USD_PATH or use household_catalog/run_tests.py --robot-usd')
    urdf = Path(usd).with_suffix('.urdf')
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    return urdf
