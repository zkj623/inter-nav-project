import os
from pathlib import Path


def default_results_root():
    configured = os.environ.get('R1LITE_RSI_RESULTS_ROOT')
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parent / 'outputs'
