"""Start ONE selected model service; --help lists service/model/device options."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from deploy.perception.server import main
    main()
