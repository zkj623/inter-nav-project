"""Output-path helpers shared by navigation entry points."""

from pathlib import Path
from typing import Optional


def resolve_map_output(record_dir: str, requested: Optional[str]) -> str:
    """Keep the final map beside recordings unless explicitly overridden."""

    if requested is not None:
        return str(requested)
    if not record_dir:
        return ''
    return str(Path(record_dir) / 'final_map')


__all__ = ['resolve_map_output']
