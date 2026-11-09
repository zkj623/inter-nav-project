import importlib
from typing import Iterable, Optional

EXTENSION_GROUPS = (
    'agents',
    'controllers',
    'interactions',
    'metrics',
    'objects',
    'robots',
    'sensors',
    'tasks',
)


def import_extensions(groups: Optional[Iterable[str]] = None):
    """Register all extension groups, or only an explicitly requested subset."""

    selected_groups = EXTENSION_GROUPS if groups is None else tuple(groups)
    unknown_groups = set(selected_groups) - set(EXTENSION_GROUPS)
    if unknown_groups:
        raise ValueError(f'unknown extension groups: {sorted(unknown_groups)}')
    for group in selected_groups:
        importlib.import_module(f'grutopia_extension.{group}')
