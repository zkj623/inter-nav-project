"""Explicit start headings for new task IDs without importing offline plans."""
import math

LEGACY_HEADINGS = {'T01': 0., 'T02': math.pi, 'T03': math.pi/2, 'T04': 0., 'T05': math.pi}


def task_initial_heading(task):
    if 'initial_heading_rad' in task:
        heading = float(task['initial_heading_rad'])
    elif task['id'] in LEGACY_HEADINGS:
        heading = LEGACY_HEADINGS[task['id']]
    else:
        raise ValueError(f"Task {task['id']} requires initial_heading_rad")
    if not math.isfinite(heading):
        raise ValueError('Initial heading must be finite')
    return heading


def online_task_definition(task):
    result = {key: task[key] for key in ['id', 'start', 'goal']}
    if 'initial_heading_rad' in task:
        result['initial_heading_rad'] = task_initial_heading(task)
    else:
        task_initial_heading(task)
    return result
