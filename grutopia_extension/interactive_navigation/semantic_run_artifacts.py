"""Small, durable records for comparing and diagnosing semantic runs."""

import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _write_json(path: Path, payload: dict):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _git_revision():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True, timeout=3,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=normal'],
            capture_output=True, text=True, check=True, timeout=3,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {'commit': None, 'dirty': None}
    return {'commit': commit, 'dirty': dirty}


class SemanticRunArtifacts:
    """Write metadata once and flush sparse progress/events during a run.

    A hard process kill cannot execute ``finally``. In that case progress.json,
    trace.csv and events.jsonl still describe the completed portion, while the
    initial run_summary.json remains explicitly unfinished.
    """

    TRACE_FIELDS = (
        'step', 'x_m', 'y_m', 'z_m', 'yaw_rad', 'state', 'goal_kind', 'goal_x_m', 'goal_y_m',
        'goal_distance_m', 'target_node_id', 'target_confirmed',
        'target_label_support', 'target_total_support', 'target_repeated_label_support',
        'target_cue_id', 'base_tilt_deg', 'simulation_time', 'policy_inferences',
        'command_vx', 'command_vy', 'command_wz',
        'body_vx', 'body_vy', 'body_wz',
    )

    def __init__(self, record_dir: str, run, profile):
        self.directory = Path(record_dir)
        existing = [
            self.directory / name for name in (
                'manifest.json', 'run_summary.json', 'combined.mp4', 'final_map.json',
            )
            if (self.directory / name).exists()
        ]
        map_output = getattr(run, 'map_output', '')
        if map_output:
            existing.extend(
                path for path in (Path(str(map_output) + '.json'), Path(str(map_output) + '.npz'))
                if path.exists() and path not in existing
            )
        if existing:
            raise FileExistsError(
                f'run output already exists at {existing[0]}; choose a new --record-dir'
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = self.directory.name
        self.started_at = _now()
        self.manifest = {
            'schema_version': 1,
            'run_id': self.run_id,
            'started_at': self.started_at,
            'git': _git_revision(),
            'python_version': sys.version.split()[0],
            'random_seed': None,  # The current demo does not set a simulation seed.
            'run_config': asdict(run),
            'scene_profile': profile.to_dict(),
        }
        _write_json(self.directory / 'manifest.json', self.manifest)
        self._events = (self.directory / 'events.jsonl').open('w', encoding='utf-8', buffering=1)
        self._trace_file = (self.directory / 'trace.csv').open('w', encoding='utf-8', newline='', buffering=1)
        self._trace = csv.DictWriter(self._trace_file, fieldnames=self.TRACE_FIELDS)
        self._trace.writeheader()
        self._previous_node = None
        self._previous_confirmed = False
        self._previous_goal = None
        self._cue_event_count = 0
        self._exploration_event_count = 0
        self._plan_count = 0
        self._planning_failures = 0
        self._voronoi_fallbacks = 0
        self._voronoi_reasons = {}
        self._last_trace_step = None

    def set_runtime_details(self, component, perception=None, services=None, qwen_active=False):
        self.manifest['semantic_config'] = asdict(component.config)
        self.manifest['perception_config'] = (
            None if perception is None else asdict(perception.config)
        )
        self.manifest['perception_services'] = services or {}
        self.manifest['qwen_active'] = bool(qwen_active)
        _write_json(self.directory / 'manifest.json', self.manifest)

    def _emit(self, step: int, event: str, **fields):
        self._events.write(json.dumps({
            'step': step, 'event': event, 'at': _now(), **fields,
        }, ensure_ascii=False) + '\n')

    @staticmethod
    def _snapshot(component, observation):
        position = tuple(float(value) for value in observation['position'][:3])
        orientation = observation.get('orientation')
        yaw = None
        if orientation is not None and len(orientation) == 4:
            w, x, y, z = (float(value) for value in orientation)
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        node = component.target_node
        cue = getattr(component, 'target_cue', None)
        goal = component.current_goal
        matching, total, repeated = (0, 0, 0) if node is None else component._node_label_evidence(node)
        control = observation.get('controllers', {}).get('move_by_speed', {})
        tilt = None
        if orientation is not None and len(orientation) == 4:
            norm = sum(float(value) ** 2 for value in orientation)
            if norm > 0:
                tilt = math.degrees(math.acos(max(-1., min(1., 1 - 2 * (x*x + y*y) / norm))))
        return {
            'base_tilt_deg': tilt,
            'simulation_time': control.get('simulation_time'),
            'policy_inferences': control.get('policy_inferences'),
            'command': control.get('command', (None, None, None)),
            'body_velocity': control.get('body_lin_vel', (None, None, None)),
            'body_angular_velocity': control.get('body_ang_vel', (None, None, None)),
            'position': position,
            'yaw': yaw,
            'node_id': None if node is None else node.node_id,
            'node_label': None if node is None else node.label,
            'confirmed': bool(component.target_confirmed),
            'matching': matching,
            'total': total,
            'repeated': repeated,
            'goal': None if goal is None else tuple(float(value) for value in goal[:3]),
            'goal_kind': ('target' if node is not None else 'target_cue' if cue is not None
                          else 'frontier' if goal is not None else 'none'),
            'cue_id': None if cue is None else cue.cue_id,
            'state': component.state,
        }

    def observe(self, step: int, component, observation, *, sample=False, heartbeat=False):
        state = self._snapshot(component, observation)
        cue_history = getattr(component, 'cue_history', ())
        for transition in cue_history[self._cue_event_count:]:
            self._emit(transition['step'], 'target_cue_' + transition['event'], **{
                key: value for key, value in transition.items() if key not in ('step', 'event')
            })
        self._cue_event_count = len(cue_history)
        history = getattr(component, 'frontier_history', ())
        for event in history[self._exploration_event_count:]:
            self._emit(step, 'frontier_execution', exploration_event=event)
        self._exploration_event_count = len(history)

        node = state['node_id']
        target_fields = {
            'node_id': node,
            'label': state['node_label'],
            'matching_labels': state['matching'],
            'total_labels': state['total'],
            'repeated_label': state['repeated'],
        }
        if node != self._previous_node:
            self._emit(step, 'target_changed', previous_node_id=self._previous_node, **target_fields)
            self._previous_node = node
        if state['confirmed'] != self._previous_confirmed:
            self._emit(step, 'target_confirmation_changed', confirmed=state['confirmed'], **target_fields)
            self._previous_confirmed = state['confirmed']
        # Millimetre precision prevents immaterial floating-point changes from
        # filling the event log with duplicate goal updates.
        goal_key = None if state['goal'] is None else tuple(round(value, 3) for value in state['goal'])
        if goal_key != self._previous_goal:
            self._emit(step, 'goal_changed', goal_kind=state['goal_kind'], goal=state['goal'])
            self._previous_goal = goal_key
        topology = getattr(component.mapping, 'semantic_voronoi', None)
        counters = None if topology is None else getattr(topology, '_stats', None)
        if counters is not None and counters.incremental_fallbacks > self._voronoi_fallbacks:
            reasons = {
                key: count - self._voronoi_reasons.get(key, 0)
                for key, count in counters.incremental_fallback_reasons.items()
                if count > self._voronoi_reasons.get(key, 0)
            }
            details = topology.statistics()
            self._emit(
                step, 'voronoi_fallback',
                count=counters.incremental_fallbacks - self._voronoi_fallbacks,
                reasons=reasons,
                changed_cells=details.get('last_changed_cells'),
                window_cells=details.get('last_window_cells'),
            )
            self._voronoi_fallbacks = counters.incremental_fallbacks
            self._voronoi_reasons = dict(counters.incremental_fallback_reasons)
        if sample:
            self._write_trace(step, state)
        if heartbeat:
            _write_json(self.directory / 'progress.json', {
                'schema_version': 1,
                'run_id': self.run_id,
                'updated_at': _now(),
                'last_step': step,
                'state': state['state'],
                'position': state['position'],
                'target_node_id': node,
                'target_confirmed': state['confirmed'],
                'target_cue_id': state['cue_id'],
            })

    def _write_trace(self, step, state):
        if step == self._last_trace_step:
            return
        position, goal = state['position'], state['goal']
        distance = None if goal is None else math.dist(position[:2], goal[:2])
        self._trace.writerow({
            'step': step,
            'x_m': position[0], 'y_m': position[1], 'z_m': position[2],
            'yaw_rad': state['yaw'],
            'state': state['state'], 'goal_kind': state['goal_kind'],
            'goal_x_m': None if goal is None else goal[0],
            'goal_y_m': None if goal is None else goal[1],
            'goal_distance_m': None if distance is None else round(distance, 4),
            'target_node_id': state['node_id'],
            'target_confirmed': state['confirmed'],
            'target_label_support': state['matching'],
            'target_total_support': state['total'],
            'target_repeated_label_support': state['repeated'],
            'target_cue_id': state['cue_id'],
            'base_tilt_deg': state['base_tilt_deg'],
            'simulation_time': state['simulation_time'],
            'policy_inferences': state['policy_inferences'],
            'command_vx': state['command'][0],
            'command_vy': state['command'][1],
            'command_wz': state['command'][2],
            'body_vx': state['body_velocity'][0],
            'body_vy': state['body_velocity'][1],
            'body_wz': state['body_angular_velocity'][2],
        })
        self._trace_file.flush()
        self._last_trace_step = step

    def observe_plans(self, step, component):
        history = component.mapping._plan_history
        for plan in history[self._plan_count:]:
            self._emit(
                plan['step'], 'plan_changed',
                reason=plan.get('reason', plan.get('source')),
                state=plan.get('state'), goal=plan.get('goal'),
                waypoints=len(plan.get('path', ())),
            )
        self._plan_count = len(history)
        failures = getattr(component.mapping, 'planning_failures', 0)
        if failures > self._planning_failures:
            self._emit(step, 'planning_failure', count=failures - self._planning_failures)
        self._planning_failures = failures

    def finish(self, step, component, observation):
        if step is not None and component is not None and observation is not None:
            self.observe(step, component, observation, sample=True, heartbeat=True)
            self.observe_plans(step, component)
        self._events.close()
        self._trace_file.close()
