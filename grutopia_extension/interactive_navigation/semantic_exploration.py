"""Adaptive geometric/semantic frontier exploration.

The module is deliberately independent of Isaac and Transformers at import time.
It accepts either dictionaries or lightweight objects from semantic Voronoi
implementations, which makes it suitable for online use and unit testing.
"""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


Position = Tuple[float, float]


class ExplorationMode(str, Enum):
    GEOMETRIC = 'geometric'
    SEMANTIC = 'semantic'


@dataclass(frozen=True)
class ExplorationConfig:
    semantic_evidence_threshold: float = 0.35
    semantic_weight: float = 0.45
    path_distance_weight: float = 0.45
    information_gain_weight: float = 1.0
    clearance_weight: float = 0.35
    degree_weight: float = 0.20
    extensibility_weight: float = 0.30
    direction_weight: float = 0.35
    visit_penalty: float = 0.50
    failure_penalty: float = 1.5
    dead_end_penalty: float = 2.0
    information_radius: float = 0.8
    clearance_radius: float = 1.0
    reached_tolerance: float = 0.35
    blacklist_failure_threshold: int = 2
    max_no_progress_updates: int = 3
    deadlock_window: int = 3
    position_quantization: float = 0.25
    max_graph_json_bytes: int = 12_000
    qwen_timeout_seconds: float = 8.0

    def __post_init__(self):
        if not 0.0 <= self.semantic_evidence_threshold <= 1.0:
            raise ValueError('semantic_evidence_threshold must be in [0, 1]')
        if self.blacklist_failure_threshold < 1:
            raise ValueError('blacklist_failure_threshold must be positive')
        if self.max_no_progress_updates < 1 or self.deadlock_window < 1:
            raise ValueError('progress limits must be positive')
        if self.position_quantization <= 0 or self.max_graph_json_bytes < 64:
            raise ValueError('serialization limits must be positive')


# A more explicit name is useful to callers and preserves a compact public API.
SemanticExplorationConfig = ExplorationConfig


@dataclass
class FrontierCandidate:
    frontier_id: str
    position: Position
    path_distance: float = math.inf
    information_gain: float = 0.0
    clearance: float = 0.0
    degree: int = 0
    extensibility: float = 0.0
    direction_alignment: float = 0.0
    visit_count: int = 0
    failure_count: int = 0
    dead_end: bool = False
    geometric_score: float = -math.inf
    semantic_score: float = 0.0
    score: float = -math.inf
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def id(self) -> str:
        return self.frontier_id

    @property
    def total_score(self) -> float:
        return self.score


@dataclass(frozen=True)
class ExplorationDecision:
    mode: ExplorationMode
    selected_frontier_id: Optional[str]
    reason: str
    candidates: Tuple[FrontierCandidate, ...] = ()
    used_model: bool = False
    fell_back: bool = False

    @property
    def selected_id(self) -> Optional[str]:
        return self.selected_frontier_id


def serialize_local_semantic_voronoi(
    snapshot: Any,
    frontiers: Iterable[Any] = (),
    max_bytes: int = 12_000,
) -> str:
    """Serialize a bounded, deterministic semantic Voronoi context as JSON.

    Region and doorway summaries are the highest-level hierarchy layer and are
    dropped last when the payload must be truncated to fit ``max_bytes``.
    """
    if max_bytes < 64:
        raise ValueError('max_bytes must be at least 64')
    nodes = [_jsonable_node(item) for item in _items(snapshot, 'nodes')]
    edges = [_jsonable_edge(item) for item in _items(snapshot, 'edges')]
    frontier_rows = []
    for index, item in enumerate(frontiers):
        row = {'id': _frontier_id(item, index), 'position': list(_position(item))}
        region = _value(item, 'region_id', _value(item, 'region', None))
        if region is not None:
            row['region'] = str(region)
        frontier_rows.append(row)
    semantics = [_jsonable_semantic(item) for item in _items(snapshot, 'semantics')]
    regions = [_jsonable_region(item) for item in _items(snapshot, 'regions')]
    doorways = [_jsonable_doorway(item) for item in _items(snapshot, 'doorways')]
    payload = {
        'nodes': nodes,
        'edges': edges,
        'semantics': semantics,
        'regions': regions,
        'doorways': doorways,
        'frontiers': frontier_rows,
        'truncated': False,
    }

    def encode() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'), sort_keys=True)

    result = encode()
    while len(result.encode('utf-8')) > max_bytes and (
        payload['edges'] or payload['nodes'] or payload['semantics'] or payload['doorways'] or payload['regions']
    ):
        payload['truncated'] = True
        if payload['edges']:
            payload['edges'].pop()
        elif payload['nodes']:
            payload['nodes'].pop()
        elif payload['semantics']:
            payload['semantics'].pop()
        elif payload['doorways']:
            payload['doorways'].pop()
        else:
            payload['regions'].pop()
        result = encode()
    while len(result.encode('utf-8')) > max_bytes and payload['frontiers']:
        payload['truncated'] = True
        payload['frontiers'].pop()
        result = encode()
    if len(result.encode('utf-8')) > max_bytes:
        result = json.dumps({'truncated': True}, separators=(',', ':'))
    return result


class Qwen3Scorer:
    """Lazy local Qwen scorer with strict JSON validation and safe fallback."""

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen3-8B',
        device: Optional[str] = None,
        generator: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = 8.0,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.max_new_tokens = max_new_tokens
        self._generator = generator
        self._load_attempted = False
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        # An injected generator is known to be available. A local model remains
        # potentially available until a lazy loading attempt proves otherwise.
        if self._generator is not None:
            return bool(getattr(self._generator, 'available', True))
        return not self._load_attempted

    def score(
        self,
        candidates: Sequence[FrontierCandidate],
        graph_json: str,
        target_query: str = '',
        target_direction: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, float]]:
        if not candidates:
            return {}
        try:
            generator = self._ensure_generator()
            prompt = self._prompt(candidates, graph_json, target_query, target_direction)
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._generate, generator, prompt)
            try:
                output = future.result(timeout=self.timeout_seconds)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            scores = self._parse_scores(output, {item.frontier_id for item in candidates})
            self.last_error = None
            return scores
        except (TimeoutError, Exception) as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            return None

    def _ensure_generator(self):
        if self._generator is not None:
            return self._generator
        self._load_attempted = True
        try:
            from transformers import pipeline

            kwargs: Dict[str, Any] = {'model': self.model_name}
            if self.device is not None:
                kwargs['device'] = self.device
            self._generator = pipeline('text-generation', **kwargs)
            return self._generator
        except Exception:
            self._generator = None
            raise

    def _generate(self, generator, prompt):
        if hasattr(generator, 'generate_prompt'):
            return generator.generate_prompt(prompt, timeout=self.timeout_seconds)
        messages = [
            {
                'role': 'system',
                'content': (
                    'Score exploration frontiers. Return only one JSON object mapping every '
                    'candidate id to a numeric score in [0,1]. No markdown or explanation.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ]
        tokenizer = getattr(generator, 'tokenizer', None)
        if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return generator(
                text,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        try:
            return generator(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        except TypeError:
            return generator(prompt)

    @staticmethod
    def _prompt(candidates, graph_json, target_query, target_direction):
        compact = [
            {
                'id': candidate.frontier_id,
                'position': candidate.position,
                'path_distance': candidate.path_distance,
                'information_gain': candidate.information_gain,
                'clearance': candidate.clearance,
                'degree': candidate.degree,
                'extensibility': candidate.extensibility,
                'direction_alignment': candidate.direction_alignment,
            }
            for candidate in candidates
        ]
        return json.dumps(
            {
                'target_query': str(target_query),
                'target_direction': target_direction,
                'candidates': compact,
                'semantic_voronoi': json.loads(graph_json),
            },
            ensure_ascii=False,
            separators=(',', ':'),
        )

    @staticmethod
    def _parse_scores(output: Any, expected_ids: set[str]) -> Dict[str, float]:
        if isinstance(output, list):
            if len(output) != 1:
                raise ValueError('generator must return one result')
            output = output[0]
        if isinstance(output, Mapping) and 'generated_text' in output:
            output = output['generated_text']
        if isinstance(output, list) and output and isinstance(output[-1], Mapping):
            output = output[-1].get('content')
        if not isinstance(output, str):
            raise ValueError('model output is not text')
        parsed = json.loads(output.strip())
        if not isinstance(parsed, dict):
            raise ValueError('score response must be a JSON object')
        if set(parsed) != expected_ids:
            raise ValueError('score response must contain exactly all candidate ids')
        scores: Dict[str, float] = {}
        for key, value in parsed.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f'invalid score for {key}')
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f'score outside [0,1] for {key}')
            scores[key] = float(value)
        return scores


class Qwen3WorkerGenerator:
    """JSON-lines client for the isolated local Qwen3 worker process."""

    def __init__(
        self,
        python_executable: str,
        worker_script: str,
        model_name: str = 'Qwen/Qwen3-8B',
        device: str = 'cuda:6',
        startup_timeout: float = 180.0,
    ):
        self.device = str(device)
        worker_device = 'cuda:0' if self.device.startswith('cuda:') else self.device
        self.command = [
            str(python_executable),
            str(worker_script),
            '--model',
            str(model_name),
            '--device',
            worker_device,
        ]
        self.startup_timeout = float(startup_timeout)
        self._process = None
        self._stderr = None
        self._lock = threading.Lock()
        self._disabled = False

    def _worker_env(self):
        env = os.environ.copy()
        if self.device.startswith('cuda:') and ':' in self.device:
            env['CUDA_VISIBLE_DEVICES'] = self.device.split(':', 1)[1]
        return env

    @property
    def available(self) -> bool:
        return not self._disabled and (self._process is None or self._process.poll() is None)

    def start(self):
        """Eagerly load the worker so online scoring keeps its short timeout."""

        with self._lock:
            self._ensure_started()

    def _ensure_started(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        stderr_path = os.environ.get(
            'QWEN3_WORKER_LOG',
            '/tmp/qwen3_topology_worker.err',
        )
        self._stderr = open(stderr_path, 'w', encoding='utf-8')
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        try:
            ready = self._read_json(self.startup_timeout)
        except Exception:
            self._disabled = True
            self.close()
            raise
        if ready.get('event') != 'qwen3_ready':
            self._disabled = True
            self.close()
            raise RuntimeError(f'Qwen3 worker did not become ready: {ready}')

    def generate_prompt(self, prompt: str, timeout: float = 8.0) -> str:
        with self._lock:
            self._ensure_started()
            request_id = uuid.uuid4().hex
            request = json.dumps({'request_id': request_id, 'prompt': str(prompt)})
            self._process.stdin.write(request + '\n')
            self._process.stdin.flush()
            response = self._read_json(timeout)
            if response.get('request_id') != request_id:
                raise RuntimeError('Qwen3 worker response id mismatch')
            if response.get('error'):
                raise RuntimeError(str(response['error']))
            return str(response.get('text', ''))

    def _read_json(self, timeout: float) -> dict:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError('Qwen3 worker is not running')
        ready, _, _ = select.select([self._process.stdout], [], [], max(0.0, float(timeout)))
        if not ready:
            raise TimeoutError('Qwen3 worker timed out')
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f'Qwen3 worker exited with code {code}')
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('Qwen3 worker returned non-object JSON')
        return value

    def close(self):
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()
            self._process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def __del__(self):
        self.close()


class AdaptiveExplorationPlanner:
    def __init__(
        self,
        config: ExplorationConfig = ExplorationConfig(),
        scorer: Optional[Qwen3Scorer] = None,
    ):
        self.config = config
        self.scorer = scorer
        self.mode = ExplorationMode.GEOMETRIC
        self.visit_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        self.blacklist: set[str] = set()
        self.dead_ends: set[str] = set()
        self.selection_history: list[str] = []
        self._last_selected: Optional[str] = None
        self._last_distance: Optional[float] = None
        self._no_progress_updates = 0

    def select_goal(
        self,
        robot_position: Sequence[float],
        frontiers: Optional[Iterable[Any]] = None,
        semantic_voronoi: Any = None,
        occupancy: Any = None,
        target_query: str = '',
        target_direction: Optional[Sequence[float]] = None,
        semantic_evidence: Optional[float] = None,
        model_available: Optional[bool] = None,
    ) -> Tuple[ExplorationDecision, Optional[Position]]:
        raw_frontiers = tuple(
            _items(semantic_voronoi, 'frontiers') if frontiers is None else frontiers
        )
        occupancy = _value(occupancy, 'occupancy', occupancy)
        candidates = [
            self._candidate(item, index, robot_position, semantic_voronoi, occupancy, target_direction)
            for index, item in enumerate(raw_frontiers)
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate.frontier_id not in self.blacklist and math.isfinite(candidate.path_distance)
        ]
        evidence = self._semantic_evidence(semantic_voronoi) if semantic_evidence is None else float(semantic_evidence)
        available = self.scorer is not None and self.scorer.available
        if model_available is not None:
            available = available and bool(model_available)
        deadlocked = self._is_deadlocked()
        desired_mode = (
            ExplorationMode.SEMANTIC
            if evidence >= self.config.semantic_evidence_threshold and available and not deadlocked
            else ExplorationMode.GEOMETRIC
        )
        mode_changed = desired_mode != self.mode
        self.mode = desired_mode

        model_scores: Optional[Dict[str, float]] = None
        if candidates and self.mode == ExplorationMode.SEMANTIC and self.scorer is not None:
            graph_json = serialize_local_semantic_voronoi(
                semantic_voronoi,
                raw_frontiers,
                self.config.max_graph_json_bytes,
            )
            model_scores = self.scorer.score(
                candidates,
                graph_json,
                target_query=target_query,
                target_direction=target_direction,
            )
            if model_scores is None:
                self.mode = ExplorationMode.GEOMETRIC

        for candidate in candidates:
            candidate.semantic_score = (model_scores or {}).get(candidate.frontier_id, 0.0)
            candidate.score = candidate.geometric_score
            if model_scores is not None:
                candidate.score += self.config.semantic_weight * candidate.semantic_score
        candidates.sort(key=lambda item: (
            item.failure_count if item.metadata.get('reachable_boundary') else 0,
            -item.score, item.path_distance, item.frontier_id,
        ))
        if not candidates:
            decision = ExplorationDecision(
                mode=self.mode,
                selected_frontier_id=None,
                reason='no_unblacklisted_frontiers',
                candidates=(),
                fell_back=model_scores is None and desired_mode == ExplorationMode.SEMANTIC,
            )
            return decision, None

        selected = candidates[0]
        self._last_selected = selected.frontier_id
        self._last_distance = _distance(robot_position, selected.position)
        self.selection_history.append(selected.frontier_id)
        reason = 'semantic_score' if model_scores is not None else 'geometric_score'
        if deadlocked:
            reason = 'deadlock_geometric_recovery'
        elif mode_changed:
            reason = f'switched_to_{self.mode.value}'
        decision = ExplorationDecision(
            mode=self.mode,
            selected_frontier_id=selected.frontier_id,
            reason=reason,
            candidates=tuple(candidates),
            used_model=model_scores is not None,
            fell_back=model_scores is None and desired_mode == ExplorationMode.SEMANTIC,
        )
        return decision, selected.position

    def update_progress(self, robot_position: Sequence[float]) -> bool:
        """Record motion toward the active frontier; return whether progress occurred."""
        if self._last_selected is None or self._last_distance is None:
            return False
        # The selected position is recovered from the latest decision history key.
        position = self._position_by_key(self._last_selected)
        if position is None:
            return False
        distance = _distance(robot_position, position)
        progressed = distance + 1e-6 < self._last_distance
        self._no_progress_updates = 0 if progressed else self._no_progress_updates + 1
        self._last_distance = distance
        return progressed

    def record_arrival(self, frontier: Any):
        key = self._feedback_key(frontier)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1
        self.failure_counts.pop(key, None)
        self._no_progress_updates = 0

    def record_failure(self, frontier: Any, dead_end: bool = False):
        key = self._feedback_key(frontier)
        count = self.failure_counts.get(key, 0) + 1
        self.failure_counts[key] = count
        if dead_end:
            self.dead_ends.add(key)
        if dead_end or count >= self.config.blacklist_failure_threshold:
            self.blacklist.add(key)
        self._no_progress_updates += 1

    def report_feedback(self, frontier: Any, reached: bool, dead_end: bool = False):
        if reached:
            self.record_arrival(frontier)
        else:
            self.record_failure(frontier, dead_end=dead_end)

    # Common integration names.
    mark_reached = record_arrival
    mark_failed = record_failure

    def clear_blacklist(self):
        self.blacklist.clear()

    def _candidate(self, item, index, robot, snapshot, occupancy, target_direction):
        frontier_id = _frontier_id(item, index)
        position = _position(item)
        distance = _value(item, 'path_distance', None)
        path_distance = self._path_distance(robot, position, occupancy) if distance is None else float(distance)
        information_gain = self._information_gain(position, occupancy, item)
        clearance = self._clearance(position, occupancy, item)
        degree, extensibility = self._topology(position, item, snapshot)
        direction_alignment = _direction_alignment(robot, position, target_direction)
        visits = self.visit_counts.get(frontier_id, 0)
        failures = self.failure_counts.get(frontier_id, 0)
        dead_end = frontier_id in self.dead_ends or bool(_value(item, 'dead_end', False))
        score = (
            self.config.information_gain_weight * information_gain
            + self.config.clearance_weight * clearance
            + self.config.degree_weight * degree
            + self.config.extensibility_weight * extensibility
            + self.config.direction_weight * direction_alignment
            - self.config.path_distance_weight * path_distance
            - self.config.visit_penalty * visits
            - self.config.failure_penalty * failures
            - self.config.dead_end_penalty * float(dead_end)
        )
        reachable_boundary = bool(_value(item, 'reachable_boundary', False))
        if reachable_boundary:
            score = float(_value(item, 'geometric_score', information_gain / (1.0 + path_distance)))
        candidate = FrontierCandidate(
            metadata={'reachable_boundary': reachable_boundary},
            frontier_id=frontier_id,
            position=position,
            path_distance=path_distance,
            information_gain=information_gain,
            clearance=clearance,
            degree=degree,
            extensibility=extensibility,
            direction_alignment=direction_alignment,
            visit_count=visits,
            failure_count=failures,
            dead_end=dead_end,
            geometric_score=score,
            score=score,
        )
        self._known_positions[frontier_id] = position
        return candidate

    @property
    def _known_positions(self):
        if not hasattr(self, '__known_positions'):
            self.__known_positions: Dict[str, Position] = {}
        return self.__known_positions

    def _position_by_key(self, key):
        return self._known_positions.get(key)

    def _feedback_key(self, frontier):
        if isinstance(frontier, str):
            return frontier
        return _frontier_id(frontier, 0)

    def _is_deadlocked(self):
        repeated = (
            len(self.selection_history) >= self.config.deadlock_window
            and len(set(self.selection_history[-self.config.deadlock_window :])) == 1
        )
        return repeated or self._no_progress_updates >= self.config.max_no_progress_updates

    @staticmethod
    def _semantic_evidence(snapshot):
        explicit = _value(snapshot, 'semantic_evidence', None)
        if explicit is not None:
            return max(0.0, min(1.0, float(explicit)))
        semantics = _items(snapshot, 'semantics')
        if semantics:
            attached = sum(_value(item, 'node_id', None) is not None for item in semantics)
            return min(1.0, attached / max(1.0, len(semantics)))
        nodes = _items(snapshot, 'nodes')
        if not nodes:
            return 0.0
        semantic = sum(
            bool(_value(node, 'label', ''))
            and _value(node, 'kind', '') in ('object', 'semantic')
            for node in nodes
        )
        return semantic / len(nodes)

    @staticmethod
    def _path_distance(robot, position, occupancy):
        if occupancy is None:
            return _distance(robot, position)
        planner = getattr(occupancy, 'planner', None)
        if planner is None and hasattr(occupancy, 'world_to_cell'):
            try:
                from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner

                planner = AStarMapPlanner(occupancy)
            except Exception:
                planner = None
        try:
            path = planner.plan(tuple(robot[:2]) + (0.0,), position + (0.0,))
            points = [tuple(robot[:2])] + [tuple(point[:2]) for point in path]
            return sum(_distance(left, right) for left, right in zip(points, points[1:]))
        except Exception:
            return math.inf if planner is not None else _distance(robot, position)

    def _information_gain(self, position, occupancy, item):
        explicit = _value(item, 'information_gain', None)
        if explicit is not None:
            return float(explicit)
        if occupancy is None or not hasattr(occupancy, 'world_to_cell') or not hasattr(occupancy, 'observed'):
            return 0.0
        center = occupancy.world_to_cell(position)
        if center is None:
            return 0.0
        resolution = float(occupancy.config.grid_resolution)
        radius = max(1, int(math.ceil(self.config.information_radius / resolution)))
        rows = slice(max(0, center[0] - radius), min(occupancy.observed.shape[0], center[0] + radius + 1))
        cols = slice(max(0, center[1] - radius), min(occupancy.observed.shape[1], center[1] + radius + 1))
        patch = occupancy.observed[rows, cols]
        return float((~patch).mean()) if patch.size else 0.0

    def _clearance(self, position, occupancy, item):
        explicit = _value(item, 'clearance', None)
        if explicit is not None:
            return float(explicit)
        if occupancy is None or not hasattr(occupancy, 'world_to_cell'):
            return 0.0
        center = occupancy.world_to_cell(position)
        if center is None:
            return 0.0
        blocked = occupancy.inflated_mask()
        cells = np.argwhere(blocked)
        if len(cells) == 0:
            return self.config.clearance_radius
        resolution = float(occupancy.config.grid_resolution)
        distance = float(np.linalg.norm(cells - np.asarray(center), axis=1).min() * resolution)
        return min(distance, self.config.clearance_radius)

    @staticmethod
    def _topology(position, item, snapshot):
        explicit_degree = _value(item, 'degree', None)
        explicit_extension = _value(item, 'extensibility', _value(item, 'extension', None))
        nodes = _items(snapshot, 'nodes')
        edges = _items(snapshot, 'edges')
        node_id = _value(item, 'node_id', _value(item, 'id', None))
        if node_id is None and nodes:
            nearest = min(nodes, key=lambda node: _distance(position, _position(node)))
            node_id = str(_value(nearest, 'node_id', _value(nearest, 'id', '')))
        degree = 0
        extension = 0.0
        for edge in edges:
            source = str(_value(edge, 'source', ''))
            target = str(_value(edge, 'target', ''))
            if str(node_id) in (source, target):
                degree += 1
                extension += float(_value(edge, 'length', _value(edge, 'distance', 1.0)))
        return int(explicit_degree if explicit_degree is not None else degree), float(
            explicit_extension if explicit_extension is not None else extension
        )


def _items(container: Any, name: str) -> list:
    value = _value(container, name, ())
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    return list(value)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if item is None:
        return default
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _position(item: Any) -> Position:
    value = _value(item, 'position', _value(item, 'centroid', item))
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) < 2 or not np.isfinite(array[:2]).all():
        raise ValueError('frontier position must contain two finite coordinates')
    return float(array[0]), float(array[1])


def _frontier_id(item: Any, index: int) -> str:
    explicit = _value(item, 'frontier_id', _value(item, 'id', _value(item, 'node_id', None)))
    if explicit is not None:
        return str(explicit)
    position = _position(item)
    return f'frontier:{position[0]:.3f}:{position[1]:.3f}'


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _direction_alignment(robot, position, target_direction):
    if target_direction is None:
        return 0.0
    target = np.asarray(target_direction, dtype=float).reshape(-1)[:2]
    travel = np.asarray(position, dtype=float) - np.asarray(robot, dtype=float).reshape(-1)[:2]
    denominator = float(np.linalg.norm(target) * np.linalg.norm(travel))
    if denominator <= 1e-9:
        return 0.0
    return float(np.clip(np.dot(target, travel) / denominator, -1.0, 1.0))


def _jsonable_node(item):
    return {
        'id': str(_value(item, 'node_id', _value(item, 'id', ''))),
        'kind': str(_value(item, 'kind', '')),
        'label': str(_value(item, 'label', '')),
        'position': list(_position(item)),
        'observations': int(_value(item, 'observations', 0)),
    }


def _jsonable_edge(item):
    return {
        'source': str(_value(item, 'source', '')),
        'target': str(_value(item, 'target', '')),
        'relation': str(_value(item, 'relation', '')),
        'distance': float(_value(item, 'length', _value(item, 'distance', 0.0))),
    }


def _jsonable_semantic(item):
    return {
        'object_id': str(_value(item, 'object_id', '')),
        'node_id': _value(item, 'node_id', None),
        'label': str(_value(item, 'label', '')),
        'distance': _value(item, 'distance', None),
    }


def _jsonable_region(item):
    labels = [
        [str(label), int(count)]
        for label, count in list(_value(item, 'labels', ()) or ())[:5]
    ]
    return {
        'id': str(_value(item, 'region_id', _value(item, 'id', ''))),
        'area': round(float(_value(item, 'area', 0.0)), 2),
        'node_count': len(_items(item, 'node_ids')),
        'labels': labels,
        'adjacent': [str(value) for value in _items(item, 'adjacent')],
        'frontier_count': int(_value(item, 'frontier_count', 0)),
    }


def _jsonable_doorway(item):
    position = np.asarray(_value(item, 'position', (0.0, 0.0)), dtype=float).reshape(-1)[:2]
    return {
        'id': str(_value(item, 'doorway_id', _value(item, 'id', ''))),
        'position': [round(float(value), 2) for value in position],
        'width': round(float(_value(item, 'width', 0.0)), 2),
        'regions': [str(value) for value in _items(item, 'regions')],
    }


__all__ = [
    'AdaptiveExplorationPlanner',
    'ExplorationConfig',
    'ExplorationDecision',
    'ExplorationMode',
    'FrontierCandidate',
    'Qwen3Scorer',
    'Qwen3WorkerGenerator',
    'SemanticExplorationConfig',
    'serialize_local_semantic_voronoi',
]
