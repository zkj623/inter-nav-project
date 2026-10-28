"""Bounded asynchronous reuse of the existing open-vocabulary pipeline."""
import queue
import threading
import time
from dataclasses import asdict, fields
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
    RequestBudget, PerceptionCancelled,
)


class AsyncPerception:
    def __init__(self, config, backend=None, clock=time.monotonic):
        if config['semantic_classifier'] != 'qwen-vl':
            raise ValueError('Initial robot deployment supports remote Qwen-VL classification; no local CLIP loading')
        self.config, self.clock = config, clock
        valid = {f.name for f in fields(OpenVocabularyPerceptionConfig)}
        self.backend = backend or OpenVocabularyPerception(OpenVocabularyPerceptionConfig(
            **{k: v for k, v in config.items() if k in valid}, query_interval=0.0))
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.generation = 0
        self.last_frame = None
        self.frame = None
        self.last_query_status = 'idle'
        self.last_error = None
        self.last_success = None
        self.last_target_cues = []
        self.last_delivery = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def set_frame(self, frame_id, captured_at):
        self.frame = (frame_id, captured_at)

    def reset(self):
        self.generation += 1
        self._clear(self.jobs)
        self._clear(self.results)
        self.frame = None
        self.last_query_status = 'idle'
        self.last_frame = None
        self.last_success = None
        self.last_error = None
        self.last_target_cues = []
        self.last_delivery = None

    @staticmethod
    def _clear(channel):
        while True:
            try:
                channel.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _replace(channel, item):
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(item)
        except queue.Full:
            pass

    def perceive(self, *, rgba, depth, point_image, target, step):
        if self.frame is not None and self.frame[0] != self.last_frame:
            frame_id, captured_at = self.frame
            self.last_frame = frame_id
            self._replace(self.jobs, (self.generation, captured_at, {
                'rgba': rgba.copy(), 'depth': depth.copy(), 'point_image': point_image.copy(),
                'target': target, 'step': int(frame_id),
            }))
        self.last_query_status = 'rate_limited'
        try:
            generation, captured_at, result, error = self.results.get_nowait()
        except queue.Empty:
            return []
        if generation != self.generation:
            return []  # An old task must not overwrite this task's diagnostics.
        if error:
            self.last_error = error  # Preserve HTTP errors even if the request expired.
            return []
        if self.clock()-captured_at >= self.config['result_max_age_s']:
            self.last_error = 'discarded stale perception result'
            return []
        self.last_success = self.clock()
        self.last_error = None
        self.last_query_status = 'ok'
        self.last_delivery = {'captured_at': captured_at, 'received_at': self.last_success,
                              'target': target, 'detections': [asdict(item) for item in result]}
        return result

    def _work(self):
        while not self.closed.is_set():
            try:
                generation, captured_at, arguments = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            budget = RequestBudget(
                deadline=captured_at+self.config['result_max_age_s'],
                cancelled=lambda generation=generation: self.closed.is_set() or generation != self.generation,
                clock=self.clock,
            )
            try:
                budget.check()
                # The shared backend checks before each HTTP request and caps its
                # timeout to the remaining frame budget. Geometry stays captured.
                detections = self.backend.perceive(**arguments, request_budget=budget)
                budget.check()
                result = [item.to_semantic_detection() for item in detections]
                error = None
            except PerceptionCancelled:
                continue
            except Exception as exc:
                result, error = [], f'{type(exc).__name__}: {exc}'
            if generation == self.generation and not self.closed.is_set():
                self._replace(self.results, (generation, captured_at, result, error))

    def close(self):
        self.closed.set()
        self._clear(self.jobs)
        self._clear(self.results)
        self.thread.join(timeout=.2)


class AsyncScorer:
    """Optional Qwen3 ranking; reuse original prompt/parser, fall back to geometry."""
    def __init__(self, config, session=None):
        import requests
        self.config = config
        self.session = session or requests.Session()
        self.jobs = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.cached = None
        self.pending_key = None
        self.last_error = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    @property
    def available(self):
        return True

    def score(self, candidates, graph_json, target_query='', target_direction=None):
        import hashlib
        import json
        from grutopia_extension.interactive_navigation.semantic_exploration import Qwen3Scorer
        key = hashlib.sha256(json.dumps([target_query, graph_json,
                [(item.frontier_id, item.position) for item in candidates], target_direction],
                default=lambda value: value.tolist()).encode()).hexdigest()
        cached = self.cached
        if cached and cached[0] == key and time.monotonic()-cached[1] <= self.config['max_age_s']:
            return cached[2]
        if key != self.pending_key:
            prompt = Qwen3Scorer._prompt(candidates, graph_json, target_query, target_direction)
            self.pending_key = key
            AsyncPerception._replace(self.jobs, (key, time.monotonic(), prompt,
                                                  {item.frontier_id for item in candidates}))
        return None

    def _work(self):
        from grutopia_extension.interactive_navigation.semantic_exploration import Qwen3Scorer
        while not self.closed.is_set():
            try:
                key, captured_at, prompt, ids = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                response = self.session.post(self.config['url'], json={'prompt': prompt},
                                             timeout=self.config['timeout_s'], proxies={'http': '', 'https': ''})
                response.raise_for_status()
                scores = Qwen3Scorer._parse_scores(response.json()['text'], ids)
                self.cached = (key, captured_at, scores)
                self.last_error = None
            except Exception as error:
                self.last_error = str(error)
            finally:
                if self.pending_key == key:
                    self.pending_key = None

    def close(self):
        self.closed.set()
        self.thread.join(.2)
