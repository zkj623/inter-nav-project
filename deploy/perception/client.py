"""Bounded RGB-D interaction recognition using the release catalog and protocol.

Geometry is a padded AABB of segmented *visible* metric returns, not hidden
surface reconstruction. geometry_verified defaults to False: only separately
accepted complete conservative envelopes may enter the interaction planner.
No scene IDs, asset dimensions, detector labels or estimated mass grant access.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
import base64
import math
import queue
import tempfile
import threading
import time

import cv2
import numpy as np

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.interaction_catalog import (
    InteractionCatalog, RecognitionEvidence, RecognitionStore,
)
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend, OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
)
from grutopia_extension.interactive_navigation.rgb_object_recognition import (
    context_crop, emphasize_instance,
)


class PerceptionCancelled(RuntimeError):
    pass


class PerceptionDeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class RequestBudget:
    deadline: float
    cancelled: object
    clock: object = time.monotonic

    def remaining(self):
        if self.cancelled():
            raise PerceptionCancelled('interaction perception cancelled')
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise PerceptionDeadlineExceeded('interaction perception capture budget expired')
        return remaining


class _BudgetSession:
    """Limit every existing service request by the remaining capture budget."""
    def __init__(self, session, budget):
        self.session, self.budget = session, budget

    def post(self, url, **kwargs):
        kwargs['timeout'] = min(float(kwargs['timeout']), self.budget.remaining())
        response = self.session.post(url, **kwargs)
        self.budget.remaining()
        return response


@dataclass(frozen=True)
class _Frame:
    generation: int
    frame_id: str
    captured_at: float
    planning_frame: str
    rgba: np.ndarray
    depth: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class _Detection:
    center: tuple
    dimensions: tuple
    category: str
    score: float
    model: str
    image_path: str
    crop_jpeg_base64: str
    raw_response: str


class AsyncInteractionPerception:
    """update(snapshot, now) returns (tuple[BoxState], RecognitionStore).

    Live proposals require fresh capture geometry and geometry_verified. Their
    presence never grants push permission: the returned RecognitionStore and
    catalog require two distinct RGB votes. Unverified bounds remain diagnostic.
    One worker and one replaceable pending frame bound inference concurrency.
    """
    def __init__(self, config, backend=None, clock=time.monotonic, catalog=None):
        self.config, self.clock = dict(config), clock
        if self.config.get('semantic_classifier', 'qwen-vl') != 'qwen-vl':
            raise ValueError('Robot interaction requires remote Qwen-VL classification')
        self.catalog = catalog or InteractionCatalog()
        self.result_max_age = self._positive('result_max_age_s', 8.)
        self.input_max_age = self._positive('input_max_age_s', .5)
        self.geometry_max_age = self._positive('geometry_max_age_s', .75)
        self.recognition_max_age = self._positive('recognition_max_age_s', 10.)
        self.interval = float(self.config.get('query_interval_s', .25))
        if not math.isfinite(self.interval) or self.interval < 0:
            raise ValueError('query_interval_s must be finite and nonnegative')
        self.min_score = float(self.config.get('confidence_threshold', .30))
        if not .30 <= self.min_score <= 1.:
            raise ValueError('confidence_threshold must be between .30 and 1')
        self.min_points = int(self.config.get('min_points', 30))
        self.max_candidates = min(6, int(self.config.get('qwen_vl_max_candidates', 6)))
        self.min_pixels = int(self.config.get('min_box_pixels', 1024))
        if self.min_points < 3 or self.max_candidates < 1 or self.min_pixels < 1:
            raise ValueError('Invalid interaction proposal limits')
        self.margin = float(self.config.get('geometry_margin_m', .03))
        if not math.isfinite(self.margin) or not 0 <= self.margin <= .20:
            raise ValueError('geometry_margin_m must be between 0 and .20 m')
        self.min_extent = self._positive('min_geometry_extent_m', .04)
        self.max_extent = self._positive('max_geometry_extent_m', 2.)
        self.association_distance = self._positive('association_distance_m', .30)
        self.geometry_verified = self.config.get('geometry_verified', False)
        if not isinstance(self.geometry_verified, bool):
            raise ValueError('geometry_verified must be an explicit boolean')
        valid = {field.name for field in fields(OpenVocabularyPerceptionConfig)}
        parameters = {key: value for key, value in config.items() if key in valid}
        parameters['semantic_classifier'] = 'qwen-vl'
        self.backend = backend or AgentVLMBackend(OpenVocabularyPerceptionConfig(**parameters))
        self._temporary = None
        if self.config.get('evidence_dir'):
            self.directory = Path(self.config['evidence_dir'])
            self.directory.mkdir(parents=True, exist_ok=True)
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix='inter-nav-recognition-')
            self.directory = Path(self._temporary.name)
        self.jobs, self.results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self._images = deque()
        self.generation = 0
        self._new_state()
        self.thread = threading.Thread(target=self._work, daemon=True, name='interaction-perception')
        self.thread.start()

    def _positive(self, name, default):
        value = float(self.config.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    def _new_state(self):
        self.store = RecognitionStore(min_frames=2, min_detector_score=self.min_score,
                                      max_age_s=self.recognition_max_age,
                                      aliases=self.catalog.label_aliases)
        self.last_error = self.last_success = self.last_delivery = None
        self.last_query_status, self.last_frame = 'idle', None
        self.last_submit = -math.inf
        self._seen_frames = deque(maxlen=256)
        self._tracks, self._visible = {}, {}
        self._next_id, self._planning_frame = 0, None

    def reset(self):
        self.generation += 1
        self._clear(self.jobs)
        self._clear(self.results)
        self._new_state()

    @staticmethod
    def _clear(channel):
        while True:
            try:
                channel.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _replace(channel, item):
        AsyncInteractionPerception._clear(channel)
        try:
            channel.put_nowait(item)
        except queue.Full:
            pass

    def _capture(self, snapshot, now):
        camera = snapshot.observation.get('sensors', {}).get('camera')
        if camera is None or snapshot.camera_at is None:
            self.last_query_status = 'no_rgbd'
            return None
        if not snapshot.geometry_calibrated or snapshot.planning_frame in (None, '', 'unknown'):
            raise ValueError('Interaction RGB-D requires calibrated, named planning-frame geometry')
        if camera.get('pointcloud_frame', snapshot.planning_frame) != snapshot.planning_frame:
            raise ValueError('RGB-D pointcloud frame differs from the planning frame')
        stamp = float(snapshot.camera_at)
        age = now - stamp
        if not math.isfinite(age) or age < 0 or age >= self.input_max_age:
            raise ValueError('Interaction RGB-D camera capture is stale, future or invalid')
        if self._planning_frame is not None and self._planning_frame != snapshot.planning_frame:
            raise ValueError('Interaction planning frame changed; reset perception before use')
        frame_id = str(camera['capture_step'])
        if frame_id in self._seen_frames or now-self.last_submit < self.interval:
            return None
        rgba, depth = np.asarray(camera['rgba']), np.asarray(camera['depth'])
        points = np.asarray(camera.get('pointcloud_image', camera.get('pointcloud')))
        if depth.ndim == 2 and points.shape == (depth.size, 3):
            points = points.reshape(*depth.shape, 3)
        if (rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] not in (3, 4)
                or depth.ndim != 2 or depth.shape != rgba.shape[:2]
                or points.shape != (*depth.shape, 3)):
            raise ValueError('Interaction requires pixel-aligned uint8 RGB, depth and planning-frame XYZ')
        self._planning_frame = snapshot.planning_frame
        self.last_submit, self.last_frame = now, frame_id
        self._seen_frames.append(frame_id)
        return _Frame(self.generation, frame_id, stamp, snapshot.planning_frame,
                      rgba.copy(), depth.copy(), points.copy())

    def update(self, snapshot, now):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('Interaction update time must be finite')
        if self.closed.is_set():
            return (), self.store
        self._expire(now)
        try:
            frame, detections, error = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            if frame.generation == self.generation:
                if error is not None:
                    self.last_error, self.last_query_status = error, 'error'
                    self._visible.clear()
                elif not 0 <= now-frame.captured_at < self.result_max_age:
                    self.last_error = 'discarded stale or future interaction perception result'
                    self.last_query_status = 'stale_result'
                else:
                    self._accept(frame, detections, now)
        try:
            frame = self._capture(snapshot, now)
        except (KeyError, TypeError, ValueError) as error:
            if self.last_query_status not in ('error', 'stale_result'):
                self.last_error, self.last_query_status = str(error), 'invalid_rgbd'
            self._visible.clear()
            return (), self.store
        if frame is not None:
            self._replace(self.jobs, frame)
        if snapshot.camera_at is None or snapshot.observation.get('sensors', {}).get('camera') is None:
            self._visible.clear()
            return (), self.store
        boxes = tuple(box for box, stamp in self._visible.values()
                      if 0 <= now-stamp < self.geometry_max_age)
        return boxes if self.geometry_verified else (), self.store

    def _expire(self, now):
        # The release store expires by newest vote; expire individual votes too.
        for key, item in list(self.store.objects.items()):
            item.evidence = [e for e in item.evidence if 0 <= now-e.observed_at < self.recognition_max_age]
            item.seen_frames = {e.frame_id for e in item.evidence}
            if not item.evidence:
                self.store.objects.pop(key)
        self._tracks = {key: track for key, track in self._tracks.items()
                        if 0 <= now-track[1] < self.recognition_max_age}
        self._visible = {key: value for key, value in self._visible.items()
                         if 0 <= now-value[1] < self.geometry_max_age}

    def _accept(self, frame, detections, now):
        claimed, visible, records = set(), {}, []
        for detection in detections:
            center, size = np.asarray(detection.center), np.asarray(detection.dimensions)
            choices = []
            for key, (previous, stamp) in self._tracks.items():
                if key in claimed or stamp >= frame.captured_at:
                    continue
                distance = float(np.linalg.norm(center-np.asarray(previous.center)))
                ratio = np.maximum(size/previous.dimensions, np.asarray(previous.dimensions)/size)
                if distance <= self.association_distance and np.all(ratio <= 1.8):
                    choices.append((distance, key))
            choices.sort()
            # Ambiguous neighbours never inherit a nearby object's permission.
            if choices and (len(choices) == 1 or choices[1][0]-choices[0][0] > .08):
                object_id = choices[0][1]
            else:
                self._next_id += 1
                object_id = f'observed-{self.generation}-{self._next_id}'
            claimed.add(object_id)
            self._tracks[object_id] = (detection, frame.captured_at)
            evidence = RecognitionEvidence(object_id, detection.category, frame.frame_id,
                frame.captured_at, detection.score, detection.model, detection.image_path,
                'rgb_dino_qwen_vl', raw_response=detection.raw_response)
            if detection.category == 'unknown':
                self.store.objects.pop(object_id, None)
            else:
                self.store.observe(evidence)
            visible[object_id] = (BoxState(object_id, tuple(center[:2]), detection.dimensions,
                                 mass_kg=None, center_z=float(center[2])), frame.captured_at)
            records.append({'object_id': object_id, 'category': detection.category,
                'center': detection.center, 'dimensions': detection.dimensions,
                'captured_at': frame.captured_at, 'image_path': detection.image_path,
                'frame_id': frame.frame_id, 'classifier_model': detection.model,
                'detector_score': detection.score, 'classifier_response': detection.raw_response,
                'semantic_source': 'rgb_dino_qwen_vl',
                'crop_jpeg_base64': detection.crop_jpeg_base64,
                'geometry_source': 'segmented_rgbd_visible_aabb',
                'geometry_verified': self.geometry_verified})
        self._visible = visible
        self.last_success, self.last_error, self.last_query_status = now, None, 'ok'
        self.last_delivery = {'frame_id': frame.frame_id, 'captured_at': frame.captured_at,
            'received_at': now, 'planning_frame': frame.planning_frame,
            'geometry_source': 'segmented_rgbd_visible_aabb',
            'geometry_verified': self.geometry_verified, 'detections': records}
        if len(self._tracks) > 64:
            keep = set(sorted(self._tracks, key=lambda key: self._tracks[key][1], reverse=True)[:64])
            self._tracks = {key: value for key, value in self._tracks.items() if key in keep}
            self.store.objects = {key: value for key, value in self.store.objects.items() if key in keep}

    def _infer(self, frame, budget):
        backend = self.backend
        if isinstance(backend, AgentVLMBackend):
            backend = AgentVLMBackend(backend.config, _BudgetSession(backend._session, budget))
        budget.remaining()
        image = cv2.cvtColor(frame.rgba, cv2.COLOR_RGBA2BGR if frame.rgba.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        raw = backend.detect(image, self.catalog.proposal_query)
        budget.remaining()
        proposals = []
        for item in raw:
            _, score, bbox = OpenVocabularyPerception._parse_detection(item, image.shape[:2])
            if (math.isfinite(score) and self.min_score <= score <= 1. and np.isfinite(bbox).all()
                    and (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) >= self.min_pixels):
                proposals.append((score, bbox))
        detections, used_masks = [], []
        for index, (score, bbox) in enumerate(sorted(proposals, reverse=True)[:self.max_candidates]):
            budget.remaining()
            mask = np.asarray(backend.segment(image, bbox), bool)
            budget.remaining()
            if mask.shape != frame.depth.shape:
                raise ValueError('SAM mask must match captured RGB-D pixels')
            if any(np.count_nonzero(mask & other)/max(1, np.count_nonzero(mask | other)) > .65
                   for other in used_masks):
                continue
            used_masks.append(mask)
            valid = mask & np.isfinite(frame.depth) & (frame.depth > .05) & (frame.depth < 5.)
            valid &= np.isfinite(frame.points).all(axis=-1)
            points = frame.points[valid]
            if len(points) < self.min_points:
                continue
            if np.any(np.abs(points) > 1.e4):
                raise ValueError('RGB-D XYZ exceeds the supported metric planning-frame range')
            lower, upper = points.min(axis=0), points.max(axis=0)
            span = upper-lower
            # Padding must never manufacture thickness from a single flat face.
            if np.any(span < self.min_extent) or np.any(span+2*self.margin > self.max_extent):
                continue
            crop = context_crop(emphasize_instance(image, mask), bbox)
            budget.remaining()
            classified = backend.classify(crop, self.catalog.classification_labels)
            budget.remaining()
            category, model = classified.get('label'), classified.get('model')
            if category not in self.catalog.classification_labels or not isinstance(model, str) or not model.strip():
                raise ValueError('Classifier must return a supported RGB category and model provenance')
            image_path = self.directory / f'{frame.generation}_{abs(hash(frame.frame_id))}_{index}.jpg'
            encoded, jpeg = cv2.imencode('.jpg', crop)
            if not encoded:
                raise OSError('Could not encode RGB interaction classification evidence')
            image_path.write_bytes(jpeg.tobytes())
            self._images.append(image_path)
            if len(self._images) > 2048:
                expired_image = self._images.popleft()
                if self._temporary is not None:
                    expired_image.unlink(missing_ok=True)
            detections.append(_Detection(tuple((lower+upper)/2), tuple(span+2*self.margin),
                category, score, model, str(image_path), base64.b64encode(jpeg).decode('ascii'),
                classified.get('raw_text', '')))
        return detections

    def _work(self):
        try:
            while not self.closed.is_set():
                try:
                    frame = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                budget = RequestBudget(frame.captured_at+self.result_max_age,
                    lambda: self.closed.is_set() or frame.generation != self.generation, self.clock)
                try:
                    detections = self._infer(frame, budget)
                    budget.remaining()
                    error = None
                except PerceptionCancelled:
                    continue
                except Exception as exc:
                    detections, error = [], f'{type(exc).__name__}: {exc}'
                if frame.generation == self.generation and not self.closed.is_set():
                    self._replace(self.results, (frame, detections, error))
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()

    def close(self):
        self.closed.set()
        self._clear(self.jobs)
        self._clear(self.results)
        self.thread.join(timeout=.2)
