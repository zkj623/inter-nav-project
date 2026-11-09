"""Robot recognition contracts with fake RGB-D and HTTP; never contacts a robot."""
import base64
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import cv2
import pytest
import requests

from deploy.perception.client import AsyncInteractionPerception
from deploy.robot.observation import Snapshot
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend, OpenVocabularyPerceptionConfig,
)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Backend:
    def __init__(self, category='cardboard box'):
        self.category = category
        self.calls = []
        self.before_detect = None

    def detect(self, image, query):
        self.calls.append('detect')
        if self.before_detect:
            self.before_detect()
        return [{'bbox': [0, 0, 39, 39], 'confidence': .9, 'label': 'box'}]

    def segment(self, image, bbox):
        self.calls.append('segment')
        return np.ones(image.shape[:2], bool)

    def classify(self, image, labels):
        self.calls.append('classify')
        return {'label': self.category, 'model': 'fake-qwen', 'raw_text': self.category}


def snapshot(frame_id=1, captured_at=0., *, offset=(0., 0., 0.), dense=True):
    row, col = np.mgrid[:40, :40]
    points = np.stack((.6+.4*row/39, -.2+.4*col/39, .05+.4*(row+col)/78), -1)
    points += np.asarray(offset)
    camera = {'rgba': np.zeros((40, 40, 3), np.uint8), 'depth': np.ones((40, 40)),
              'capture_step': frame_id, 'pointcloud_image' if dense else 'pointcloud':
              points if dense else points.reshape(-1, 3)}
    return Snapshot({'position': np.zeros(3), 'sensors': {'camera': camera}},
                    captured_at, captured_at, captured_at, captured_at, (0., 0., 0.),
                    frame_id, planning_frame='experiment')


def client(clock=None, backend=None, **options):
    return AsyncInteractionPerception({'query_interval_s': 0., 'geometry_verified': True,
                                      **options}, backend=backend or Backend(), clock=clock or Clock())


def wait_for(condition):
    deadline = time.monotonic()+2.
    while not condition() and time.monotonic() < deadline:
        time.sleep(.005)
    assert condition()


def deliver(engine, packet, now):
    engine.update(packet, now)
    wait_for(lambda: not engine.results.empty())
    return engine.update(packet, now)


def confirmed(engine, clock, **kwargs):
    _, store = deliver(engine, snapshot(1, clock(), **kwargs), clock())
    assert not engine.catalog.eligible_ids('push_aside', store, clock())
    clock.now += .1
    return deliver(engine, snapshot(2, clock(), **kwargs), clock())


@pytest.mark.parametrize('dense', [True, False])
def test_two_distinct_rgb_frames_measure_planning_frame_bounds_and_never_guess_mass(dense):
    clock, backend = Clock(), Backend()
    engine = client(clock, backend)
    try:
        boxes, store = confirmed(engine, clock, offset=(10., -2., 0.), dense=dense)
        assert len(boxes) == 1
        box = boxes[0]
        assert box.xy == pytest.approx((10.8, -2.))
        assert box.dimensions == pytest.approx((.46, .46, .46))
        assert box.center_z == pytest.approx(.25)
        assert box.mass_kg is None
        permission = engine.catalog.permission(box.id, 'push_aside', store, clock())
        assert permission.evidence_frames == ('1', '2')
        assert permission.category == 'box'
        assert backend.calls == ['detect', 'segment', 'classify']*2
        engine.update(snapshot(2, clock(), dense=dense), clock())
        assert engine.jobs.empty() and len(backend.calls) == 6
    finally:
        engine.close()


def test_visible_envelope_is_not_an_executable_box_without_geometry_acceptance():
    clock = Clock()
    engine = client(clock, geometry_verified=False)
    try:
        boxes, store = confirmed(engine, clock)
        assert boxes == ()
        assert engine.catalog.eligible_ids('push_aside', store, clock())
        assert engine.last_delivery['geometry_verified'] is False
        assert engine.last_delivery['detections'][0]['geometry_source'] == 'segmented_rgbd_visible_aabb'
    finally:
        engine.close()


def test_classification_image_roundtrip_survives_temporary_evidence_cleanup():
    engine = client()
    try:
        deliver(engine, snapshot(), 0.)
        detection = engine.last_delivery['detections'][0]
        saved_path = Path(detection['image_path'])
        encoded = base64.b64decode(detection['crop_jpeg_base64'], validate=True)
        assert encoded == saved_path.read_bytes()
        restored = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        assert restored is not None and max(restored.shape[:2]) == 384
        assert detection['classifier_model'] == 'fake-qwen'
        assert detection['semantic_source'] == 'rgb_dino_qwen_vl'
        assert detection['frame_id'] == '1' and detection['detector_score'] == .9
    finally:
        engine.close()
    assert not saved_path.exists()
    assert cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR) is not None


@pytest.mark.parametrize('category', ['unknown', 'chair', 'person'])
def test_non_box_rgb_category_never_gets_push_permission(category):
    clock, backend = Clock(), Backend(category)
    engine = client(clock, backend)
    try:
        boxes, store = confirmed(engine, clock)
        assert len(boxes) == 1  # An inspectable obstacle, never a push permission.
        assert not engine.catalog.eligible_ids('push_aside', store, clock())
    finally:
        engine.close()


def test_unknown_revokes_previous_permission_and_requires_two_new_positive_frames():
    clock, backend = Clock(), Backend()
    engine = client(clock, backend)
    try:
        boxes, _ = confirmed(engine, clock)
        original_id = boxes[0].id
        backend.category = 'unknown'
        clock.now += .1
        boxes, store = deliver(engine, snapshot(3, clock()), clock())
        assert len(boxes) == 1 and engine.catalog.permission(original_id, 'push_aside', store, clock()) is None
        backend.category = 'box'
        clock.now += .1
        _, store = deliver(engine, snapshot(4, clock()), clock())
        assert not engine.catalog.eligible_ids('push_aside', store, clock())
    finally:
        engine.close()


@pytest.mark.parametrize('category', ['made-up-object', 'BOX'])
def test_unrecognized_classifier_label_is_rejected(category):
    engine = client(backend=Backend(category))
    try:
        assert deliver(engine, snapshot(), 0.)[0] == ()
        assert 'supported RGB category' in engine.last_error
        assert not engine.store.objects
    finally:
        engine.close()


@pytest.mark.parametrize('failure', ['unaligned', 'flat', 'oversize', 'invalid_depth', 'invalid_points'])
def test_bad_or_unmeasured_geometry_never_enters_interaction(failure):
    engine, packet = client(), snapshot()
    camera = packet.observation['sensors']['camera']
    if failure == 'unaligned':
        camera['pointcloud_image'] = np.zeros((10, 10, 3))
    elif failure == 'flat':
        camera['pointcloud_image'][:, :, 0] = .8
    elif failure == 'oversize':
        camera['pointcloud_image'][:, :, 0] *= 10
    elif failure == 'invalid_depth':
        camera['depth'][:] = np.nan
    else:
        camera['pointcloud_image'][:] = np.nan
    try:
        engine.update(packet, 0.)
        if failure != 'unaligned':
            wait_for(lambda: not engine.results.empty())
            assert engine.update(packet, 0.)[0] == ()
        else:
            assert 'pixel-aligned' in engine.last_error
            assert engine.jobs.empty()
        assert not engine.store.objects
    finally:
        engine.close()


@pytest.mark.parametrize('failure', ['no_camera', 'stale_capture', 'future_capture', 'uncalibrated', 'wrong_frame'])
def test_actual_camera_time_and_coordinate_contract_gate_input(failure):
    backend, packet = Backend(), snapshot()
    if failure == 'no_camera':
        packet.observation['sensors'].clear()
        packet.camera_at = None
    elif failure == 'stale_capture':
        packet.captured_at = 10.  # Packet receipt must not renew the old RGB-D.
        packet.camera_at = -.6
    elif failure == 'future_capture':
        packet.camera_at = .001  # Even a small future capture cannot vote yet.
    elif failure == 'uncalibrated':
        packet.geometry_calibrated = False
    else:
        packet.observation['sensors']['camera']['pointcloud_frame'] = 'camera_optical'
    engine = client(backend=backend)
    try:
        assert engine.update(packet, 0.)[0] == ()
        assert engine.jobs.empty() and not backend.calls
    finally:
        engine.close()


def test_late_classification_can_update_evidence_but_cannot_renew_geometry():
    clock, backend = Clock(), Backend()
    engine = client(clock, backend, geometry_max_age_s=.15)
    try:
        confirmed(engine, clock)
        clock.now = .2
        packet = snapshot(3, clock())
        backend.before_detect = lambda: setattr(clock, 'now', .4)
        boxes, store = deliver(engine, packet, .4)
        assert boxes == ()
        assert engine.catalog.eligible_ids('push_aside', store, .4)
        assert engine.last_delivery['captured_at'] == .2
        assert engine.last_delivery['received_at'] == .4
    finally:
        engine.close()


def test_old_first_vote_does_not_confirm_new_frame():
    clock = Clock()
    engine = client(clock, recognition_max_age_s=.2)
    try:
        deliver(engine, snapshot(), 0.)
        clock.now = .3
        boxes, store = deliver(engine, snapshot(2, clock()), clock())
        assert len(boxes) == 1 and not engine.catalog.eligible_ids('push_aside', store, clock())
    finally:
        engine.close()


@pytest.mark.parametrize('raise_old', [False, True])
def test_reset_discards_inflight_work_and_cancels_followup_calls(raise_old):
    entered, release = threading.Event(), threading.Event()
    clock, backend = Clock(), Backend()
    def block():
        entered.set()
        assert release.wait(2.)
        if raise_old:
            raise requests.ConnectionError('old task failure')
    backend.before_detect = block
    engine = client(clock, backend)
    try:
        engine.update(snapshot(), 0.)
        assert entered.wait(1.)
        engine.update(snapshot(2), 0.)
        assert engine.jobs.qsize() == 1
        engine.reset()
        engine.last_error = 'new task status'
        release.set()
        wait_for(lambda: engine.jobs.empty())
        time.sleep(.02)
        assert engine.results.empty() and engine.last_error == 'new task status'
        assert backend.calls == ['detect']
        assert not engine.store.objects
    finally:
        release.set()
        engine.close()


class HTTP:
    def __init__(self, clock, duration=1.):
        self.clock, self.duration, self.calls = clock, duration, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        timeout = kwargs['timeout']
        self.clock.now += min(timeout, self.duration)
        if timeout < self.duration:
            raise requests.ReadTimeout('model timed out')
        if url.endswith('/gdino'):
            payload = {'boxes': [[0, 0, 39, 39]], 'scores': [.9], 'labels': ['box']}
        elif url.endswith('/mobile_sam'):
            payload = {'cropped_mask': base64.b64encode(np.ones((40, 40), np.uint8).tobytes()).decode()}
        else:
            payload = {'label': 'box', 'model': 'fake-qwen'}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def test_http_pipeline_uses_one_capture_deadline_and_keeps_timeout_diagnostic():
    clock = Clock()
    http = HTTP(clock)
    backend = AgentVLMBackend(OpenVocabularyPerceptionConfig(
        semantic_classifier='qwen-vl', request_timeout=2., qwen_vl_timeout=5.), http)
    engine = client(clock, backend, result_max_age_s=2.5)
    try:
        engine.update(snapshot(), 0.)
        wait_for(lambda: not engine.results.empty())
        assert engine.update(snapshot(), clock())[0] == ()
        assert [call[1]['timeout'] for call in http.calls] == [2., 1.5, .5]
        assert engine.last_error == 'ReadTimeout: model timed out'
        assert not engine.store.objects
    finally:
        engine.close()


def test_expired_result_does_not_create_permission():
    clock = Clock()
    engine = client(clock, result_max_age_s=.3)
    try:
        engine.update(snapshot(), 0.)
        wait_for(lambda: not engine.results.empty())
        clock.now = .4
        assert engine.update(snapshot(), clock())[0] == ()
        assert not engine.store.objects
        assert 'stale' in engine.last_error
    finally:
        engine.close()
