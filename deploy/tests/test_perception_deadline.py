"""Real inference plumbing with fake HTTP/clock; no model or network access."""
import base64
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from deploy.perception.client import AsyncPerception
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend, OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
    PerceptionCancelled, PerceptionDeadlineExceeded, RequestBudget,
)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakeHTTP:
    def __init__(self, clock, duration=.5):
        self.clock, self.duration = clock, duration
        self.calls = []
        self.before = None

    def post(self, url, **kwargs):
        index = len(self.calls)
        self.calls.append((url, kwargs))
        if self.before is not None:
            self.before(index)
        allowed = kwargs['timeout']
        self.clock.now += min(self.duration, allowed)
        if self.duration > allowed:
            raise requests.ReadTimeout('test model response timed out')
        if url.endswith('/gdino'):
            payload = {'boxes': [[1, 1, 8, 8], [11, 11, 18, 18]],
                       'scores': [.9, .8], 'labels': ['chair', 'chair']}
        elif url.endswith('/mobile_sam'):
            mask = np.ones((20, 20), dtype=np.uint8)
            payload = {'cropped_mask': base64.b64encode(mask.tobytes()).decode()}
        elif url.endswith('/classify'):
            payload = {'label': 'chair', 'model': 'fake-qwen'}
        else:
            raise AssertionError(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def pipeline(http):
    config = OpenVocabularyPerceptionConfig(
        semantic_classifier='qwen-vl', query_interval=0., request_timeout=2., qwen_vl_timeout=5.)
    transport = AgentVLMBackend(config, session=http)
    return OpenVocabularyPerception(config, detector=transport, segmenter=transport,
                                    classifier=transport, clock=http.clock)


def arguments(target='chair'):
    return {'rgba': np.zeros((20, 20, 3), dtype=np.uint8),
            'depth': np.ones((20, 20)), 'point_image': np.ones((20, 20, 3)),
            'target': target, 'step': 1}


def client_config(max_age=8.):
    return {'semantic_classifier': 'qwen-vl', 'result_max_age_s': max_age}


def wait_for(condition):
    deadline = time.monotonic()+2
    while not condition() and time.monotonic() < deadline:
        time.sleep(.005)
    assert condition()


def test_default_shared_pipeline_keeps_all_candidates_and_configured_timeouts():
    clock = Clock()
    http = FakeHTTP(clock)
    detections = pipeline(http).perceive(**arguments())
    assert len(detections) == 2
    assert [kw['timeout'] for _, kw in http.calls] == [2., 2., 2., 5., 2., 5.]


def test_each_http_request_uses_remaining_capture_budget():
    clock = Clock()
    http = FakeHTTP(clock, duration=1.)
    budget = RequestBudget(3.5, cancelled=lambda: False, clock=clock)
    with pytest.raises(requests.ReadTimeout, match='test model response timed out'):
        pipeline(http).perceive(**arguments(), request_budget=budget)
    # Two DINO passes, one SAM, one Qwen; the second proposal is never requested.
    assert [kw['timeout'] for _, kw in http.calls] == [2., 2., 1.5, .5]
    assert clock() == 3.5


def test_expiry_after_successful_response_does_not_start_another_candidate():
    clock = Clock()
    http = FakeHTTP(clock)
    budget = RequestBudget(2., cancelled=lambda: False, clock=clock)
    with pytest.raises(PerceptionDeadlineExceeded):
        pipeline(http).perceive(**arguments(), request_budget=budget)
    assert len(http.calls) == 4


def test_cancel_after_current_request_prevents_next_request_and_fallback():
    clock = Clock()
    cancelled = [False]
    http = FakeHTTP(clock)
    http.before = lambda _: cancelled.__setitem__(0, True)
    backend = pipeline(http)
    backend.simulator_fallback = lambda **kw: pytest.fail('cancellation must not use a fallback')
    budget = RequestBudget(8., cancelled=lambda: cancelled[0], clock=clock)
    with pytest.raises(PerceptionCancelled):
        backend.perceive(**arguments(), request_budget=budget)
    assert len(http.calls) == 1


def test_expired_frame_never_starts_http():
    clock = Clock()
    http = FakeHTTP(clock)
    with pytest.raises(PerceptionDeadlineExceeded):
        pipeline(http).perceive(**arguments(), request_budget=RequestBudget(
            0., cancelled=lambda: False, clock=clock))
    assert not http.calls


def test_async_preserves_http_failure_when_response_uses_up_frame_budget():
    clock = Clock()
    clock.now = 2.
    http = FakeHTTP(clock, duration=2.)
    client = AsyncPerception(client_config(3.), backend=pipeline(http), clock=clock)
    try:
        client.set_frame(1, 0.)
        assert client.perceive(**arguments()) == []
        wait_for(lambda: not client.results.empty())
        assert client.perceive(**arguments()) == []
        assert 'ReadTimeout: test model response timed out' == client.last_error
        assert client.last_success is None
        assert len(http.calls) == 1 and http.calls[0][1]['timeout'] == 1.
    finally:
        client.close()


@pytest.mark.parametrize('old_request_fails', [False, True])
def test_reset_cancels_followup_requests_clears_queue_and_keeps_new_task_diagnostics(old_request_fails):
    clock = Clock()
    http = FakeHTTP(clock, duration=.1)
    old_entered, old_release = threading.Event(), threading.Event()
    new_entered, new_release = threading.Event(), threading.Event()

    def block(index):
        if index == 0:
            old_entered.set()
            assert old_release.wait(2)
            if old_request_fails:
                raise requests.ConnectionError('old task connection failed')
        elif index == 1:
            new_entered.set()
            assert new_release.wait(2)

    http.before = block
    client = AsyncPerception(client_config(), backend=pipeline(http), clock=clock)
    try:
        client.set_frame(1, clock())
        client.perceive(**arguments('refrigerator'))
        assert old_entered.wait(1)
        client.set_frame(2, clock())
        client.perceive(**arguments('refrigerator'))  # Pending old-generation frame.
        assert not client.jobs.empty()
        client.reset()
        assert client.jobs.empty() and client.results.empty() and client.frame is None
        client.last_error = 'new task diagnostic'
        client.set_frame(3, clock())
        client.perceive(**arguments('chair'))
        old_release.set()
        assert new_entered.wait(1)
        assert client.results.empty() and client.last_error == 'new task diagnostic'
        new_release.set()
        wait_for(lambda: not client.results.empty())
        result = client.perceive(**arguments('chair'))
        assert len(result) == 2 and all(item.step == 3 for item in result)
        assert len(http.calls) == 7  # One old in-flight request + six new-frame requests.
        assert client.last_delivery['target'] == 'chair'
        assert client.last_error is None
    finally:
        old_release.set()
        new_release.set()
        client.close()


def test_close_does_not_interrupt_inflight_http_but_prevents_followup_work():
    clock = Clock()
    http = FakeHTTP(clock)
    entered, release = threading.Event(), threading.Event()

    def block(index):
        entered.set()
        assert release.wait(2)

    http.before = block
    client = AsyncPerception(client_config(), backend=pipeline(http), clock=clock)
    try:
        client.set_frame(1, clock())
        client.perceive(**arguments())
        assert entered.wait(1)
        client.close()
        assert client.thread.is_alive()  # An in-flight request is not forcibly killed.
        release.set()
        client.thread.join(1)
        assert not client.thread.is_alive()
        assert len(http.calls) == 1 and client.results.empty()
    finally:
        release.set()
        client.close()
