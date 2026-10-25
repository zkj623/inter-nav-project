"""Viewer contracts: stale/independent streams, ROS layouts, read-only HTTP."""
import base64
import json
import threading
import time
from types import SimpleNamespace as Message
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from deploy.visualization.server import make_server
from deploy.visualization.sources import LatestFrames, jpeg_from_bgr
from deploy.tests.test_robot_inputs import cloud
from deploy.robot.sensors import decode_cloud, decode_image


def test_bounded_cloud_frame_and_no_double_pose_transform():
    frames = LatestFrames(max_points=2)
    message = cloud()
    frames.cloud(decode_cloud(message), message.header.frame_id, 10., 0.)
    result = frames.snapshot()['cloud']
    assert result['frame'] == 'odom'
    np.testing.assert_allclose(result['points'], [[2, .2, .7], [3, .2, .7]])
    frames.cloud([[i, 0, 0] for i in range(10)] + [[np.nan, 0, 0]], 'odom')
    result = frames.snapshot()['cloud']
    assert result['count'] == 10
    assert len(result['points']) == 2
    assert result['points'] == [[0, 0, 0], [9, 0, 0]]


def test_stale_streams_and_errors_do_not_hide_each_other():
    frames = LatestFrames()
    now = time.monotonic()
    frames.put('cloud', {'frame': 'odom', 'stamp': 3., 'source_age': 0.}, now - 1)
    frames.put('rgb', {'source': 'sdk', 'source_age': None}, now)
    frames.error('pose', 'No odometry')
    state = frames.snapshot(now)
    assert state['cloud']['stale']
    assert not state['rgb']['stale']
    assert state['pose'] is None
    assert state['errors'] == {'pose': 'No odometry'}
    frames.error('rgb', 'Camera failed')
    assert frames.snapshot(now)['rgb']['stale']
    frames.put('rgb', {'source_age': None}, now)
    assert 'rgb' not in frames.snapshot(now)['errors']


@pytest.mark.parametrize('source_age', [2., -.1])
def test_old_or_future_capture_not_fresh_despite_new_receipt(source_age):
    frames = LatestFrames()
    frames.put('cloud', {'source_age': source_age})
    assert frames.snapshot()['cloud']['stale']


@pytest.mark.parametrize('frame,stamp,overlay', [('odom', 1.02, True), ('base_link', 1., False), ('odom', 2., False)])
def test_pose_overlay_requires_matching_frame_and_time(frame, stamp, overlay):
    frames = LatestFrames()
    frames.cloud([[1, 0, 0]], 'odom', 1., 0.)
    frames.put('pose', {'frame': frame, 'stamp': stamp, 'source_age': 0.})
    assert frames.snapshot()['pose_overlay'] is overlay


def test_rgb_ros_colour_order_and_no_depth_requirement():
    frames = LatestFrames()
    msg = Message(width=16, height=16, step=48, encoding='rgb8', is_bigendian=False,
                  data=bytes([255, 0, 0]) * 256)
    frames.put('rgb', {'jpeg': jpeg_from_bgr(decode_image(msg)[..., ::-1])})
    payload = frames.snapshot()['rgb']
    image = cv2.imdecode(np.frombuffer(base64.b64decode(payload['jpeg']), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image[0, 0, 2] > 240 and image[0, 0, 0] < 10
    assert frames.snapshot()['cloud'] is None


def test_jpeg_size_bound_and_invalid_input():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    raw = base64.b64decode(jpeg_from_bgr(image))
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (540, 960, 3)
    with pytest.raises(ValueError):
        jpeg_from_bgr(None)


def test_actual_http_has_only_read_routes_and_explicit_demo():
    frames = LatestFrames(demo=True)
    frames.cloud([[1, 2, 3]], 'demo_odom')
    server = make_server(frames, 0, {'fps': 5})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = 'http://127.0.0.1:' + str(server.server_port)
    try:
        with urlopen(url + '/') as response:
            html = response.read().decode()
            assert '累计点云图' in html and '前置 RGB' in html
        with urlopen(url + '/api/frame') as response:
            state = json.load(response)
            assert state['demo'] and state['rgb'] is None
            assert state['cloud']['points'] == [[1, 2, 3]]
        for path, method, expected in [('/move', 'GET', 404), ('/api/frame', 'POST', 501), ('/../configs/robot.yaml', 'GET', 404)]:
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(url + path, method=method))
            assert exc.value.code == expected
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize('code', [0, 1001])
def test_sdk_video_worker_decodes_image_or_reports_return_code(monkeypatch, code):
    import queue
    import sys
    from deploy.visualization.sources import _video_worker
    calls = []
    stop = threading.Event()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 255
    _, encoded = cv2.imencode('.jpg', image)

    class VideoClient:
        def SetTimeout(self, timeout):
            calls.append(('timeout', timeout))

        def Init(self):
            calls.append(('init',))

        def GetImageSample(self):
            stop.set()
            return code, encoded.tobytes()

    monkeypatch.setitem(sys.modules, 'unitree_sdk2py.core.channel', Message(
        ChannelFactoryInitialize=lambda *args: calls.append(args)))
    monkeypatch.setitem(sys.modules, 'unitree_sdk2py.go2.video.video_client', Message(VideoClient=VideoClient))
    output = queue.Queue(maxsize=1)
    _video_worker(output, stop, 'test_interface', 5)
    _, payload, error = output.get_nowait()
    assert calls == [(0, 'test_interface'), ('timeout', 1.), ('init',)]
    if code:
        assert payload is None and '1001' in error
    else:
        assert error is None and payload['stamp'] is None
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload['jpeg']), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded[0, 0, 1] > 240


def test_view_exit_closes_client_without_stopping_run(monkeypatch):
    import signal
    import deploy.visualization.server as module
    calls = []
    original = signal.getsignal(signal.SIGINT)

    class FakeClient:
        def __init__(self, *args):
            pass

        def close(self):
            calls.append('closed_client')

    class FakeServer:
        def handle_request(self):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            calls.append('handled')

        def server_close(self):
            calls.append('closed_http')

    monkeypatch.setattr(module, 'SessionClient', FakeClient)
    monkeypatch.setattr(module, 'make_server', lambda *args: FakeServer())
    assert module.serve(Message(source='http://localhost:8765', port=8080)) == 0
    assert calls == ['handled', 'closed_http', 'closed_client']
    assert signal.getsignal(signal.SIGINT) == original
