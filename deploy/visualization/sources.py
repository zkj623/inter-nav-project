"""Preview buffers and SDK camera worker, owned only by run.py."""
import base64
import multiprocessing as mp
import queue
import threading
import time

import cv2
import numpy as np



def jpeg_from_bgr(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected a decoded colour image')
    h, w = image.shape[:2]
    scale = min(1., 960 / w, 540 / h)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError('JPEG encode failed')
    return base64.b64encode(encoded).decode('ascii')


class LatestFrames:
    """One immutable payload per stream; old data remains visible but is marked stale."""
    def __init__(self, max_points=12000, stale_after=.4, demo=False):
        self.lock = threading.Lock()
        self.max_points, self.stale_after, self.demo = max_points, stale_after, demo
        self.streams = {name: None for name in ('cloud', 'pose', 'rgb')}
        self.errors = {}

    def error(self, name, error):
        with self.lock:
            self.errors[name] = str(error)

    def put(self, name, payload, received=None):
        with self.lock:
            self.streams[name] = (time.monotonic() if received is None else received, payload)
            self.errors.pop(name, None)

    def cloud(self, points, frame, stamp=None, source_age=None):
        if not frame:
            raise ValueError('Point cloud frame_id is empty')
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected N x 3 points')
        points = points[np.isfinite(points).all(axis=1)]
        count = len(points)
        if not count:
            raise ValueError('No finite point cloud returns')
        if count > self.max_points:
            points = points[np.linspace(0, count - 1, self.max_points, dtype=int)]
        self.put('cloud', {'points': np.round(points.astype(float), 3).tolist(),
                           'count': count, 'frame': frame, 'stamp': stamp, 'source_age': source_age})

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            result = {'demo': self.demo, 'errors': dict(self.errors)}
            for name, item in self.streams.items():
                if item is None:
                    result[name] = None
                    continue
                received, payload = item
                age = max(0., now - received)
                source_age = payload.get('source_age')
                stale = age > self.stale_after or name in self.errors
                if source_age is not None:
                    stale = stale or source_age + age > self.stale_after or source_age < -.05
                result[name] = dict(payload, age=round(age, 3), stale=bool(stale))
        cloud, pose = result['cloud'], result['pose']
        # No TF guessing and no overlay of mismatched-time/mismatched-frame poses.
        result['pose_overlay'] = bool(cloud and pose and not cloud['stale'] and not pose['stale']
                                      and cloud['frame'] == pose['frame']
                                      and cloud.get('stamp') is not None and pose.get('stamp') is not None
                                      and abs(cloud['stamp'] - pose['stamp']) <= .1)
        return result


def _offer_latest(output, item):
    try:
        output.put_nowait(item)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        try:
            output.put_nowait(item)
        except queue.Full:
            pass


def _video_worker(output, stop, interface, fps):
    # Spawn isolates the SDK's DDS runtime from ROS Foxy's DDS libraries.
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
        ChannelFactoryInitialize(0, interface)
        client = VideoClient()
        client.SetTimeout(1.)
        client.Init()
        while not stop.is_set():
            started = time.monotonic()
            try:
                code, data = client.GetImageSample()
                if code != 0:
                    raise RuntimeError(f'VideoClient.GetImageSample returned {code}')
                image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
                jpeg = jpeg_from_bgr(image)
                # SDK supplies no acquisition timestamp: label receipt time honestly.
                _offer_latest(output, (time.monotonic(), {'jpeg': jpeg, 'source': 'Go2 VideoClient (receipt time)',
                                                         'stamp': None, 'source_age': None}, None))
            except Exception as error:
                _offer_latest(output, (time.monotonic(), None, str(error)))
            stop.wait(max(0., 1 / fps - (time.monotonic() - started)))
    except Exception as error:
        _offer_latest(output, (time.monotonic(), None, f'{type(error).__name__}: {error}'))


class SdkVideo:
    def __init__(self, frames, interface, fps):
        self.frames = frames
        ctx = mp.get_context('spawn')
        self.output, self.stop = ctx.Queue(maxsize=1), ctx.Event()
        self.process = ctx.Process(target=_video_worker, args=(self.output, self.stop, interface, fps), daemon=True)
        self.process.start()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        while not self.stop.is_set():
            try:
                received, payload, error = self.output.get(timeout=.2)
            except queue.Empty:
                if not self.process.is_alive():
                    if 'rgb' not in self.frames.snapshot()['errors']:
                        self.frames.error('rgb', 'Camera worker stopped; check SDK installation and video service')
                    return
                continue
            if error:
                self.frames.error('rgb', error)
            else:
                self.frames.put('rgb', payload, received)

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.thread.join(timeout=1)
        self.output.close()
