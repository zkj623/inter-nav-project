"""run.py owns one bounded, read-only experiment feed; view/record are clients."""
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import io
import json
import queue
import threading
import time
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4

import numpy as np
import requests

from deploy.recording.recorder import pack, unpack, json_value, provenance
from deploy.robot.observation import Snapshot
from deploy.visualization.scene import SceneCapture, map_payload
from deploy.visualization.sources import LatestFrames, jpeg_from_bgr


def encode_packet(snapshot, scene):
    arrays = {}
    metadata = pack({'snapshot': asdict(snapshot), 'scene': scene}, arrays)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return {'metadata': metadata, 'npz': base64.b64encode(output.getvalue()).decode()}


def decode_packet(packet):
    with np.load(io.BytesIO(base64.b64decode(packet['npz'], validate=True)), allow_pickle=False) as arrays:
        data = unpack(packet['metadata'], arrays)
    return Snapshot(**data['snapshot']), data['scene']


class _RecordedFrame:
    """Detached frame with a single lazy compression shared by all recorders."""
    def __init__(self, frame_id, snapshot, scene, visual, mapping):
        self.frame_id = frame_id
        self.snapshot, self.scene = snapshot, scene
        self.visual, self.mapping = visual, mapping
        self._packet = None
        self._lock = threading.Lock()

    def payload(self):
        # Compression never holds Session.lock or blocks display publication.
        with self._lock:
            if self._packet is None:
                self._packet = encode_packet(self.snapshot, self.scene)
            packet = self._packet
        return {'frame_id': self.frame_id, 'packet': packet,
                'visual': self.visual, 'map': self.mapping}


class Session:
    def __init__(self, config, repo, port=8765, demo=False, mode='live_observe_only', frame_limit=4, event_limit=4096):
        self.info = {'session_id': uuid4().hex, 'started_at': datetime.now(timezone.utc).isoformat(),
                     'config': config, 'provenance': provenance(repo), 'mode': mode, 'demo': demo}
        self.lock = threading.Lock()
        self.frames = LatestFrames(stale_after=1., demo=demo)
        self.capture = SceneCapture()
        self.pending = queue.Queue(maxsize=1)
        self.frames_ring, self.events_ring = deque(maxlen=frame_limit), deque(maxlen=event_limit)
        self.frame_id = self.event_id = self.dropped = 0
        self.unpublished_frames = 0
        self.last_publish = float('-inf')
        self.publish_interval = .2
        self.input_times = None
        self.published_at = None
        self.publish_times = deque(maxlen=30)
        self.timings = {}
        self.map = {'ready': False, 'reason': '等待 run.py 的有效观测和标定'}
        self.map_at = None
        self.status = {'state': 'starting', 'motion': 'disabled', 'mode': 'observe'}
        self.heartbeat = time.monotonic()
        self.summary = None
        self.closing = False
        self.publication_closed = False
        self.worker_stop = threading.Event()
        self.server = self._server(port)  # Bind BEFORE ROS/SDK/motion startup.
        self.worker = threading.Thread(target=self._encode_loop, daemon=True)
        self.http = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .1}, daemon=True)
        self.worker.start()
        self.http.start()
        self.event('session_started', session_id=self.info['session_id'])

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}'

    def event(self, kind, **details):
        # Events contain only small metadata, never point/image arrays.
        item = json.loads(json.dumps({'event': kind, 'at': time.monotonic(), **details}, default=json_value, allow_nan=False))
        with self.lock:
            if self.summary is None:
                self._append_event(item)

    def _append_event(self, item):
        # Caller holds Session.lock, including the atomic final event/summary.
        self.event_id += 1
        self.events_ring.append(dict(item, event_id=self.event_id))

    def set_status(self, state, mode, motion, *, error=None, map_ready=False, result=None):
        """Publish activity independently of mapping, including motion acceptance."""
        with self.lock:
            if self.closing:
                return False
            self.heartbeat = time.monotonic()
            self.status = {'state': state, 'mode': mode, 'motion': motion, 'error': error,
                           'map_ready': map_ready, 'result': result}
        return True

    def update(self, snapshot, tasks, motion, error=None):
        if not self.set_status(tasks.state, tasks.mode, motion, error=error,
                               map_ready=tasks.map_ready, result=tasks.result):
            return
        now = time.monotonic()
        with self.lock:
            if self.closing:
                return
            self.input_times = (snapshot.pose_at, snapshot.lidar_at) if snapshot is not None else None
        if snapshot is None or now-self.last_publish < self.publish_interval:
            return
        self.last_publish = now
        with self.lock:
            if self.pending.full():
                try:
                    self.pending.get_nowait()
                    self.unpublished_frames -= 1
                    self.dropped += 1
                except queue.Empty:
                    pass
        try:
            # Mutable planner arrays are copied only on their owning thread.
            scene = self.capture.capture(snapshot, tasks)
            captured = time.monotonic()
            with self.lock:
                if not self.closing:
                    self.pending.put_nowait((snapshot, scene, captured, (captured-now)*1000))
                    self.unpublished_frames += 1
        except Exception as exc:
            with self.lock:
                if not self.publication_closed:
                    self.frames.error('session', exc)
                    self.dropped += 1

    def rgb(self, rgba, captured_at=None, source='ROS RGB'):
        payload = {'jpeg': jpeg_from_bgr(rgba[..., :3][..., ::-1]),
                   'source': source, 'stamp': captured_at, 'source_age': None,
                   'timestamp_basis': 'monotonic_capture_estimate'}
        with self.lock:
            if not self.publication_closed:
                self.frames.put('rgb', payload, captured_at)

    def _encode_loop(self):
        while not self.worker_stop.is_set() or not self.pending.empty():
            try:
                snapshot, scene, queued_at, capture_ms = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                encoding_started = time.monotonic()
                map_image = map_payload(scene)
                lidar = snapshot.observation['sensors']['lidar']
                points = scene['points'] if scene['ready'] else lidar['pointcloud']
                if len(points) > 12000:
                    points = points[np.linspace(0, len(points)-1, 12000, dtype=int)]
                display_count = scene['voxel_count'] if scene['ready'] else len(lidar['pointcloud'])
                map_at = scene['updated_at'] if scene['ready'] else snapshot.lidar_at
                cloud_payload = {'points': np.round(np.asarray(points, dtype=float), 3).tolist(),
                                         'count': display_count, 'frame': snapshot.planning_frame,
                                         'display_count': len(points), 'accumulated': bool(scene['ready']),
                                         'stamp': map_at, 'source_age': None,
                                         'ground_relative_z': snapshot.geometry_calibrated,
                                         'kind': '累计建图点云' if scene['ready'] else '当前原始点云（未建图）'}
                w, x, y, z = snapshot.observation['orientation']
                pose_payload = {'position': snapshot.observation['position'].tolist(),
                                        'yaw': float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                                        'frame': snapshot.planning_frame, 'stamp': snapshot.pose_at,
                                        'source_age': None}
                camera = snapshot.observation['sensors'].get('camera')
                if camera is not None:
                    self.rgb(camera['rgba'], snapshot.camera_at, 'ROS RGB-D colour')
                with self.lock:
                    if self.publication_closed:
                        return
                    self.frames.put('cloud', cloud_payload, map_at)
                    self.frames.put('pose', pose_payload, snapshot.pose_at)
                    self.map, self.map_at = map_image, map_at
                    self.published_at = time.monotonic()
                    self.publish_times.append(self.published_at)
                    self.timings = {'capture_ms': capture_ms, 'queue_ms': (encoding_started-queued_at)*1000,
                                    'encode_ms': (self.published_at-encoding_started)*1000}
                    self.unpublished_frames -= 1
                    self.frame_id += 1
                    self.frames_ring.append(_RecordedFrame(
                        self.frame_id, snapshot, scene, self.frames.snapshot(), map_image))
                with self.frames.lock:
                    self.frames.errors.pop('session', None)
            except Exception as error:
                with self.lock:
                    if self.publication_closed:
                        return
                    self.frames.error('session', error)
                    self.unpublished_frames -= 1
                    self.dropped += 1

    def display(self):
        now = time.monotonic()
        with self.lock:
            mapping = dict(self.map, age=None if self.map_at is None else max(0., now-self.map_at),
                           stale=self.map_at is None or now-self.map_at > 1.)
            state = dict(self.status)
            age = now-self.heartbeat
            finished = self.summary is not None
            sequence = self.frame_id
            frames = self.frames.snapshot()
            input_ages = None if self.input_times is None else [now-t for t in self.input_times]
            input_stale = input_ages is None or any(t > self.info['config']['robot']['max_data_age_s'] or t < -.05 for t in input_ages)
            measured_fps = ((len(self.publish_times)-1)/(self.publish_times[-1]-self.publish_times[0])
                            if len(self.publish_times) > 1 else 0.)
            performance = dict(self.timings, display_fps=measured_fps, input_ages_s=input_ages,
                               input_stale=input_stale,
                               display_age_s=None if self.published_at is None else now-self.published_at)
        return dict(frames, session_id=self.info['session_id'], sequence=sequence,
                    map=mapping, status=state, session_age=age, finished=finished, fps=5., performance=performance,
                    cloud_topic=self.info['config']['robot']['lidar_topic'])

    def batch(self, after_frame, after_event):
        with self.lock:
            oldest_frame = self.frames_ring[0].frame_id if self.frames_ring else self.frame_id+1
            oldest_event = self.events_ring[0]['event_id'] if self.events_ring else self.event_id+1
            result = {'info': self.info, 'frames': [p for p in self.frames_ring if p.frame_id > after_frame],
                    'events': [p for p in self.events_ring if p['event_id'] > after_event],
                    'frame_gap': max(0, oldest_frame-after_frame-1),
                    'event_gap': max(0, oldest_event-after_event-1),
                    'producer_dropped': self.dropped, 'summary': self.summary}
        # Recording compression is demand-driven and outside the display lock.
        # An unused recorder must not delay every live visual frame.
        return dict(result, frames=[frame.payload() for frame in result['frames']])

    def _server(self, port):
        session = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlsplit(self.path)
                try:
                    if parsed.path == '/api/frame':
                        payload = session.display()
                    elif parsed.path == '/api/record':
                        params = parse_qs(parsed.query)
                        frame, event = int(params.get('frame', ['0'])[0]), int(params.get('event', ['0'])[0])
                        if min(frame, event) < 0:
                            raise ValueError('negative cursor')
                        payload = session.batch(frame, event)
                    else:
                        self.send_error(404)
                        return
                    body = json.dumps(payload, default=json_value, allow_nan=False, separators=(',', ':')).encode()
                except (ValueError, TypeError):
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        server.daemon_threads = True
        return server

    def close(self, summary, grace=2., drain_timeout=3.):
        # Called only AFTER motor shutdown. Drain accepted frames before declaring
        # a final cursor; a timed-out worker must never publish after that boundary.
        with self.lock:
            if self.closing:
                return
            self.closing = True
        self.worker_stop.set()
        self.worker.join(timeout=drain_timeout)
        with self.lock:
            drained = not self.worker.is_alive()
            self.publication_closed = True
            if not drained:
                self.dropped += self.unpublished_frames
                self.unpublished_frames = 0
                summary = dict(summary, completed=False, telemetry_error='encoder did not finish')
            self.summary = dict(summary, telemetry_complete=drained,
                                final_frame_id=self.frame_id, final_event_id=self.event_id+1)
            self._append_event({'event': 'session_finished', 'at': time.monotonic(), 'summary': self.summary})
        # This is a bounded read-only drain window, not a guarantee for disconnected
        # or slow recorders. Their final cursors and gap counts disclose any loss.
        time.sleep(grace)
        self.server.shutdown()
        self.http.join(timeout=1)
        self.server.server_close()


class SessionClient:
    def __init__(self, url='http://127.0.0.1:8765'):
        self.url = url.rstrip('/')
        self.http = requests.Session()
        self.http.trust_env = False

    def get(self, path):
        response = self.http.get(self.url+path, timeout=2.)
        response.raise_for_status()
        return response.json()

    def snapshot(self):
        return self.get('/api/frame')

    def close(self):
        self.http.close()
