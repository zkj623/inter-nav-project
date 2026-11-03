"""run.py owns one bounded, read-only experiment feed; view/record are clients."""
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import io
import json
import multiprocessing as mp
import queue
import threading
import time
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4

import numpy as np
import requests

from deploy.recording.recorder import pack, unpack, json_value, provenance
from deploy.robot.observation import Snapshot
from deploy.visualization.scene import SceneCapture
from deploy.visualization.worker import process_frames
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
    def __init__(self, config, repo, port=8765, demo=False, mode='live_observe_only', frame_limit=4, event_limit=4096,
                 separate_cloud=False):
        self.info = {'session_id': uuid4().hex, 'started_at': datetime.now(timezone.utc).isoformat(),
                     'config': config, 'provenance': provenance(repo), 'mode': mode, 'demo': demo,
                     'separate_cloud': separate_cloud}
        self.lock = threading.Lock()
        self.frames = LatestFrames(stale_after=1., demo=demo)
        self.separate_cloud = separate_cloud
        self.capture = SceneCapture(include_cloud=not separate_cloud)
        context = mp.get_context('spawn')
        self.pending = context.Queue(maxsize=4)
        self.rendered, output = context.Pipe(duplex=False)
        self.renderer = context.Process(target=process_frames,
                                        args=(self.pending, output, config['navigation']['mapping'], separate_cloud),
                                        name='go2-visualization', daemon=True)
        self.last_scan = None
        self.cloud_input_dropped = 0
        self.render_failure = None
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
        self.server = self._server(port)  # Bind BEFORE ROS/SDK/motion startup.
        try:
            self.renderer.start()
            output.close()  # One writer: child death must deliver EOF to the collector.
            if not self.rendered.poll(10.):
                raise RuntimeError('Visualization process did not initialize')
            ready = self.rendered.recv()
            if not ready.get('ready'):
                raise RuntimeError(f'Visualization startup failed: {ready}')
        except BaseException:
            output.close()
            self._stop_renderer()
            self.server.server_close()
            self._close_channels()
            raise
        self.worker = threading.Thread(target=self._collect_frames, daemon=True)
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

    def update(self, snapshot, tasks, motion, error=None, *, decision_duration_s=None):
        if not self.set_status(tasks.state, tasks.mode, motion, error=error,
                               map_ready=tasks.map_ready, result=tasks.result):
            return
        now = time.monotonic()
        with self.lock:
            if self.closing:
                return
            self.input_times = (snapshot.pose_at, snapshot.lidar_at) if snapshot is not None else None
        if snapshot is None:
            return
        lidar = snapshot.observation['sensors']['lidar']
        scan = None
        if (self.separate_cloud and snapshot.geometry_calibrated
                and lidar['physics_step'] == tasks.last_lidar and lidar['physics_step'] != self.last_scan):
            self.last_scan = lidar['physics_step']
            scan = dict(lidar, captured_at=snapshot.lidar_at, step=snapshot.sequence)
        publish = now-self.last_publish >= self.publish_interval
        if not publish and scan is None:
            return
        try:
            # Copy the mutable navigation map on its owner; 3D work stays in
            # the display process. Every accepted scan is offered, not just 5 Hz frames.
            scene = self.capture.capture(snapshot, tasks) if publish else None
            if publish:
                self.last_publish = now
                scene['status'] = dict(self.status)
                scene['decision_duration_s'] = decision_duration_s
            captured = time.monotonic()
            packet = dict(scan=scan, snapshot=snapshot if publish else None, scene=scene,
                          queued_at=captured, capture_ms=(captured-now)*1000)
            with self.lock:
                if not self.closing:
                    if self.render_failure:
                        self._discard(packet)
                        return
                    try:
                        self.pending.put_nowait(packet)
                    except queue.Full:
                        # Even get_nowait can wait for a partially serialized
                        # pipe payload. Only the display process consumes this
                        # bounded queue; navigation never drains it to make room.
                        self._discard(packet)
                        return
                    self.unpublished_frames += int(publish)
        except Exception as exc:
            with self.lock:
                if not self.publication_closed:
                    self.frames.error('session', exc)
                    self.dropped += int(publish)
                    self.cloud_input_dropped += int(scan is not None)

    def _discard(self, packet):
        # Caller holds the session lock. Scan loss and published-frame loss
        # have separate meanings for a recorded experiment.
        frame = packet['scene'] is not None
        self.dropped += int(frame)
        self.cloud_input_dropped += int(packet['scan'] is not None)

    def rgb(self, rgba, captured_at=None, source='ROS RGB'):
        payload = {'jpeg': jpeg_from_bgr(rgba[..., :3][..., ::-1]),
                   'source': source, 'stamp': captured_at, 'source_age': None,
                   'timestamp_basis': 'monotonic_capture_estimate'}
        with self.lock:
            if not self.publication_closed:
                self.frames.put('rgb', payload, captured_at)

    def _collect_frames(self):
        while not self.publication_closed:
            try:
                if not self.rendered.poll(.1):
                    continue
                result = self.rendered.recv()
            except (EOFError, OSError) as error:
                if not self.publication_closed:
                    self.render_failure = f'Visualization process disconnected: {type(error).__name__}'
                    self.frames.error('session', self.render_failure)
                    self.event('visualization_error', reason=self.render_failure)
                return
            if result is None:
                return
            if 'error' in result:
                with self.lock:
                    if self.publication_closed:
                        return
                    self.dropped += int(result['frame_dropped'])
                    self.unpublished_frames -= int(result['frame_dropped'])
                    self.cloud_input_dropped += int(result['cloud_failed'])
                    self.frames.error('session', result['error'])
                self.event('visualization_error', reason=result['error'])
                continue
            try:
                self._publish_frame(result)
            except Exception as error:
                if not self.publication_closed:
                    self.render_failure = f'Visualization publication failed: {error}'
                    self.frames.error('session', self.render_failure)
                    self.event('visualization_error', reason=self.render_failure)
                return

    def _publish_frame(self, result):
        snapshot, scene = result['snapshot'], result['scene']
        with self.lock:
            if self.publication_closed:
                return
            self.frames.put('cloud', result['cloud'], result['cloud']['stamp'])
            self.frames.put('pose', result['pose'], snapshot.pose_at)
            if result['rgb'] is not None:
                self.frames.put('rgb', result['rgb'], snapshot.camera_at)
            self.map, self.map_at = result['map'], scene['updated_at'] if scene['ready'] else snapshot.lidar_at
            self.published_at = time.monotonic()
            self.publish_times.append(self.published_at)
            self.timings = dict(result['timings'], delivery_ms=(self.published_at-result['finished_at'])*1000)
            scene['performance'] = dict(self.timings)
            scene['cloud_input_dropped'] = self.cloud_input_dropped
            self.unpublished_frames -= 1
            self.frame_id += 1
            self.frames_ring.append(_RecordedFrame(
                self.frame_id, snapshot, scene, self.frames.snapshot(), self.map))
            with self.frames.lock:
                self.frames.errors.pop('session', None)

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
                               cloud_input_dropped=self.cloud_input_dropped,
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
                    'producer_dropped': self.dropped, 'cloud_input_dropped': self.cloud_input_dropped,
                    'summary': self.summary}
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
        deadline = time.monotonic()+drain_timeout
        try:
            self.pending.put(None, timeout=max(0., deadline-time.monotonic()))
        except queue.Full:
            pass
        self.worker.join(timeout=max(0., deadline-time.monotonic()))
        self.renderer.join(timeout=max(0., deadline-time.monotonic()))
        with self.lock:
            drained = (not self.worker.is_alive() and self.renderer.exitcode == 0
                       and self.render_failure is None and self.unpublished_frames == 0)
            self.publication_closed = True
            if not drained:
                self.dropped += self.unpublished_frames
                self.unpublished_frames = 0
                summary = dict(summary, completed=False, telemetry_error='encoder did not finish')
            self.summary = dict(summary, telemetry_complete=drained,
                                cloud_input_dropped=self.cloud_input_dropped,
                                cloud_history_complete=drained and self.cloud_input_dropped == 0,
                                final_frame_id=self.frame_id, final_event_id=self.event_id+1)
            self._append_event({'event': 'session_finished', 'at': time.monotonic(), 'summary': self.summary})
        self._stop_renderer()
        self.worker.join(timeout=1)
        self._close_channels()
        # This is a bounded read-only drain window, not a guarantee for disconnected
        # or slow recorders. Their final cursors and gap counts disclose any loss.
        time.sleep(grace)
        self.server.shutdown()
        self.http.join(timeout=1)
        self.server.server_close()

    def _close_channels(self):
        self.pending.cancel_join_thread()
        self.pending.close()
        self.rendered.close()

    def _stop_renderer(self):
        if self.renderer.pid is None:
            return
        if self.renderer.is_alive():
            self.renderer.terminate()
        self.renderer.join(timeout=1)
        if self.renderer.is_alive():
            self.renderer.kill()
            self.renderer.join(timeout=1)


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
