"""Archive one run identity, raw observations, visual evidence, and event gaps."""
import base64
import json
from pathlib import Path
import signal
import threading
import time

import requests
import cv2
from deploy.visualization.cloud_image import render_cloud

from deploy.app.config import REPO
from deploy.app.session import SessionClient, decode_packet
from .recorder import Recorder


def save_visual(directory, frame, snapshot, scene):
    index = frame['frame_id']
    stem = directory / f'{index:07d}'
    # Numeric packet also preserves full occupancy masks/log odds, path/trajectory,
    # sampled cumulative cloud and exact raw observation. No pickle.
    stem.with_suffix('.npz').write_bytes(base64.b64decode(frame['packet']['npz'], validate=True))
    payload = frame['visual']
    metadata = {'frame_id': index, 'packet': frame['packet']['metadata'],
                'visual': {k: v for k, v in payload.items() if k not in ('cloud', 'rgb')},
                'rgb': None, 'map': {k: v for k, v in frame['map'].items() if k != 'png'}}
    if payload['rgb']:
        metadata['rgb'] = {k: v for k, v in payload['rgb'].items() if k != 'jpeg'}
        stem.with_suffix('.jpg').write_bytes(base64.b64decode(payload['rgb']['jpeg'], validate=True))
    if frame['map'].get('png'):
        stem.with_suffix('.png').write_bytes(base64.b64decode(frame['map']['png'], validate=True))
    points = scene['points'] if scene['ready'] else snapshot.observation['sensors']['lidar']['pointcloud']
    if not cv2.imwrite(str(stem.with_suffix('.cloud.png')), render_cloud(points, snapshot.observation['position'],
                      ground_relative_z=snapshot.geometry_calibrated)):
        raise OSError('Failed to save cloud image')
    stem.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False))


def record(url, output=None, wait=30.):
    client = SessionClient(url)
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    writer = None
    session_id = None
    session_mode = None
    frame_id = event_id = frame_gap = event_gap = 0
    producer_dropped = 0
    final = None
    error = None
    reason = 'operator_stop'
    deadline = time.monotonic()+wait
    try:
        print('等待 run.py；只记录，不控制机器人。Ctrl+C 停止记录。', flush=True)
        while not stop.is_set():
            try:
                batch = client.get(f'/api/record?frame={frame_id}&event={event_id}')
            except requests.RequestException as exc:
                if writer is None and time.monotonic() < deadline:
                    stop.wait(.25)
                    continue
                raise RuntimeError(f'run.py 数据源不可用：{exc}') from exc
            info = batch['info']
            if writer is None:
                session_id, session_mode = info['session_id'], info['mode']
                root = output or Path(info['config']['runtime']['recording']['root'])
                if not root.is_absolute():
                    root = REPO / root
                writer = Recorder(root, info['config'], info['mode'], REPO, source=info['provenance'])
                (writer.directory / 'session.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
                (writer.directory / 'frames').mkdir()
                print(f'记录会话 {session_id} → {writer.directory}', flush=True)
            if info['session_id'] != session_id:
                raise RuntimeError('run.py 已启动新会话；停止记录，避免混合两轮实验')
            frame_gap += batch['frame_gap']
            event_gap += batch['event_gap']
            producer_dropped = batch['producer_dropped']
            for event in batch['events']:
                writer.event(event['event'], **{k: v for k, v in event.items() if k != 'event'})
                event_id = event['event_id']
            for frame in batch['frames']:
                snapshot, scene = decode_packet(frame['packet'])
                writer.observation(snapshot)
                save_visual(writer.directory / 'frames', frame, snapshot, scene)
                frame_id = frame['frame_id']
            if batch['summary'] is not None:
                final, reason = batch['summary'], 'run_finished'
                break
            stop.wait(.2)
    except (RuntimeError, ValueError, OSError, KeyError, cv2.error) as exc:
        error, reason = str(exc), 'record_error'
        print(error, flush=True)
    finally:
        drained = (final is not None and final.get('telemetry_complete', True)
                   and frame_id == final.get('final_frame_id', frame_id)
                   and event_id == final.get('final_event_id', event_id))
        has_data = frame_id > 0 or (session_mode == 'motion_test' and event_id > 0)
        summary = {'reason': reason, 'error': error, 'completed': error is None,
                   'session_id': session_id, 'run_summary': final, 'run_final_received': final is not None,
                   'missing_frames': frame_gap, 'missing_events': event_gap,
                   'producer_dropped': producer_dropped, 'last_frame': frame_id, 'last_event': event_id,
                   'recording_complete': error is None and drained and has_data and frame_gap == event_gap == producer_dropped == 0,
                   'hardware_validated': False}
        try:
            if writer:
                writer.close(summary)
                print(f'记录结束：{writer.directory}；导航状态不受 record.py 退出影响。', flush=True)
        finally:
            client.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return 2 if error else 0
