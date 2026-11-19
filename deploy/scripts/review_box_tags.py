"""Read an ongoing or finished deploy recording and inspect tag visibility and LiDAR fits."""
import argparse
import base64
from bisect import bisect_left
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np


class Recording:
    def __init__(self, directory):
        self.directory = directory
        self.event_offset = 0
        self.fits = {}
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(dictionary)

    def read_events(self):
        path = self.directory / 'events.jsonl'
        with path.open('rb') as stream:
            stream.seek(self.event_offset)
            for line in stream:
                if not line.endswith(b'\n'):
                    break  # record.py is still writing this line.
                event = json.loads(line)
                if event.get('event') == 'perception_result' and 'scan_id' in event:
                    key = (int(event['scan_id']), event.get('captured_at'))
                    self.fits[key] = event['detections']
                self.event_offset = stream.tell()

    def photo(self, path, rgb):
        if not rgb or not path.is_file():
            return '<p>该帧没有 RGB 照片。</p>'
        image = cv2.imread(str(path))
        if image is None:
            return '<p>该帧照片暂不可读。</p>'
        corners, ids, _ = self.detector.detectMarkers(image)
        labels = []
        if ids is not None:
            for quad, tag in zip(corners, ids.flatten()):
                if int(tag) not in (0, 1):
                    continue
                xy = quad.reshape(4, 2)
                shortest = np.linalg.norm(xy - np.roll(xy, 1, axis=0), axis=1).min()
                cv2.aruco.drawDetectedMarkers(image, [quad], np.array([[tag]], dtype=np.int32))
                labels.append(f'ID {tag} · 最短边 {shortest:.0f} px')
        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            raise ValueError('照片编码失败')
        data = base64.b64encode(encoded).decode('ascii')
        status = '，'.join(labels) if labels else '未检出 ID 0 或 1'
        age = rgb.get('age')
        if isinstance(age, (int, float)):
            status += f'。录制时距最近 RGB 收包 {age:.2f} s'
        if rgb.get('stale'):
            status += '（历史画面已过期）'
        return f'<p>{status}</p><img alt="前视图与标签四角" src="data:image/jpeg;base64,{data}">'

    def panel(self, path):
        frame = json.loads(path.read_text())
        snapshot = frame['packet']['snapshot']
        scan = snapshot['observation']['sensors']['lidar']['physics_step']
        rows = self.fits.get((int(scan), snapshot['lidar_at']), [])
        lines = []
        for row in rows:
            label = escape(str(row['object_id']))
            if 'tracking_state' in row:
                state = escape(str(row['tracking_state']))
                track = escape(str(row.get('track_id') or '—'))
                label += f' · 1 秒窗口 · {state} · 轨迹 {track}'
            if row.get('center') is None or not row.get('valid'):
                reason = escape(str(row.get('invalid_reason') or '未确认'))
                lines.append(f'<p>{label} · {reason}</p>')
                continue
            x, y, _ = row['center']
            if 'face_angle_rad' in row:
                angle = math.degrees(row['face_angle_rad'])
                lines.append(f'<p>{label} · 中心 ({x:.3f}, {y:.3f}) m · 箱面方向 {angle:.1f}°（模 90°）</p>')
                continue
            angle = math.degrees(row['yaw_rad'])
            faces = escape(', '.join(row['visible_faces']))
            rays = escape(str(row.get('boundary_free_ray_counts', [])))
            residual = row['fit_residual_m'] * 100
            lines.append(f'<p>{label} · 本次扫描拟合有效 · 中心 ({x:.3f}, {y:.3f}) m · '
                         f'朝向 {angle:.1f}° · 可见面 {faces} · 边缘射线 {rays} · 残差 {residual:.2f} cm</p>')
        fit = ''.join(lines) or '<p>该扫描没有对应的登记箱体结果。</p>'
        return (f'<h2>LiDAR 扫描 {scan}</h2>{fit}<h2>前视相机</h2>'
                + self.photo(path.with_suffix('.jpg'), frame.get('rgb')))

    def page(self, requested):
        self.read_events()
        frames = sorted(int(p.stem) for p in (self.directory / 'frames').glob('*.json') if p.stem.isdigit())
        live = requested is None and not (self.directory / 'run_summary.json').exists()
        refresh = '<meta http-equiv="refresh" content="1">' if live else ''
        content = '<p>等待 record.py 写入显示帧。</p>'
        if frames:
            index = len(frames) - 1 if requested is None else min(bisect_left(frames, requested), len(frames) - 1)
            number = frames[index]
            nav = (f'<nav><a href="/?frame={frames[max(0, index-1)]}">上一帧</a>'
                   f'<a href="/?frame={number}">暂停在此帧</a>'
                   f'<a href="/?frame={frames[min(index+1, len(frames)-1)]}">下一帧</a>'
                   f'<a href="/">跟随最新</a></nav><form>记录帧 <input type="number" name="frame" '
                   f'value="{number}" min="{frames[0]}" max="{frames[-1]}"> / {frames[-1]} <button>跳转</button></form>')
            try:
                content = nav + self.panel(self.directory / 'frames' / f'{number:07d}.json')
            except (json.JSONDecodeError, FileNotFoundError):
                content = nav + '<p>该帧仍在写入，请刷新。</p>'
        return (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>箱体观测回看</title><style>body{font:16px system-ui;margin:24px auto;padding:0 16px;'
                'max-width:900px}nav{display:flex;gap:18px;flex-wrap:wrap}form{margin:16px 0}img{max-width:100%}'
                'h2{font-size:18px}input{width:7em}</style><h1>箱体观测回看</h1>'
                '<p>LiDAR 按原始扫描匹配。SDK 照片按接收时间展示，多张记录帧可能共用一张照片。</p>'
                + content + '</html>').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path, help='record.py 打印的单次记录目录')
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    if not (args.recording / 'events.jsonl').is_file() or not (args.recording / 'frames').is_dir():
        parser.error('需要包含 events.jsonl 和 frames/ 的单次记录目录')
    if not 1 <= args.port <= 65535:
        parser.error('端口应为 1 到 65535')
    recording = Recording(args.recording)
    reader_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            target = urlsplit(self.path)
            if target.path != '/':
                self.send_error(404)
                return
            value = parse_qs(target.query).get('frame')
            try:
                requested = int(value[0]) if value else None
            except ValueError:
                self.send_error(400, 'Invalid frame')
                return
            with reader_lock:
                body = recording.page(requested)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'箱体观测回看 http://127.0.0.1:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
