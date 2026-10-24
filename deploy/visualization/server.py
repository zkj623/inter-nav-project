"""Loopback-only HTTP viewer, accessible from a laptop using SSH forwarding."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
from urllib.parse import urlsplit

from .sources import DemoSource, LatestFrames, RosViewer, SdkVideo


def make_server(frames, port, metadata):
    page = Path(__file__).with_name('index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                body, mime = page, 'text/html; charset=utf-8'
            elif path == '/api/frame':
                body = json.dumps(dict(frames.snapshot(), **metadata), allow_nan=False).encode()
                mime = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
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


def serve(config, args, rgb_topic):
    frames = LatestFrames(args.max_points, max(config['max_data_age_s'], 2 / args.fps), args.demo)
    metadata = {'fps': args.fps, 'cloud_topic': 'synthetic/demo_cloud' if args.demo else config['lidar_topic'],
                'rgb_source': 'synthetic' if args.demo else (rgb_topic if args.camera == 'ros' else args.camera)}
    server = make_server(frames, args.port, metadata)
    resources, handlers = [], {}
    stop = threading.Event()
    server.timeout = .2
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, lambda *_: stop.set())
        if args.demo:
            resources.append(DemoSource(frames, args.fps))
        else:
            resources.append(RosViewer(frames, config, args.fps, rgb_topic if args.camera == 'ros' else None))
            if args.camera == 'sdk':
                resources.append(SdkVideo(frames, config['network_interface'], args.fps))
            elif args.camera == 'none':
                frames.error('rgb', 'RGB disabled (--camera none)')
        # rclpy.init may install its own SIGINT handler; keep viewer shutdown in charge.
        for signum in handlers:
            signal.signal(signum, lambda *_: stop.set())
        print(f'可视化：http://127.0.0.1:{args.port}  [只读；Ctrl+C 关闭界面服务]', flush=True)
        print(f'笔记本 SSH 转发：ssh -N -L {args.port}:127.0.0.1:{args.port} <用户>@<Go2地址>', flush=True)
        print('本程序不控制运动，退出它不会停止另一个终端的 run.py。', flush=True)
        if args.demo:
            print('演示数据：未连接机器人。', flush=True)
        while not stop.is_set():
            server.handle_request()
    finally:
        server.server_close()
        try:
            for resource in reversed(resources):
                try:
                    resource.close()
                except Exception as error:
                    print(f'可视化清理异常：{error}', flush=True)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return 0
