"""Loopback-only HTTP viewer, accessible from a laptop using SSH forwarding."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
from urllib.parse import urlsplit

from deploy.app.session import SessionClient
import requests


def make_server(frames, port, metadata=None):
    page = Path(__file__).with_name('index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                body, mime = page, 'text/html; charset=utf-8'
            elif path == '/api/frame':
                try:
                    body = json.dumps(dict(frames.snapshot(), **(metadata or {})), allow_nan=False).encode()
                except (requests.RequestException, ValueError):
                    self.send_error(503, 'run.py unavailable; start run.py then reconnect')
                    return
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


def serve(args):
    client = SessionClient(args.source)
    server = make_server(client, args.port)
    handlers = {}
    stop = threading.Event()
    server.timeout = .2
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, lambda *_: stop.set())
        print(f'三图界面：http://127.0.0.1:{args.port}；数据来自 {args.source}', flush=True)
        print('请先启动 run.py；view.py 不采集、不建图、不控制运动。', flush=True)
        print(f'笔记本 SSH 转发：ssh -L {args.port}:127.0.0.1:{args.port} go2', flush=True)
        while not stop.is_set():
            server.handle_request()
    finally:
        server.server_close()
        client.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0
