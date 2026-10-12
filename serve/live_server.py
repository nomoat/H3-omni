#!/usr/bin/env python3
"""H3-omni finite live service: twelve connected parts as fragmented MP4."""
import argparse
import hmac
import json
import os
import secrets
import select
import socket
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from server import Worker, SIZES, MAX_PROMPT

PARTS = 12
DURATION = (5 + 119 * PARTS) / 24
CODECS = 'avc1.42C01F, mp4a.40.2'


class LiveWorker(Worker):
    def warm_up(self, work_dir):
        self.loaded.wait()
        output = os.path.join(work_dir, 'warmup.mp4')
        try:
            self.generate('A quiet lake at sunrise, gentle waves and birds.',
                          *SIZES['landscape'], -1, 0, output)
            self.ready.set()
            print('live model ready', flush=True)
        except (RuntimeError, ValueError, OSError) as error:
            print(f'live warm-up failed: {error}', flush=True)
            self.process.kill()
        finally:
            Path(output).unlink(missing_ok=True)


def make_live_handler(worker, key, work_dir, gpus):
    occupied = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'H3-omni'

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def disconnected(self):
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable and self.connection.recv(1, socket.MSG_PEEK) == b'')

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': {'message': 'not found'}})
            ready = worker.ready.is_set() and not worker.dead.is_set()
            self.reply(200 if ready else 503, {'status': 'ready' if ready else 'loading',
                       'model': 'H3-omni Live', 'gpus': gpus, 'parts': PARTS,
                       'seconds': DURATION, 'streaming': True})

        def do_POST(self):
            if self.path != '/v1/live':
                self.close_connection = True
                return self.reply(404, {'error': {'message': 'not found'}})
            if not hmac.compare_digest(self.headers.get('Authorization', '').encode(),
                                       ('Bearer ' + key).encode()):
                self.close_connection = True
                return self.reply(401, {'error': {'message': 'invalid key'}})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('invalid request length')
                body = json.loads(self.rfile.read(length))
                prompt = body.get('prompt', '')
                if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= MAX_PROMPT:
                    raise ValueError(f'prompt must be 1-{MAX_PROMPT} characters')
                if body.get('parts', PARTS) != PARTS:
                    raise ValueError('live sessions contain exactly 12 parts')
                seed = body.get('seed', secrets.randbelow(2**31))
                if type(seed) is not int or not 0 <= seed < 2**63:
                    raise ValueError('invalid seed')
                width, height = SIZES[body.get('orientation', 'landscape')]
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self.close_connection = True
                return self.reply(400, {'error': {'message': str(error)}})
            if not worker.ready.is_set() or worker.dead.is_set():
                return self.reply(503, {'error': {'message': 'live model is loading'}})
            if not occupied.acquire(blocking=False):
                return self.reply(429, {'error': {'message': 'live model is busy'}})
            output = Path(work_dir) / (uuid.uuid4().hex + '.mp4')
            cancel = Path(str(output) + '.cancel')
            finished = threading.Event()
            reader_done = threading.Event()
            result = {}

            def generate():
                try:
                    result['seconds'] = worker.generate(prompt.strip(), width, height,
                                                        -PARTS, seed, str(output))
                except (RuntimeError, ValueError, OSError) as error:
                    result['error'] = str(error)
                finally:
                    finished.set()
                    reader_done.wait()
                    output.unlink(missing_ok=True)
                    cancel.unlink(missing_ok=True)
                    occupied.release()

            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            began = time.monotonic()
            started = False
            try:
                self.connection.settimeout(30)
                while not output.exists() or output.stat().st_size == 0:
                    if self.disconnected():
                        raise ConnectionError('client disconnected')
                    if finished.wait(0.1):
                        if 'error' in result:
                            return self.reply(503, {'error': {'message': result['error']}})
                        if not output.exists():
                            raise RuntimeError('worker returned no live video')
                    if time.monotonic() - began > 600:
                        raise TimeoutError('live startup timed out')
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Transfer-Encoding', 'chunked')
                self.send_header('Cache-Control', 'no-store, no-transform')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('X-Video-Seed', str(seed))
                self.send_header('X-Video-Parts', str(PARTS))
                self.send_header('X-Video-Duration', str(DURATION))
                self.send_header('X-Video-Codecs', CODECS)
                self.end_headers()
                started = True
                with output.open('rb') as media:
                    while True:
                        data = media.read(64 * 1024)
                        if data:
                            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                            self.wfile.flush()
                        elif finished.is_set():
                            if 'error' in result:
                                raise RuntimeError(result['error'])
                            self.wfile.write(b'0\r\n\r\n')
                            self.wfile.flush()
                            break
                        else:
                            if self.disconnected():
                                raise ConnectionError('client disconnected')
                            if time.monotonic() - began > 600:
                                raise TimeoutError('live generation timed out')
                            finished.wait(0.1)
            except (OSError, RuntimeError) as error:
                self.close_connection = True
                if not started:
                    try:
                        self.reply(503, {'error': {'message': 'live stream failed'}})
                    except OSError:
                        pass
                print(f'live stream ended early: {type(error).__name__}', flush=True)
            finally:
                if not finished.is_set():
                    cancel.touch()
                reader_done.set()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--port', type=int, default=18961)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()
    key = Path(args.key_file).read_text().strip()
    if len(key) < 32:
        parser.error('key must contain at least 32 characters')
    with tempfile.TemporaryDirectory(prefix='h3-live-') as directory:
        worker = LiveWorker(args.worker, args.model, args.gpus, 3, args.log)
        server = ThreadingHTTPServer((args.host, args.port),
                                    make_live_handler(worker, key, directory, args.gpus))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Thread(target=worker.warm_up, args=(directory,), daemon=True).start()
        print(f'live server listening on {args.host}:{args.port}', flush=True)
        worker.dead.wait()
        server.shutdown()
        raise SystemExit('live worker exited')


if __name__ == '__main__':
    main()
