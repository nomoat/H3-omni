#!/usr/bin/env python3
"""HTTP front end for the resident h3 worker (serve/h3_worker.c).

One worker process holds the model on all GPUs of the job and runs one
request at a time; requests wait in a short queue. Python standard library
only, so it runs in the CUDA container without extra packages.

  GET  /health      200 {"status":"ready",...} or 503 while loading or warming up
  POST /v1/videos   JSON {"prompt": str, "seconds": 1-10, "seed": int?,
                          "orientation": "landscape"|"portrait"}
                    -> 200 video/mp4 (H.264 + AAC), headers X-Video-Seed,
                       X-Generation-Seconds

POST requires "Authorization: Bearer <key>" with the key in --key-file.
"""
import argparse
import hmac
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FPS = 24
MAX_PROMPT = 2000
MAX_SECONDS = 10
SIZES = {"landscape": (864, 480), "portrait": (480, 864)}


class Worker:
    def __init__(self, binary, model, gpus, steps, log_path):
        env = dict(os.environ, H3_GPUS=str(gpus))
        self.log = open(log_path, "ab", buffering=0)
        self.process = subprocess.Popen([binary, model], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, env=env)
        self.steps = steps
        self.loaded = threading.Event()  # worker process has loaded
        self.ready = threading.Event()  # warm-up generation finished
        self.dead = threading.Event()
        self.lock = threading.Lock()
        self.pending = {}
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.process.stdout:
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            kind = event.get("event")
            if kind == "ready":
                self.loaded.set()
            elif kind in ("done", "error"):
                slot = self.pending.pop(event.get("id"), None)
                if slot:
                    slot["event"] = event
                    slot["signal"].set()
            elif kind == "fatal":
                print("worker failed to load:", event.get("message"), file=sys.stderr)
        self.dead.set()
        self.loaded.set()
        for slot in list(self.pending.values()):
            slot["signal"].set()

    def warm_up(self, work_dir):
        """The first generation loads every weight onto the GPUs (about 2 min).
        Run it before reporting ready, so no user request pays for it."""
        self.loaded.wait()
        output = os.path.join(work_dir, "warmup.mp4")
        try:
            seconds = self.generate("A quiet lake at sunrise, gentle waves and birds.",
                                    *SIZES["landscape"], 2 * FPS, 0, output)
            print(f"warm-up generation took {seconds:.1f} s", flush=True)
            self.ready.set()
        except (RuntimeError, ValueError) as error:
            print("warm-up failed:", error, file=sys.stderr, flush=True)
            self.process.kill()
        finally:
            try:
                os.unlink(output)
            except OSError:
                pass

    def generate(self, prompt, width, height, frames, seed, output):
        request_id = uuid.uuid4().hex
        slot = {"signal": threading.Event(), "event": None}
        prompt = " ".join(prompt.replace("\t", " ").splitlines())
        line = "\t".join([request_id, str(width), str(height), str(frames), str(self.steps),
                          str(seed), output, prompt]) + "\n"
        with self.lock:  # one request on the GPUs at a time
            if self.dead.is_set():
                raise RuntimeError("worker is not running")
            self.pending[request_id] = slot
            self.process.stdin.write(line.encode("utf-8"))
            self.process.stdin.flush()
            slot["signal"].wait()
        event = slot["event"]
        if event is None:
            raise RuntimeError("worker exited during the request")
        if event["event"] == "error":
            raise ValueError(event.get("message", "generation failed"))
        return event["seconds"]


def make_handler(worker, key, work_dir, max_waiting, model_name, gpus):
    waiting = threading.BoundedSemaphore(max_waiting)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "h3-video"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), fmt % args))

        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status, message, code):
            self._json(status, {"error": {"message": message, "code": code}})

        def do_GET(self):
            if self.path != "/health":
                return self._error(HTTPStatus.NOT_FOUND, "not found", "not_found")
            if worker.dead.is_set():
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "failed"})
            if not worker.ready.is_set():
                state = "warming_up" if worker.loaded.is_set() else "loading"
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": state})
            self._json(HTTPStatus.OK, {"status": "ready", "model": model_name, "gpus": gpus})

        def do_POST(self):
            if self.path != "/v1/videos":
                return self._error(HTTPStatus.NOT_FOUND, "not found", "not_found")
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + key).encode()):
                return self._error(HTTPStatus.UNAUTHORIZED, "invalid key", "invalid_key")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024:
                    raise ValueError
                body = json.loads(self.rfile.read(length))
                prompt = str(body.get("prompt", "")).strip()
                seconds = int(body.get("seconds", 5))
                seed = body.get("seed")
                seed = secrets.randbelow(2**31) if seed is None else int(seed)
                width, height = SIZES[body.get("orientation", "landscape")]
            except (ValueError, TypeError, KeyError, AttributeError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid request body", "invalid_request")
            if not prompt or len(prompt) > MAX_PROMPT:
                return self._error(HTTPStatus.BAD_REQUEST,
                                   f"prompt must be 1-{MAX_PROMPT} characters", "invalid_prompt")
            if not 1 <= seconds <= MAX_SECONDS or not 0 <= seed < 2**63:
                return self._error(HTTPStatus.BAD_REQUEST,
                                   f"seconds must be 1-{MAX_SECONDS}", "invalid_seconds")
            if not worker.ready.is_set() or worker.dead.is_set():
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "model is loading", "loading")
            if not waiting.acquire(blocking=False):
                return self._error(HTTPStatus.TOO_MANY_REQUESTS, "video queue is full", "busy")
            output = os.path.join(work_dir, uuid.uuid4().hex + ".mp4")
            try:
                elapsed = worker.generate(prompt, width, height, seconds * FPS, seed, output)
                with open(output, "rb") as video:
                    data = video.read()
            except ValueError as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error), "generation_failed")
            except (RuntimeError, OSError) as error:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error), "worker_failed")
            finally:
                waiting.release()
                try:
                    os.unlink(output)
                except OSError:
                    pass
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Video-Seed", str(seed))
            self.send_header("X-Generation-Seconds", f"{elapsed:.3f}")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="MiniMax-H3 FastH3")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18960)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--max-waiting", type=int, default=4)
    parser.add_argument("--log", default="worker.log")
    args = parser.parse_args()
    with open(args.key_file) as handle:
        key = handle.read().strip()
    if len(key) < 32:
        sys.exit("key file must hold at least 32 characters")
    work_dir = tempfile.mkdtemp(prefix="h3-video-")
    worker = Worker(args.worker, args.model, args.gpus, args.steps, args.log)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(worker, key, work_dir, args.max_waiting,
                                              args.model_name, args.gpus))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening on {args.host}:{args.port}", flush=True)
    threading.Thread(target=worker.warm_up, args=(work_dir,), daemon=True).start()
    worker.dead.wait()
    server.shutdown()
    sys.exit(f"worker exited with status {worker.process.wait()}")


if __name__ == "__main__":
    main()
