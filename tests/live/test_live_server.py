"""HTTP streaming, cancellation, authentication and admission without GPUs."""
import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'serve'))
from live_server import make_live_handler


class FakeWorker:
    def __init__(self):
        self.ready = threading.Event(); self.ready.set()
        self.dead = threading.Event()
        self.started = threading.Event()
        self.finish = threading.Event()
        self.cancelled = threading.Event()

    def generate(self, prompt, width, height, frames, seed, output):
        assert frames == -12
        with open(output, 'wb', buffering=0) as handle:
            handle.write(b'first-media')
            self.started.set()
            while not self.finish.wait(0.01):
                if Path(output + '.cancel').exists():
                    self.cancelled.set()
                    raise ValueError('generation cancelled')
            handle.write(b'last-media')
        return 1.0


class LiveHTTPTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.worker = FakeWorker()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0),
            make_live_handler(self.worker, 'k'*40, self.work.name, 2))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.worker.finish.set()
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.work.cleanup()

    def post(self, body, key='k'*40):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        conn.request('POST', '/v1/live', json.dumps(body),
                     {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
        return conn, conn.getresponse()

    def test_media_before_completion_and_busy(self):
        conn, response = self.post({'prompt': 'test'})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(11), b'first-media')
        self.assertFalse(self.worker.finish.is_set())
        busy, rejected = self.post({'prompt': 'another'})
        self.assertEqual(rejected.status, 429); rejected.read(); busy.close()
        self.worker.finish.set()
        self.assertEqual(response.read(), b'last-media'); conn.close()

    def test_disconnect_cancels_generation(self):
        conn, response = self.post({'prompt': 'test'})
        self.assertEqual(response.read(11), b'first-media')
        response.close(); conn.close()
        self.assertTrue(self.worker.cancelled.wait(3))

    def test_auth_and_limits(self):
        for body, key, status in [({'prompt':'x'}, 'bad', 401),
                                  ({'prompt':'x', 'parts':13}, 'k'*40, 400),
                                  ({'prompt':'x'*2001}, 'k'*40, 400),
                                  ({'prompt':'x', 'seed':True}, 'k'*40, 400)]:
            conn, response = self.post(body, key)
            self.assertEqual(response.status,status); response.read(); conn.close()
        self.assertFalse(self.worker.started.is_set())


if __name__ == '__main__':
    unittest.main()
