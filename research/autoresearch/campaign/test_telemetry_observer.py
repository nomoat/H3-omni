"""CPU-only observer tests: identity, interruption and durable metadata."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from telemetry_observer import Observer, attempt_id, canonical, sha


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage = self.root / "stage"
        self.shared = self.root / "shared/results"
        self.outbox = self.root / "shared/telemetry"
        self.stage.mkdir()
        self.entry = dict(candidate=1, commit="commit1", parent_commit="seed", source_sha256="s"*64,
                          patch_sha256="p"*64, config={"threads":256})
        self.record = dict(self.entry, started_unix=100.25, campaign_identity_sha256="c"*64,
                           terminal=False, status="running", attempts=[])
        self.progress = dict(campaign_identity_sha256="c"*64, active=1, phase="build", status="running", completed=[])
        self.folder = self.stage / "results/0001"
        self.write(self.stage / "plan.json", [self.entry])
        self.write(self.stage / "results/progress.json", self.progress)
        self.write(self.folder / "inflight.json", self.record)
        self.observer = self.make_observer()

    def make_observer(self):
        return Observer(self.stage, self.shared, self.outbox, "test-job")

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(data) + b"\n")

    def events(self, kind=None):
        values = [json.loads(p.read_text()) for p in (self.outbox / "events").glob("*.json")]
        return [e for e in values if kind is None or e["kind"] == kind]

    def attempt(self, record=None):
        return self.observer.state["attempts"][attempt_id(1, record or self.record)]

    def test_begin_phase_snapshots_and_id_match_exporter(self):
        self.observer.poll()
        self.assertEqual(len(self.events("attempt_begin")), 1)
        self.assertEqual(self.attempt()["phase"], "build")
        self.progress["phase"] = "cold_benchmark"
        self.write(self.stage / "results/progress.json", self.progress)
        self.write(self.folder / "benchmark/samples.json", [{"seconds":80.25}])
        self.observer.poll()
        ref = self.attempt()["metadata"]["benchmark/samples.json"]
        self.assertEqual(sha((self.outbox / ref["blob"]).read_bytes()), ref["sha256"])
        self.assertEqual(self.attempt()["phase"], "cold_benchmark")
        self.assertFalse(self.attempt()["counts_as_completed"])
        expected = "0001-" + sha(canonical(100.25))[:16]
        self.assertEqual(self.attempt()["attempt_id"], expected)

    def test_retry_and_restart_keep_distinct_execution_ids(self):
        self.observer.poll()
        old_id = self.attempt()["attempt_id"]
        retry = self.stage / "results/retries/0001-123456"
        retry.parent.mkdir()
        self.folder.rename(retry)
        new = dict(self.record, started_unix=200.5)
        self.write(self.folder / "inflight.json", new)
        self.observer.poll(force_discover=True)
        self.assertEqual(len(self.observer.state["attempts"]), 2)
        old = self.observer.state["attempts"][old_id]
        self.assertEqual(old["source_directory"], str(retry))
        self.assertEqual(old["state"], "interrupted")
        self.assertEqual(self.attempt(new)["state"], "running")
        event_ids = {e["event_id"] for e in self.events()}
        self.observer = self.make_observer()
        self.observer.poll(force_discover=True)
        self.assertEqual(event_ids, {e["event_id"] for e in self.events()})

    def test_terminal_waits_for_worker_commit_and_never_reads_tar(self):
        terminal = dict(self.record, terminal=True, status="discard", finished_unix=150,
                        attempts=[{"stage":"run","outcome":"success"}], score_seconds=90)
        self.write(self.folder / "result.json", terminal)
        self.write(self.shared / "0001/result.json", terminal)
        archive = self.shared / "0001/evidence.tar"
        archive.write_bytes(b"not read by observer")
        self.observer.poll()
        self.assertEqual(self.attempt()["state"], "terminal_pending_archive")
        self.progress["completed"] = [1]
        self.write(self.stage / "results/progress.json", self.progress)
        pending = archive.with_suffix(".tar.pending")
        pending.write_bytes(b"incomplete")
        self.observer.poll()
        self.assertFalse(self.attempt()["counts_as_completed"])
        pending.unlink()
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.suffix in (".tar", ".npy", ".log"):
                raise AssertionError("Observer tried to read a large artifact")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open):
            self.observer.poll()
        self.assertTrue(self.attempt()["counts_as_completed"])
        descriptor = self.attempt()["archive"]
        self.assertEqual(descriptor["kind"], "worker_terminal_tar")
        self.assertTrue(descriptor["ready"] and descriptor["hash_pending"])
        self.assertEqual(len(self.events("attempt_terminal")), 1)

    def test_stopped_drain_requests_collection_without_tar_or_fake_end_time(self):
        self.write(self.folder / "benchmark/samples.json", [{"seconds":80}, {"seconds":81}])
        (self.folder / "large.npy").write_bytes(b"large data is not opened")
        self.observer.poll(force_discover=True, stopped=True)
        attempt = self.attempt()
        self.assertEqual(attempt["state"], "interrupted")
        self.assertFalse(attempt["counts_as_completed"])
        self.assertTrue(attempt["archive"]["safe_to_collect"])
        self.assertFalse(attempt["archive"]["ready"])
        self.assertIn("benchmark/samples.json", attempt["metadata"])
        self.assertEqual(self.events("attempt_interrupted")[0]["source_time_unix"], None)
        self.assertEqual(list(self.outbox.rglob("*.tar")), [])

    def test_partial_json_and_cross_campaign_are_rejected(self):
        (self.folder / "inflight.json").write_text('{"candidate":')
        self.observer.poll()
        self.assertFalse(self.observer.state["attempts"])
        self.write(self.folder / "inflight.json", self.record)
        self.observer.poll()
        self.progress["campaign_identity_sha256"] = "different"
        self.write(self.stage / "results/progress.json", self.progress)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.observer.poll()

    def test_directory_replacement_cannot_relabel_new_metadata(self):
        self.observer.poll()
        old = self.attempt()
        inode = self.folder.stat().st_ino
        retry = self.stage / "results/retries/0001-race"
        retry.parent.mkdir()
        self.folder.rename(retry)
        self.write(self.folder / "inflight.json", dict(self.record, started_unix=200))
        self.write(self.folder / "benchmark/samples.json", [{"seconds":123}])
        self.assertIsNone(self.observer.snapshot(old, "benchmark/samples.json", inode))
        self.assertNotIn("benchmark/samples.json", old["metadata"])

    def test_terminal_identity_cannot_change_campaign(self):
        self.observer.poll()
        self.write(self.folder / "result.json", dict(self.record, terminal=True,
                   status="discard", campaign_identity_sha256="different"))
        with self.assertRaisesRegex(ValueError, "terminal differs"):
            self.observer.poll()

    def test_replacement_during_identity_read_cannot_attach_samples(self):
        self.observer.poll()
        old = self.attempt()
        read_json = self.observer.small_json
        replaced = []
        def racing_read(path):
            found = read_json(path)
            if path == self.folder / "inflight.json" and not replaced:
                replaced.append(True)
                retry = self.stage / "results/retries/0001-race"
                retry.parent.mkdir()
                self.folder.rename(retry)
                self.write(self.folder / "inflight.json", dict(self.record, started_unix=200))
                self.write(self.folder / "benchmark/samples.json", [{"seconds":123}])
            return found
        with patch.object(self.observer, "small_json", racing_read):
            self.observer.observe_attempt(old)
        self.assertNotIn("benchmark/samples.json", old["metadata"])


if __name__ == "__main__":
    unittest.main()
