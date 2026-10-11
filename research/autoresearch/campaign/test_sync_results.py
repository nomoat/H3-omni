"""Exercise real Git/journal imports with simulated process interruptions."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import sync_results as sync
from ledger import Campaign


class SyncResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        seed = base / "seed"
        seed.mkdir()
        (seed / "kernel.c").write_text("seed\n")
        self.root = base / "campaign"
        Campaign.init(self.root, seed, rounds=2)
        self.campaign = Campaign(self.root)
        started = self.campaign.start({}, "test import")
        (Path(started["worktree"]) / "kernel.c").write_text("candidate\n")
        self.campaign.seal(1)
        candidate = self.campaign.status()["candidates"][1]
        entry = {key: candidate[key] for key in ("candidate", "commit", "parent_commit", "patch_sha256")}
        self.write(self.root / "plan.json", [entry])
        self.write(self.root / "protocol.json", {"rounds": 2, "prompt_sha256": "a" * 64})
        identity = {"plan_sha256": sync.digest((self.root / "plan.json").read_bytes()),
                    "limit": 2, "prompt_sha256": "a" * 64}
        self.progress = {"campaign_identity": identity, "campaign_identity_sha256": sync.identity_hash(identity)}
        self.record = {**entry, "campaign_identity_sha256": sync.identity_hash(identity),
                       "terminal": True, "status": "discard", "correctness": True,
                       "fullgate": False, "score_seconds": 12.0, "reason": "no improvement",
                       "attempts": [{"stage": "build", "outcome": "success"},
                                    {"stage": "run", "outcome": "success"}],
                       "benchmark_parent": 0, "state_after": {"incumbent": 0, "incumbent_score": 10.0}}
        self.path = self.root / "evidence/search/0001/result.json"
        sync.install_archive(self.root, self.archive())

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def archive(self, result=None, extras=None):
        contents = {"0001/result.json": json.dumps(self.record if result is None else result).encode(),
                    "progress.json": json.dumps(self.progress).encode()}
        contents.update(extras or {})
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            for name, data in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return output.getvalue()

    def assert_one_complete_import(self):
        state = self.campaign.status()
        self.assertEqual(state["completed_rounds"], 1)
        self.assertEqual(len(state["candidates"][1]["attempts"]), 2)
        self.assertEqual(sum(e["kind"] == "finish" for e in self.campaign.events()), 1)

    def test_restart_after_first_attempt_only_appends_missing_suffix(self):
        original = self.campaign.attempt

        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("interrupted after attempt append")

        with patch.object(self.campaign, "attempt", interrupted):
            with self.assertRaises(OSError):
                sync.settle_results(self.campaign)
        self.assertEqual(len(self.campaign.status()["candidates"][1]["attempts"]), 1)
        sync.settle_results(self.campaign)
        before = self.campaign.journal.read_bytes()
        sync.settle_results(self.campaign)
        self.assertEqual(before, self.campaign.journal.read_bytes())
        self.assert_one_complete_import()

    def test_restart_between_attempts_and_finish(self):
        with patch.object(self.campaign, "finish", side_effect=OSError("interrupted before finish")):
            with self.assertRaises(OSError):
                sync.settle_results(self.campaign)
        self.assertEqual(len(self.campaign.status()["candidates"][1]["attempts"]), 2)
        sync.settle_results(self.campaign)
        self.assert_one_complete_import()

    def add_preflight_attempts(self):
        build = self.root / "evidence/compile-smoke/0001/build.json"
        checks = build.with_name("primitives.json")
        self.write(build, {"build": "passed"})
        self.write(checks, {"passed": True, "primitive_cases": 69})
        self.campaign.attempt(1, "build", "success", build,
                              note="NVCC smoke build before frozen-oracle campaign launch")
        self.campaign.attempt(1, "run", "success", checks,
                              note="Preflight only: 69 primitive checks. Full fixed-oracle trial still pending.")

    def test_preflight_attempts_are_preserved_and_not_import_cursor(self):
        self.add_preflight_attempts()
        preflight = self.campaign.status()["candidates"][1]["attempts"]
        sync.settle_results(self.campaign)
        state = self.campaign.status()
        self.assertEqual(state["completed_rounds"], 1)
        self.assertEqual(len(state["candidates"][1]["attempts"]), 4)
        self.assertEqual(state["candidates"][1]["attempts"][:2], preflight)
        before = self.campaign.journal.read_bytes()
        sync.settle_results(self.campaign)
        self.assertEqual(self.campaign.journal.read_bytes(), before)

    def test_interrupted_import_with_preflight_skips_only_owned_prefix(self):
        self.add_preflight_attempts()
        original = self.campaign.attempt

        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("interrupted after first remote attempt")

        with patch.object(self.campaign, "attempt", interrupted):
            with self.assertRaises(OSError):
                sync.settle_results(self.campaign)
        candidate = self.campaign.status()["candidates"][1]
        self.assertEqual(len(candidate["attempts"]), 3)
        self.assertEqual(len(sync.imported_attempts(candidate, self.path)), 1)
        sync.settle_results(self.campaign)
        state = self.campaign.status()
        self.assertEqual(len(state["candidates"][1]["attempts"]), 4)
        self.assertEqual(state["completed_rounds"], 1)

    def test_bad_import_hash_cannot_hide_behind_preflight_filter(self):
        self.add_preflight_attempts()
        wrong = self.root / "wrong-result.json"
        self.write(wrong, {"bad": "evidence"})
        self.campaign.attempt(1, "build", "success", wrong,
                              note=sync.REMOTE_ATTEMPT_NOTE + "; attempt 1")
        with self.assertRaisesRegex(ValueError, "Ledger evidence hash"):
            sync.settle_results(self.campaign)

    def test_restart_after_finish_does_not_write_again(self):
        original = self.campaign.finish

        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("interrupted after finish append")

        with patch.object(self.campaign, "finish", interrupted):
            with self.assertRaises(OSError):
                sync.settle_results(self.campaign)
        before = self.campaign.journal.read_bytes()
        sync.settle_results(self.campaign)
        self.assertEqual(before, self.campaign.journal.read_bytes())
        self.assert_one_complete_import()

    def test_changed_remote_and_local_terminal_results_are_rejected(self):
        sync.settle_results(self.campaign)
        changed = {**self.record, "score_seconds": 1.0}
        with self.assertRaisesRegex(ValueError, "terminal evidence changed"):
            sync.install_archive(self.root, self.archive(changed))
        self.write(self.path, changed)
        with self.assertRaisesRegex(ValueError, "Ledger evidence hash"):
            sync.settle_results(self.campaign)

    def test_changed_ledger_raw_snapshot_is_rejected(self):
        sync.settle_results(self.campaign)
        snapshot = Path(self.campaign.status()["candidates"][1]["raw_result_path"])
        snapshot.write_text("tampered evidence")
        with self.assertRaisesRegex(ValueError, "saved raw terminal evidence"):
            sync.settle_results(self.campaign)

    def test_invalid_download_never_replaces_good_local_evidence(self):
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            sync.install_archive(self.root, self.archive(extras={"0002/result.json": b'{"partial":'}))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.path.parent.parent / "0002/result.json").exists())
        with self.assertRaisesRegex(ValueError, "archive member"):
            sync.install_archive(self.root, self.archive(extras={"../../escaped.json": b"{}"}))

    def test_terminal_identity_checked_even_after_finish(self):
        sync.settle_results(self.campaign)
        self.write(self.path, {**self.record, "commit": "bad"})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            sync.settle_results(self.campaign)

    def test_unconfirmed_keep_rejected_before_any_attempt_is_written(self):
        candidate = self.campaign.status()["candidates"][1]
        invalid = {**self.record, "status": "keep", "fullgate": True,
                   "state_after": {"incumbent": 1, "incumbent_score": 12.0}}
        with self.assertRaisesRegex(ValueError, "confirmed timing"):
            sync.validate_record(invalid, candidate, json.loads((self.root / "plan.json").read_text())[0],
                                 self.progress["campaign_identity_sha256"], 1)
        self.assertEqual(candidate["attempts"], [])


if __name__ == "__main__":
    unittest.main()
