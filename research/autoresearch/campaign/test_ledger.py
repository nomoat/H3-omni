"""CPU-only tests of experiment provenance and resume/round accounting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ledger", Path(__file__).with_name("ledger.py"))
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.seed = self.base / "seed"
        self.seed.mkdir()
        (self.seed / "kernel.c").write_text("int kernel(void) { return 1; }\n")
        (self.seed / "runs").mkdir()
        (self.seed / "runs" / "previous.json").write_text("{}")
        ledger.Campaign.init(self.base / "campaign", self.seed, rounds=2)
        self.campaign = ledger.Campaign(self.base / "campaign")
        self.raw = self.base / "result.json"
        self.raw.write_text('{"measured_seconds": 12.0}')

    def tearDown(self):
        self.temp.cleanup()

    def candidate(self, value, parent=None):
        started = self.campaign.start({"gpu_count": 1}, f"kernel {value}", parent)
        (Path(started["worktree"]) / "kernel.c").write_text(
            f"int kernel(void) {{ return {value}; }}\n")
        sealed = self.campaign.seal(started["candidate"])
        return started, sealed

    def test_tree_branch_resume_and_round_limit(self):
        initial = self.campaign.status()
        started, sealed = self.candidate(2)
        self.campaign.attempt(1, "run", "success", self.raw)
        self.campaign.finish(1, "keep", True, 12.0, self.raw, fullgate=True)
        # A new process can resume and branch explicitly from the seed.
        resumed = ledger.Campaign(self.campaign.root)
        self.assertEqual(resumed.status()["incumbent_commit"], sealed["commit"])
        second, _ = self.candidate(3, initial["baseline_commit"])
        self.assertEqual(second["parent_commit"], initial["baseline_commit"])
        resumed.attempt(2, "build", "candidate_error", self.raw)
        resumed.finish(2, "crash", False, None, self.raw)
        self.assertEqual(resumed.status()["completed_rounds"], 2)
        with self.assertRaisesRegex(ledger.LedgerError, "budget"):
            resumed.start({}, "third")
        self.assertEqual((self.seed / "kernel.c").read_text(),
                         "int kernel(void) { return 1; }\n")
        self.assertFalse((self.seed / ".git").exists())
        self.assertFalse((resumed.repo / "runs").exists())

    def test_infrastructure_failure_and_fullgate(self):
        self.candidate(2)
        self.campaign.attempt(1, "build", "success")
        self.campaign.attempt(1, "run", "infra_error", self.raw)
        with self.assertRaisesRegex(ledger.LedgerError, "infrastructure"):
            self.campaign.finish(1, "crash", False, None, self.raw)
        self.assertEqual(self.campaign.status()["completed_rounds"], 0)
        self.campaign.attempt(1, "run", "success", self.raw)
        with self.assertRaisesRegex(ledger.LedgerError, "fullgate"):
            self.campaign.finish(1, "keep", True, 12.0, self.raw)
        self.campaign.finish(1, "discard", True, 12.0, self.raw)
        self.assertEqual(self.campaign.status()["completed_rounds"], 1)

    def test_duplicate_patch_rejected_even_when_config_changes(self):
        initial = self.campaign.status()
        self.candidate(2)
        duplicate = self.campaign.start({"gpu_count": 2}, "duplicate",
                                        initial["baseline_commit"])
        (Path(duplicate["worktree"]) / "kernel.c").write_text(
            "int kernel(void) { return 2; }\n")
        with self.assertRaisesRegex(ledger.LedgerError, "Duplicate"):
            self.campaign.seal(2)
        self.assertEqual(self.campaign.status()["completed_rounds"], 0)

    def test_uncommitted_source_change_blocks_finish(self):
        started, _ = self.candidate(2)
        self.campaign.attempt(1, "run", "success", self.raw)
        (Path(started["worktree"]) / "kernel.c").write_text("changed after run\n")
        with self.assertRaisesRegex(ledger.LedgerError, "changed after seal"):
            self.campaign.finish(1, "keep", True, 12.0, self.raw, fullgate=True)

    def test_tail_recovery_and_full_line_tamper(self):
        self.candidate(2)
        with self.campaign.journal.open("ab") as out:
            out.write(b'{"seq":4,"incomplete":')
        with self.assertRaisesRegex(ledger.LedgerError, "Incomplete"):
            self.campaign.status()
        recovered = self.campaign.recover_tail()
        self.assertTrue(Path(recovered["backup"]).is_file())
        self.assertEqual(self.campaign.status()["completed_rounds"], 0)
        raw = self.campaign.journal.read_text()
        self.campaign.journal.write_text(raw.replace('"max_rounds":2', '"max_rounds":9'))
        with self.assertRaisesRegex(ledger.LedgerError, "Invalid journal"):
            self.campaign.status()
        with self.assertRaises(ledger.LedgerError):
            self.campaign.recover_tail()

    def test_result_copy_is_immutable_and_metadata_checked(self):
        self.candidate(2)
        self.campaign.attempt(1, "run", "success", self.raw)
        done = self.campaign.finish(1, "discard", False, 12.0, self.raw)
        self.raw.write_text("caller overwrote its result")
        saved = Path(done["raw_result_path"]).read_bytes()
        self.assertEqual(ledger.digest(saved), done["raw_result_sha256"])
        metadata = self.campaign.root / "campaign.json"
        content = json.loads(metadata.read_text())
        content["max_rounds"] = 999
        metadata.write_text(json.dumps(content))
        with self.assertRaisesRegex(ledger.LedgerError, "metadata"):
            self.campaign.status()

    def test_interrupted_seal_reuses_saved_commit(self):
        started = self.campaign.start({}, "interrupted commit")
        (Path(started["worktree"]) / "kernel.c").write_text("new kernel\n")
        original = self.campaign.append

        def interrupted(kind, **fields):
            if kind == "seal":
                raise OSError("simulated interruption after Git commit")
            return original(kind, **fields)

        with patch.object(self.campaign, "append", interrupted):
            with self.assertRaises(OSError):
                self.campaign.seal(1)
        saved = ledger.git(Path(started["worktree"]), "rev-parse", "HEAD")
        sealed = self.campaign.seal(1)
        self.assertEqual(sealed["commit"], saved)

    def test_interrupted_start_can_recreate_unsealed_worktree(self):
        real_git = ledger.git

        def interrupted(directory, *args):
            if args[:2] == ("worktree", "add"):
                raise OSError("simulated interruption before worktree creation")
            return real_git(directory, *args)

        with patch.object(ledger, "git", interrupted):
            with self.assertRaises(OSError):
                self.campaign.start({}, "interrupted worktree")
        candidate = self.campaign.resume(1)
        self.assertTrue((Path(candidate["worktree"]) / "kernel.c").is_file())


if __name__ == "__main__":
    unittest.main()
