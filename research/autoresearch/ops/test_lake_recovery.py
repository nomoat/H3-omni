import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lake_recovery as recovery


class TerminationTests(unittest.TestCase):
    def setUp(self):
        self.now = recovery.timestamp("2026-10-10T12:05:00Z")
        self.disabled = "2026-10-10T12:01:00Z"
        self.before = {"config": {"revision": "rev"}, "instance_ids": ["old"]}

    def metrics(self, active=0, idle=0, times=("12:03:00", "12:04:00")):
        return {"http_status": 200, "series": [
            {"revision": "rev", "state": state, "points": [
                {"time": "2026-10-10T" + time + "Z", "count": count} for time in times
            ]} for state, count in [("active", active), ("idle", idle)]
        ]}

    def event(self, kind="shutdown", identity="old", time="12:02:00"):
        return {"kind": kind, "instance_id": identity, "revision": "rev", "timestamp": "2026-10-10T" + time + "Z"}

    def proof(self, metrics=None, events=()):
        return recovery.termination_evidence(self.before, events, metrics or {"http_status": 200, "series": []}, self.disabled, self.now)

    def test_missing_metrics_or_missing_state_is_not_zero(self):
        self.assertIsNone(self.proof())
        self.assertIsNone(self.proof({"http_status": 403, "series": []}))
        partial = self.metrics()
        partial["series"].pop()
        self.assertIsNone(self.proof(partial))

    def test_two_complete_recent_zero_samples_are_positive_evidence(self):
        self.assertEqual(self.proof(self.metrics())["kind"], "two_explicit_zero_samples")

    def test_one_zero_sample_or_old_zero_samples_do_not_pass(self):
        self.assertIsNone(self.proof(self.metrics(times=("12:04:00",))))
        self.assertIsNone(self.proof(self.metrics(times=("12:00:00", "12:01:00"))))

    def test_idle_instances_and_other_revisions_cannot_be_ignored(self):
        self.assertIsNone(self.proof(self.metrics(idle=1)))
        metrics = self.metrics()
        metrics["series"].append({"revision": "unexpected", "state": "active", "points": [{"time": "2026-10-10T12:04:30Z", "count": 1}]})
        self.assertIsNone(self.proof(metrics))

    def test_new_start_after_zero_samples_invalidates_them(self):
        self.assertIsNone(self.proof(self.metrics(), [self.event("start", "new", "12:04:30")]))

    def test_old_or_wrong_identity_shutdown_is_not_evidence(self):
        self.assertIsNone(self.proof(events=[self.event(time="12:00:00")]))
        self.assertIsNone(self.proof(events=[self.event(identity="other")]))

    def test_shutdown_evidence_needs_every_observed_instance(self):
        events = [self.event(), self.event("start", "new", "12:02:30")]
        self.assertIsNone(self.proof(events=events))
        events.append(self.event(identity="new", time="12:03:00"))
        self.assertEqual(self.proof(events=events)["instance_ids"], ["new", "old"])

    def test_live_metric_after_shutdown_deadline_blocks_log_evidence(self):
        self.assertIsNone(self.proof(self.metrics(active=1), [self.event()]))

    def test_restart_of_same_identity_needs_another_shutdown(self):
        self.assertIsNone(self.proof(events=[self.event(), self.event("start", "old", "12:03:00")]))


class SafetyTests(unittest.TestCase):
    def test_config_export_keeps_names_but_no_environment_values(self):
        raw = {"metadata": {"name": recovery.SERVICE, "uid": "uid", "annotations": {}},
               "spec": {"template": {"spec": {"containers": [{"image": "image", "env": [{"name": "TOKEN", "value": "SENSITIVE_MARKER"}]}]}}},
               "status": {"latestReadyRevisionName": "rev", "traffic": []}}
        safe = recovery.sanitized_config(raw)
        self.assertNotIn("SENSITIVE_MARKER", json.dumps(safe))
        self.assertEqual(safe["containers"][0]["environment_variable_names"], ["TOKEN"])

    def test_missing_approval_never_constructs_cloud_client(self):
        with patch.object(recovery, "Cloud") as cloud, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                recovery.main(["recover", "--output-dir", "/unused"])
            self.assertEqual(error.exception.code, 2)
            cloud.assert_not_called()

    def test_probe_mode_never_calls_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            result = {"ready_to_disable": True, "instance_ids": ["old"], "metrics": {"http_status": 200}}
            with patch.object(recovery, "Cloud") as cloud, patch.object(recovery, "probe", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recovery.main(["probe", "--output-dir", str(Path(directory) / "probe")]), 0)
                cloud.return_value.scale.assert_not_called()

    def test_service_changes_or_nonzero_scaling_block_restore(self):
        original = {"name": "lake", "uid": "u", "spec_sha256": "hash", "revision": "rev", "traffic": []}
        changed = dict(original, scaling={"run.googleapis.com/scalingMode": "manual", "run.googleapis.com/manualInstanceCount": "1"})
        with self.assertRaises(recovery.RecoveryError):
            recovery.assert_unchanged(original, changed, disabled=True)
        changed["scaling"]["run.googleapis.com/manualInstanceCount"] = "0"
        changed["spec_sha256"] = "different"
        with self.assertRaises(recovery.RecoveryError):
            recovery.assert_unchanged(original, changed, disabled=True)


if __name__ == "__main__":
    unittest.main()
