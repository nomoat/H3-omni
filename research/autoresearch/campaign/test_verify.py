"""CPU-only tests for the judge's fail-closed provenance checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("verify.py")
SPEC = importlib.util.spec_from_file_location("campaign_verify", MODULE_PATH)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.spec = verify.read_json(verify.TESTS / "extended.json")
        self.report = verify.read_json(verify.TESTS / "evidence/14274/extended/trajectory/results.json")

    def test_accepted_current_report(self):
        verify.check_accepted_trajectory(self.report, self.spec)

    def test_old_oracle_rejected(self):
        self.report["reference_timestep_rows"][0] = 2
        with self.assertRaises(ValueError):
            verify.check_accepted_trajectory(self.report, self.spec)

    def test_nonexact_archive_rejected_for_cache_reconstruction(self):
        self.report["results"][0]["relative_l2"] = 1e-12
        with self.assertRaises(ValueError):
            verify.check_accepted_trajectory(self.report, self.spec)

    def test_missing_or_duplicate_step_rejected(self):
        for transform in (lambda r: r.pop(), lambda r: r.__setitem__(1, copy.deepcopy(r[0]))):
            report = copy.deepcopy(self.report)
            transform(report["results"])
            with self.assertRaises(ValueError):
                verify.check_accepted_trajectory(report, self.spec)

    def test_diagnostic_and_wrong_geometry_rejected(self):
        self.report["trace_enabled"] = True
        with self.assertRaises(ValueError):
            verify.check_accepted_trajectory(self.report, self.spec)
        self.report["trace_enabled"] = False
        self.report["arguments"]["width"] = 512
        with self.assertRaises(ValueError):
            verify.check_accepted_trajectory(self.report, self.spec)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = ["initial", "final", "video-decoded", "audio-decoded"] + [f"step-{i:02d}" for i in range(20)]
        cases = verify.case_specs()
        for case in cases:
            (self.root / case["key"]).mkdir()
            for name in names:
                (self.root / case["key"] / (name + ".safetensors")).write_bytes(name.encode())
        texts = [("english", "A cat walks through a sunny garden."), ("chinese", "\u4e00\u53ea\u732b\u5728\u82b1\u56ed\u91cc\u8d70\u52a8\u3002"),
                 ("long_prompt", verify.read_json(verify.TESTS / "extended.json")["prompt"])]
        for key, _ in texts:
            (self.root / ("text-" + key + ".safetensors")).write_bytes(key.encode())
        (self.root / "tokenizer.json").write_text("{}")
        (self.root / "units").mkdir()
        unit_manifest = {"files": {}, "schema_version": 1, "primitive_cases": [{}] * 69,
            "host_cases": [{}] * 12, "layout_cases": [{}] * 32, "limits": verify.frozen_contract()["primitives"],
            "provenance": {"capture_source_sha256": verify.digest(MODULE_PATH.with_name("frozen_primitives.py")),
                "source_sha256": {name: verify.digest(verify.TESTS / name) for name in
                    ("primitives.py", "host.py", "native.py", "acceptance.json", "sources.json")}}}
        verify.write_json(self.root / "units/manifest.json", unit_manifest)
        self.manifest = {"schema": verify.SCHEMA, "source_job": 14274,
            "contract_sha256": verify.CONTRACT_SHA256, "sources_sha256": verify.SOURCES_SHA256,
            "configuration": verify.read_json(verify.TESTS / "extended.json"), "harness_sha256": {},
            "reference_timestep_plan": verify.TIMESTEP_PLAN, "reference_timestep_rows": [1] + [2] * 19,
            "units": {"directory": "units", "manifest_sha256": verify.digest(self.root / "units/manifest.json")},
            "cases": cases, "texts": [{"key": k, "prompt": p, "file": "text-" + k + ".safetensors"} for k, p in texts],
            "files": {str(p.relative_to(self.root)): verify.digest(p) for p in self.root.rglob("*") if p.is_file()}}
        self.write()

    def write(self):
        verify.write_json(self.root / "manifest.json", self.manifest)
        self.args = SimpleNamespace(oracle=self.root, oracle_sha256=verify.digest(self.root / "manifest.json"))

    def load(self):
        with patch.object(verify, "harness_hashes", return_value={}):
            return verify.load_manifest(self.args)

    def test_all_frozen_artifacts_required(self):
        self.assertEqual(self.load()["schema"], verify.SCHEMA)
        del self.manifest["files"]["extended/step-19.safetensors"]
        self.write()
        with self.assertRaises(ValueError):
            self.load()

    def test_content_tampering_rejected(self):
        (self.root / "extended/step-00.safetensors").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.load()

    def test_manifest_tampering_rejected(self):
        with (self.root / "manifest.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            self.load()

    def test_harness_change_rejected(self):
        self.manifest["harness_sha256"] = {"fake.py": "changed"}
        self.write()
        with self.assertRaises(ValueError):
            self.load()

    def test_relaxed_unit_limits_rejected(self):
        path = self.root / "units/manifest.json"
        unit = verify.read_json(path)
        unit["limits"]["float32"]["relative_l2"] = 1.0
        verify.write_json(path, unit)
        with self.assertRaisesRegex(ValueError, "precision limits"):
            verify.validate_units(self.root / "units")


if __name__ == "__main__":
    unittest.main()
