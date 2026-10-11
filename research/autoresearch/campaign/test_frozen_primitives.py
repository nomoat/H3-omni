"""CPU checks for the immutable oracle format and capture transformation."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("frozen", HERE / "frozen_primitives.py")
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)


class FrozenOracleTest(unittest.TestCase):
    def test_transform_preserves_original_inputs_and_runs_no_native_checks(self):
        source = (HERE.parents[2] / "tests/pytorch/primitives.py").read_text()
        transform = frozen.CaptureCheck()
        transformed = transform.visit(ast.parse(source))
        self.assertEqual(transform.replaced, 1)
        checks = [node for node in ast.walk(transformed)
                  if isinstance(node, ast.FunctionDef) and node.name == "check"]
        self.assertEqual(len(checks[0].body), 1)
        self.assertEqual(checks[0].body[0].value.func.id, "_capture_check")
        self.assertEqual([arg.id for arg in checks[0].body[0].value.args],
                         ["records", "name", "shape", "dtype", "function", "inputs", "scalars", "reference"])
        compile(ast.fix_missing_locations(transformed), "primitives.py", "exec")

    def test_manifest_and_blob_modification_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b"fixed input tensor"
            (root / "tensor.bin").write_bytes(data)
            manifest = {"schema_version": 1, "files": {"tensor.bin": frozen.sha(data)}}
            frozen.write_json(root / "manifest.json", manifest)
            expected = frozen.sha((root / "manifest.json").read_bytes())
            self.assertEqual(frozen.load_manifest(root, expected), manifest)
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                frozen.load_manifest(root, "0" * 64)
            (root / "tensor.bin").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                frozen.load_manifest(root, expected)

    def test_import_and_rejected_replay_do_not_import_torch(self):
        code = """
import importlib.util, sys, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location('oracle', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as folder:
    result = module.replay(folder, '0'*64, 'missing.so', 'missing.so', Path(folder)/'result.json')
    assert result['passed'] is False
    assert result['reference_recomputed'] is False
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code, str(HERE / "frozen_primitives.py")], check=True)


if __name__ == "__main__":
    unittest.main()
