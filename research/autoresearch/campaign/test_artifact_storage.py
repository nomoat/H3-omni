import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import artifact_storage as storage


class FakeGcloud:
    def __init__(self, directory, replies, readback=None):
        self.directory = directory
        self.replies = list(replies)
        self.commands = []
        self.readback = readback
        self.readback_uris = []

    def run(self, arguments):
        self.commands.append(arguments)
        code, stdout, stderr = self.replies.pop(0)
        return subprocess.CompletedProcess(arguments, code, stdout, stderr)

    def readback_sha256(self, uri):
        self.readback_uris.append(uri)
        return self.readback


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "evidence.tar"
        self.path.write_bytes(b"frozen evidence\x00\x01")
        self.local = storage.hash_file(self.path)
        self.item = {"role": "candidate", "candidate": 1, "local_path": str(self.path),
                     "sha256": self.local["sha256"], "size_bytes": self.local["size_bytes"]}
        self.metadata = {"bucket": storage.BUCKET,
                         "name": f"{storage.PREFIX}/{self.local['sha256']}/evidence.tar",
                         "size": str(self.local["size_bytes"]), "generation": "12345",
                         "md5Hash": self.local["local_md5_base64"], "crc32c": "test-crc"}
        self.uri = "gs://" + storage.BUCKET + "/" + self.metadata["name"]

    def tearDown(self):
        self.temp.cleanup()

    def ok_metadata(self):
        return 0, json.dumps(self.metadata), ""

    def test_existing_object_is_verified_without_upload(self):
        client = FakeGcloud(self.root, [self.ok_metadata()])
        receipt = storage.upload_artifact(self.item, client)
        self.assertEqual(receipt["disposition"], "existing_verified")
        self.assertEqual(len(client.commands), 1)
        self.assertTrue(receipt["verification"]["passed"])
        self.assertIn("storage.cloud.google.com", receipt["download_url"])

    def test_new_upload_requires_create_only_and_checksum(self):
        client = FakeGcloud(self.root, [(1, "", "HTTPError 404: No such object"),
                                       (0, "", ""), self.ok_metadata()])
        receipt = storage.upload_artifact(self.item, client)
        self.assertEqual(receipt["disposition"], "created")
        self.assertIn("--if-generation-match=0", client.commands[1])
        self.assertIn("--content-md5=" + self.local["local_md5_base64"], client.commands[1])
        self.assertFalse(any("acl" in arg for arg in client.commands[1]))

    def test_create_race_verifies_winner(self):
        client = FakeGcloud(self.root, [(1, "", "404"),
                                       (1, "", "HTTPError 412: conditionNotMet"), self.ok_metadata()])
        receipt = storage.upload_artifact(self.item, client)
        self.assertEqual(receipt["disposition"], "concurrent_create_verified")

    def test_same_address_wrong_bytes_fails_without_overwrite(self):
        self.metadata["md5Hash"] = base64.b64encode(hashlib.md5(b"wrong").digest()).decode()
        client = FakeGcloud(self.root, [self.ok_metadata()])
        with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
            storage.upload_artifact(self.item, client)
        self.assertEqual(len(client.commands), 1)

    def test_absent_md5_reads_exact_generation(self):
        self.metadata.pop("md5Hash")
        client = FakeGcloud(self.root, [self.ok_metadata()],
                            {"sha256": self.local["sha256"], "size_bytes": self.local["size_bytes"]})
        receipt = storage.upload_artifact(self.item, client)
        self.assertEqual(client.readback_uris, [self.uri + "#12345"])
        self.assertEqual(receipt["verification"]["method"], "generation_pinned_sha256_readback")

    def test_absent_md5_wrong_readback_fails(self):
        self.metadata.pop("md5Hash")
        client = FakeGcloud(self.root, [self.ok_metadata()],
                            {"sha256": "0" * 64, "size_bytes": self.local["size_bytes"]})
        with self.assertRaisesRegex(ValueError, "readback SHA256/size mismatch"):
            storage.upload_artifact(self.item, client)

    def test_permission_failure_is_not_missing_object(self):
        client = FakeGcloud(self.root, [(1, "", "HTTPError 403: forbidden")])
        with self.assertRaisesRegex(RuntimeError, "describe failed"):
            storage.upload_artifact(self.item, client)
        self.assertEqual(len(client.commands), 1)

    def test_local_manifest_mismatch_prevents_any_network(self):
        self.item["sha256"] = "0" * 64
        client = FakeGcloud(self.root, [])
        with self.assertRaisesRegex(ValueError, "frozen package manifest"):
            storage.upload_artifact(self.item, client)
        self.assertEqual(client.commands, [])

    def test_wrong_identity_size_or_generation_fails(self):
        for field, replacement in [("bucket", "other-bucket"), ("name", "other-object"),
                                   ("size", "1"), ("generation", ""), ("generation", "0")]:
            with self.subTest(field=field, value=replacement):
                invalid = dict(self.metadata, **{field: replacement})
                client = FakeGcloud(self.root, [(0, json.dumps(invalid), "")])
                with self.assertRaises(ValueError):
                    storage.upload_artifact(self.item, client)


if __name__ == "__main__":
    unittest.main()
