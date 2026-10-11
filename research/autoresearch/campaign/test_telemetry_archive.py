"""Small stopped-producer archive checks; local CPU/tar only, no SSH or GCS."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import telemetry_archive as archive
import lake_collect
import lake_export


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class StoppedArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage = self.root / 'stage'
        self.source = self.stage / 'results/0002'
        self.outbox = self.root / 'observer'
        self.destination = self.root / 'closed-archives'
        self.record = {'candidate': 2, 'started_unix': 1000.0, 'commit': '2' * 40,
                       'parent_commit': '0' * 40, 'source_sha256': 'a' * 64,
                       'patch_sha256': 'b' * 64, 'campaign_identity_sha256': 'c' * 64,
                       'terminal': False, 'status': 'running', 'score_seconds': None}
        write_json(self.source / 'inflight.json', self.record)
        (self.source / 'binary.safetensors').write_bytes(b'raw tensor\x00\xff')
        (self.source / 'native.log').write_bytes(b'partial inference\n')
        self.aid = archive.attempt_id(2, self.record)
        self.descriptor_path = self.outbox / 'archives' / (self.aid + '.json')
        self.request = {'candidate': 2, 'attempt_id': self.aid, 'kind': 'interrupted_directory',
                        'safe_to_collect': True, 'source_path': str(self.source),
                        'expected_started_unix': 1000.0}
        write_json(self.descriptor_path, self.request)
        identity = {key: self.record[key] for key in ('started_unix', 'commit', 'parent_commit',
                                                     'source_sha256', 'patch_sha256', 'campaign_identity_sha256')}
        write_json(self.outbox / 'attempts' / self.aid / 'index.json',
                   {'candidate': 2, 'attempt_id': self.aid, 'identity': identity, 'state': 'interrupted'})
        self.target = self.destination / self.aid / 'evidence.tar'
        self.receipt = self.outbox / 'closed' / (self.aid + '.json')

    def tearDown(self):
        self.temp.cleanup()

    def package(self):
        archive.package_stopped(self.stage, self.outbox, self.destination)

    def test_stopped_archive_is_full_hashed_and_repeated_call_is_idempotent(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        self.package()
        raw, receipt_raw = self.target.read_bytes(), self.receipt.read_bytes()
        receipt = json.loads(receipt_raw)
        self.assertEqual(receipt['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt['size_bytes'], len(raw))
        self.assertTrue(receipt['producer_stopped'])
        self.assertFalse(receipt['counts_as_completed'])
        self.assertNotIn('score_seconds', receipt)
        with tarfile.open(self.target) as bundle:
            captured = {Path(m.name).name: bundle.extractfile(m).read() for m in bundle if m.isfile()}
        self.assertEqual(captured, before)
        self.package()
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(self.receipt.read_bytes(), receipt_raw)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)

    def test_without_stopped_descriptor_confirmation_no_archive_is_created(self):
        write_json(self.descriptor_path, dict(self.request, safe_to_collect=False))
        self.package()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.receipt.exists())

    def test_replaced_attempt_start_is_rejected(self):
        write_json(self.source / 'inflight.json', dict(self.record, started_unix=2000.0))
        with self.assertRaises(ValueError):
            self.package()
        self.assertFalse(self.receipt.exists())

    def test_same_start_changed_source_identity_is_rejected(self):
        write_json(self.source / 'inflight.json', dict(self.record, commit='9' * 40))
        with self.assertRaises(ValueError):
            self.package()
        self.assertFalse(self.receipt.exists())

    def test_same_size_archive_corruption_is_rejected_on_replay(self):
        self.package()
        raw = bytearray(self.target.read_bytes())
        raw[len(raw) // 2] ^= 1
        self.target.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.package()

    def test_closed_package_import_stays_nonterminal_without_final_score(self):
        self.package()
        local = self.root / 'local-campaign'
        local_receipt = local / 'evidence/telemetry/closed' / self.receipt.name
        local_receipt.parent.mkdir(parents=True)
        shutil.copyfile(self.receipt, local_receipt)
        write_json(local / 'protocol.json', {'rounds': 100})
        write_json(local / 'operations.json', {'oracle_manifest_sha256': 'd' * 64})

        def local_download(number, target, source_path):
            self.assertEqual(number, 2)
            self.assertEqual(Path(source_path), self.target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            raw = target.read_bytes()
            metadata = {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
            write_json(target.with_suffix('.transfer.json'), metadata)
            return metadata

        with patch.object(lake_collect, 'download', side_effect=local_download) as download:
            first = lake_collect.collect(local, upload=False)
            second = lake_collect.collect(local, upload=False)
        self.assertEqual(first, second)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(len(first['packages']), 1)
        package = first['packages'][0]
        self.assertEqual(package['execution_state'], 'interrupted')
        detail = Path(package['detail_root'])
        self.assertFalse((detail / 'result.json').exists())
        self.assertFalse(json.loads((detail / 'inflight.json').read_text())['terminal'])
        obj = lake_export.Export(local)
        obj.packages([self.record])
        result = obj.streams['attempts/results'][0]
        self.assertEqual(result['state'], 'interrupted')
        self.assertFalse(result['counts_as_completed'])
        self.assertIsNone(result['score_seconds'])


if __name__ == '__main__':
    unittest.main()
