"""Offline archive collection, replay, and preservation tests. No SSH or GCS."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import lake_collect as collect


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, data, kind in entries:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                entry.linkname = '../../outside.txt'
            entry.size = len(data) if kind == tarfile.REGTYPE else 0
            archive.addfile(entry, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return buffer.getvalue()


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_detail = self.root / 'evidence/detail/0002'
        self.old_record = {'candidate': 2, 'commit': '2' * 40, 'terminal': False,
                           'status': 'running', 'started_unix': 1000.0, 'score_seconds': None}
        write_json(self.old_detail / 'inflight.json', self.old_record)
        (self.old_detail / 'old.log').write_bytes(b'paused old attempt\n')
        self.old_bytes = {p.name: p.read_bytes() for p in self.old_detail.iterdir()}
        write_json(self.root / 'evidence/packages/manifest.json', {'packages': [
            {'candidate': 2, 'sha256': 'a' * 64, 'size_bytes': 100,
             'detail_root': str(self.old_detail), 'execution_state': 'interrupted', 'members': []}]})
        self.record = {'candidate': 2, 'commit': '2' * 40, 'terminal': True,
                       'status': 'discard', 'started_unix': 2000.0, 'finished_unix': 2020.0,
                       'score_seconds': 12.0}
        write_json(self.root / 'evidence/search/0002/result.json', self.record)
        self.aid = collect.attempt_id(2, self.record)
        self.target = self.root / 'evidence/packages' / (self.aid + '-evidence.tar')
        self.process_guard = patch.object(collect.subprocess, 'Popen', side_effect=AssertionError('SSH forbidden'))
        self.process_guard.start()

    def tearDown(self):
        self.process_guard.stop()
        self.temp.cleanup()

    def archive(self, record=None):
        return tar_bytes([('result.json', json.dumps(self.record if record is None else record).encode(), tarfile.REGTYPE),
                          ('screen/verification.json', b'{"passed":true}\n', tarfile.REGTYPE),
                          ('screen/latent.safetensors', b'binary latent\x00\xff', tarfile.REGTYPE)])

    def download(self, number, target):
        self.assertEqual(number, 2)
        data = self.archive()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        metadata = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
        write_json(target.with_suffix('.transfer.json'), metadata)
        return metadata

    def assert_old_attempt_unchanged(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.old_detail.iterdir()}, self.old_bytes)

    def test_new_retry_keeps_interrupted_attempt_and_replay_is_noop(self):
        with patch.object(collect, 'download', side_effect=self.download) as download:
            first = collect.collect(self.root, upload=False)
            second = collect.collect(self.root, upload=False)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first['packages']), 2)
        new = first['packages'][1]
        self.assertEqual(new['attempt_id'], self.aid)
        self.assertNotEqual(new['detail_root'], str(self.old_detail))
        self.assertEqual(new['execution_state'], 'completed')
        self.assert_old_attempt_unchanged()
        self.assertEqual(new['file_count'], 3)
        self.assertEqual(new['compact_file_count'], 2)
        self.assertFalse((Path(new['detail_root']) / 'screen/latent.safetensors').exists())
        self.assertEqual({m['path']: m['sha256'] for m in new['members']}['screen/latent.safetensors'],
                         hashlib.sha256(b'binary latent\x00\xff').hexdigest())

    def test_archived_terminal_result_must_equal_synced_result(self):
        wrong = dict(self.record, score_seconds=99.0)
        data = self.archive(wrong)
        self.target.write_bytes(data)
        write_json(self.target.with_suffix('.transfer.json'),
                   {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)})
        before = (self.root / 'evidence/packages/manifest.json').read_bytes()
        with self.assertRaises(ValueError):
            collect.collect(self.root, upload=False)
        self.assertEqual((self.root / 'evidence/packages/manifest.json').read_bytes(), before)
        self.assert_old_attempt_unchanged()

    def test_corrupt_retained_archive_fails_before_extract_or_upload(self):
        self.download(2, self.target)
        with self.target.open('ab') as stream:
            stream.write(b'corruption')
        with patch.object(collect, 'inspect_archive', side_effect=AssertionError('Unverified bytes reached extraction')):
            with self.assertRaises(ValueError):
                collect.collect(self.root, upload=False)
        self.assert_old_attempt_unchanged()

    def test_missing_published_archive_does_not_create_result_or_package(self):
        before = (self.root / 'evidence/packages/manifest.json').read_bytes()
        with patch.object(collect, 'download', return_value=None):
            result = collect.collect(self.root, upload=False)
        self.assertEqual(len(result['packages']), 1)
        self.assertEqual((self.root / 'evidence/packages/manifest.json').read_bytes(), before)
        self.assert_old_attempt_unchanged()

    def test_traversal_absolute_symlink_hardlink_and_duplicates_rejected(self):
        bad_entries = [[('../outside.json', b'bad', tarfile.REGTYPE)],
                       [('/absolute.json', b'bad', tarfile.REGTYPE)],
                       [('link.json', b'', tarfile.SYMTYPE)],
                       [('hard.json', b'', tarfile.LNKTYPE)],
                       [('same.json', b'one', tarfile.REGTYPE), ('./same.json', b'two', tarfile.REGTYPE)]]
        for index, entries in enumerate(bad_entries):
            with self.subTest(index=index):
                path = self.root / f'bad-{index}.tar'
                path.write_bytes(tar_bytes(entries))
                with self.assertRaises(ValueError):
                    collect.inspect_archive(path, self.root / f'extract-{index}')
        self.assertFalse((self.root / 'outside.json').exists())

    def test_existing_symlink_directory_cannot_redirect_extraction(self):
        outside = self.root / 'outside'
        outside.mkdir()
        destination = outside / 'data.json'
        destination.write_bytes(b'original')
        detail = self.root / 'symlink-detail'
        detail.mkdir()
        (detail / 'nested').symlink_to(outside, target_is_directory=True)
        archive = self.root / 'safe-name.tar'
        archive.write_bytes(tar_bytes([('nested/data.json', b'overwrite', tarfile.REGTYPE)]))
        try:
            collect.inspect_archive(archive, detail)
        except (ValueError, OSError):
            pass
        self.assertEqual(destination.read_bytes(), b'original')

    def test_existing_pending_symlink_cannot_overwrite_external_file(self):
        outside = self.root / 'outside.txt'
        outside.write_bytes(b'original')
        detail = self.root / 'pending-detail'
        detail.mkdir()
        (detail / 'data.json.pending').symlink_to(outside)
        archive = self.root / 'pending.tar'
        archive.write_bytes(tar_bytes([('data.json', b'overwrite', tarfile.REGTYPE)]))
        try:
            collect.inspect_archive(archive, detail)
        except (ValueError, OSError):
            pass
        self.assertEqual(outside.read_bytes(), b'original')

    def test_failed_extraction_does_not_publish_partial_detail(self):
        detail = self.root / 'atomic-detail'
        archive = self.root / 'late-invalid.tar'
        archive.write_bytes(tar_bytes([('valid.json', b'{}', tarfile.REGTYPE),
                                       ('../escape.json', b'bad', tarfile.REGTYPE)]))
        with self.assertRaises(ValueError):
            collect.inspect_archive(archive, detail)
        self.assertFalse(detail.exists())
        self.assertEqual(list(detail.parent.glob('.extract-*')), [])

    def test_retry_rejects_stale_uninventoried_detail(self):
        detail = self.root / 'stale-detail'
        detail.mkdir()
        stale = detail / 'verification.json'
        stale.write_bytes(b'{"passed":true}')
        archive = self.root / 'no-stale-file.tar'
        archive.write_bytes(tar_bytes([('result.json', b'{}', tarfile.REGTYPE)]))
        with self.assertRaises(ValueError):
            collect.inspect_archive(archive, detail)
        self.assertEqual(stale.read_bytes(), b'{"passed":true}')
        self.assertFalse((detail / 'result.json').exists())

    def test_corrupt_existing_content_addressed_outbox_is_rejected(self):
        payload = {'experiment_id': 'offline', 'streams': {'test/rows': [{'event_id': 'abc'}]}}
        identity = collect.sha(collect.canonical(payload).encode())
        target = self.root / 'lake/outbox' / (identity + '.json')
        write_json(target, {'corrupt': True})
        with patch.object(collect, 'Export') as exporter:
            exporter.return_value.build.return_value = payload
            with self.assertRaises(ValueError):
                collect.persist_payload(self.root)
        self.assertEqual(json.loads(target.read_text()), {'corrupt': True})


if __name__ == '__main__':
    unittest.main()
