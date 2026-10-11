"""Offline exporter contract tests; never instantiate a network client.

Run: python3 -m unittest discover -s research/autoresearch/campaign -p test_lake_export.py -v
The small fixtures use the actual lake client's stdlib event/schema code. The
optional actual-run test patches manifest writes, so source evidence is read-only.
"""
import base64
from collections import Counter, defaultdict
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
import lake_export as export


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False) + '\n')


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def reassemble(rows):
    rows = sorted(rows, key=lambda row: row['chunk_index'])
    result = b''
    for index, row in enumerate(rows):
        assert row['chunk_index'] == index
        assert row['chunk_count'] == len(rows)
        assert row['offset_bytes'] == len(result)
        part = row['data'].encode('utf-8') if row['encoding'] == 'utf-8' else base64.b64decode(row['data'], validate=True)
        assert row['chunk_bytes'] == len(part)
        result += part
    assert all(row['file_bytes'] == len(result) and row['content_sha256'] == checksum(result) for row in rows)
    return result


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # client_module imports only stdlib code; credentials are loaded only by
        # LakeClient construction, which this test explicitly forbids.
        client = export.client_module()
        self.network_guard = patch.object(client, 'LakeClient', side_effect=AssertionError('Offline test attempted network access'))
        self.network_guard.start()
        self.plan = []
        self.put('protocol.json', {'seed': 7, 'rounds': 2})
        self.put('operations.json', {'oracle_manifest_sha256': 'a' * 64})
        self.put('pause.json', {'state': 'paused', 'interrupted_candidate': 2,
                              'cancelled_at_utc': '1970-01-01T00:33:20Z'})
        self.put('evidence/search/progress.json', {'active': 2, 'status': 'running', 'incumbent': 0,
                                                'incumbent_score': 10.0})
        self.put('evidence/preflight/baseline/benchmark.json', {'passed': True, 'score_seconds': 10.0})
        self.put('evidence/preflight/baseline/samples.json', [self.sample(0, 10.0)])
        self.put('evidence/calibration/seed-verification.json', self.verification(True))
        (self.root / 'events.jsonl').write_text('{"kind":"start","candidate":1}\n')
        for candidate in (1, 2):
            source = f'// candidate {candidate}\n'.encode()
            path = self.root / 'sources' / f'{candidate:04d}.cu'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
            candidate_patch = self.root / 'artifacts' / f'{candidate:04d}' / 'candidate.patch'
            candidate_patch.parent.mkdir(parents=True)
            candidate_patch.write_bytes(b'--- old\n+++ new\n')
            row = {'candidate': candidate, 'commit': str(candidate) * 40, 'parent_commit': '0' * 40,
                   'source_sha256': checksum(source), 'patch_sha256': checksum(candidate_patch.read_bytes()),
                   'config': {'elementwise_threads': 256}}
            self.plan.append(row)
            record = {**row, 'started_unix': 1000.0 + candidate * 100,
                      'terminal': candidate == 1, 'status': 'discard' if candidate == 1 else 'running',
                      'correctness': True, 'fullgate': False, 'score_seconds': 11.0 if candidate == 1 else None,
                      'benchmark_parent': 0}
            if candidate == 1:
                record.update(finished_unix=1190.0, reason='no measured improvement')
            self.put(f'evidence/detail/{candidate:04d}/' + ('result.json' if candidate == 1 else 'inflight.json'), record)
            self.put(f'evidence/detail/{candidate:04d}/screen/verification.json', self.verification(True))
            self.put(f'evidence/detail/{candidate:04d}/screen/units.json', self.verification(True))
            self.put(f'evidence/detail/{candidate:04d}/benchmark/samples.json',
                     [self.sample(i, 11.0 + i) for i in range(3 if candidate == 1 else 2)])
        self.put('plan.json', self.plan)
        self.repackage()

    def tearDown(self):
        self.network_guard.stop()
        self.temp.cleanup()

    def put(self, relative, obj):
        write_json(self.root / relative, obj)

    @staticmethod
    def sample(index, seconds):
        return {'index': index, 'seconds': seconds, 'seed': 7, 'prompt_sha256': 'b' * 64,
                'command': ['h3', '--seed', '7'], 'media': {'sha256': 'c' * 64}}

    @staticmethod
    def verification(passed):
        return {'passed': passed, 'comparisons': [{'label': 'velocity', 'passed': passed,
                'relative_l2': 0.0 if passed else 1.0, 'normalized_max': 0.0 if passed else 1.0}],
                'wall_seconds_including_checks_and_io': 2.0}

    def repackage(self):
        packages, cloud = [], []
        for candidate in (1, 2):
            detail = self.root / 'evidence/detail' / f'{candidate:04d}'
            members = []
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w') as archive:
                for path in sorted(detail.rglob('*')):
                    if not path.is_file():
                        continue
                    data = path.read_bytes()
                    name = str(path.relative_to(detail))
                    members.append({'path': name, 'size_bytes': len(data), 'sha256': checksum(data),
                                    'detail_path': str(path)})
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            data = buffer.getvalue()
            package_path = self.root / 'evidence/packages' / f'{candidate:04d}.tar'
            package_path.parent.mkdir(parents=True, exist_ok=True)
            package_path.write_bytes(data)
            package = {'candidate': candidate, 'sha256': checksum(data), 'size_bytes': len(data),
                       'local_path': str(package_path), 'members': members}
            packages.append(package)
            uri = 'gs://test-bucket/evidence/sha256/' + package['sha256'] + '/evidence.tar'
            md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
            cloud.append({'candidate': candidate, 'role': 'candidate', 'sha256': package['sha256'],
                          'size_bytes': len(data), 'storage_uri': uri,
                          'download_url': 'https://storage.cloud.google.com/' + uri[5:],
                          'generation': str(candidate), 'verification': {'passed': True,
                              'verified_generation': str(candidate), 'method': 'server_md5_and_size'},
                          'server_md5_base64': md5, 'local_md5_base64': md5})
        self.put('evidence/packages/manifest.json', {'packages': packages})
        self.put('evidence/packages/cloud-artifacts.json', {'status': 'PASS', 'artifacts': cloud})

    def build(self):
        return export.Export(self.root).build()

    def result(self, payload, candidate):
        return next(row for row in payload['streams']['attempts/results'] if row['candidate'] == candidate)

    def test_replay_has_identical_payload_and_unique_stable_ids(self):
        first, second = self.build(), self.build()
        self.assertEqual(first, second)
        for stream, rows in first['streams'].items():
            self.assertEqual(len(rows), len({r['event_id'] for r in rows}), stream)
        fresh = export.Export(self.root)
        fresh.add('test/rows', 'same', value=1)
        fresh.add('test/rows', 'same', value=1)
        self.assertEqual(len(fresh.streams['test/rows']), 1)

    def test_attempt_identity_separates_retry_and_requires_recorded_start(self):
        first = export.attempt_id(2, {'started_unix': 1200.0})
        self.assertEqual(first, export.attempt_id(2, {'started_unix': 1200.0}))
        self.assertNotEqual(first, export.attempt_id(2, {'started_unix': 1200.1}))
        with self.assertRaises(ValueError):
            export.attempt_id(2, {})

    def test_all_text_encodings_reconstruct_exact_original_bytes(self):
        fixtures = [b'', b'line1\r\nline2\n\x00', ('\ufeff中文🙂e\u0301\r\n' * 20000).encode(),
                    bytes(range(256)) * 700]
        for index, data in enumerate(fixtures):
            with self.subTest(index=index):
                path = self.root / f'text-{index}.log'
                path.write_bytes(data)
                obj = export.Export(self.root)
                self.assertEqual(obj.text(path), checksum(data))
                self.assertEqual(reassemble(obj.streams['raw/text_chunks']), data)
                count = len(obj.streams['raw/text_chunks'])
                obj.text(path)
                self.assertEqual(len(obj.streams['raw/text_chunks']), count)

    def test_interrupted_attempt_has_no_completion_or_score(self):
        payload = self.build()
        row = self.result(payload, 2)
        self.assertEqual(row['state'], 'interrupted')
        self.assertEqual(row['decision'], 'interrupted')
        self.assertFalse(row['counts_as_completed'])
        self.assertIsNone(row['score_seconds'])
        self.assertIsNone(row['round_seconds'])
        self.assertIsNone(row['correctness_full_passed'])
        samples = [r for r in payload['streams']['metrics/cold_samples'] if r['candidate'] == 2]
        self.assertEqual(len(samples), 2)
        states = [r['state'] for r in payload['streams']['attempts/lifecycle'] if r['candidate'] == 2]
        self.assertEqual(states, ['started', 'interrupted'])

    def test_baseline_not_counted_and_paused_has_no_active_execution(self):
        payload = self.build()
        self.assertFalse(self.result(payload, 0)['counts_as_completed'])
        row = payload['streams']['search/overview'][0]
        self.assertEqual((row['completed'], row['remaining'], row['discarded'], row['crashed']), (1, 1, 1, 0))
        self.assertEqual(row['status'], 'paused')
        self.assertIsNone(row['active'])

    def test_absent_and_failed_baseline_reports_never_create_correctness_pass(self):
        path = self.root / 'evidence/calibration/seed-verification.json'
        path.unlink()
        row = self.result(self.build(), 0)
        self.assertIsNone(row['correctness_full_passed'])
        self.assertIsNone(row['correctness_screen_passed'])
        write_json(path, self.verification(False))
        row = self.result(self.build(), 0)
        self.assertIs(row['correctness_full_passed'], False)
        self.assertFalse(row['counts_as_completed'])
        # A measured duration can remain as historical data, independent of eligibility.
        self.assertEqual(row['score_seconds'], 10.0)

    def test_record_flag_without_verification_does_not_create_a_pass(self):
        (self.root / 'evidence/detail/0002/screen/verification.json').unlink()
        self.repackage()
        row = self.result(self.build(), 2)
        self.assertIsNone(row['correctness_screen_passed'])
        self.assertIsNone(row['correctness_full_passed'])

    def test_changed_detail_bytes_are_rejected(self):
        (self.root / 'evidence/detail/0001/screen/verification.json').write_text('{"passed":false}')
        with self.assertRaises(ValueError):
            self.build()

    def test_required_missing_detail_is_rejected(self):
        (self.root / 'evidence/detail/0001/screen/units.json').unlink()
        with self.assertRaises(ValueError):
            self.build()

    def test_changed_planned_source_is_rejected(self):
        (self.root / 'sources/0001.cu').write_text('// silently changed\n')
        with self.assertRaises(ValueError):
            self.build()

    def test_invalid_cloud_verification_size_generation_or_uri_is_rejected(self):
        path = self.root / 'evidence/packages/cloud-artifacts.json'
        original = json.loads(path.read_text())
        changes = [('verification', {'passed': False}), ('size_bytes', 1), ('generation', '0'),
                   ('storage_uri', 'gs://test-bucket/other/evidence.tar')]
        for key, value in changes:
            with self.subTest(field=key):
                changed = json.loads(json.dumps(original))
                changed['artifacts'][0][key] = value
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    self.build()
        write_json(path, original)

    def test_reference_packages_never_create_search_results(self):
        path = self.root / 'evidence/packages/cloud-artifacts.json'
        cloud = json.loads(path.read_text())
        for role, char in [('oracle', 'd'), ('baseline', 'e')]:
            cloud['artifacts'].append({'candidate': 0, 'role': role, 'sha256': char * 64,
                'size_bytes': 1024, 'generation': '100', 'verification': {'passed': True},
                'storage_uri': f'gs://test-bucket/sha256/{char * 64}/evidence.tar',
                'download_url': f'https://storage.cloud.google.com/test-bucket/sha256/{char * 64}/evidence.tar'})
        write_json(path, cloud)
        payload = self.build()
        self.assertEqual(len([r for r in payload['streams']['packages/catalog'] if r['candidate'] == 0]), 2)
        self.assertEqual(len(payload['streams']['attempts/results']), 3)
        self.assertEqual(payload['streams']['search/overview'][0]['completed'], 1)

    def test_uninventoried_fullgate_cannot_create_a_pass(self):
        self.put('evidence/detail/0002/fullgate/verification.json', self.verification(True))
        # Deliberately do not repackage: this report is not in the immutable archive.
        with self.assertRaises(ValueError):
            self.build()

    def test_invalid_duration_rejected_without_inventing_samples(self):
        for value in [0.0, -1.0, float('nan'), float('inf'), True]:
            with self.subTest(value=value):
                self.put('bad-samples.json', [self.sample(0, value)])
                with self.assertRaises(ValueError):
                    export.Export(self.root).samples(self.root / 'bad-samples.json', candidate=1, attempt='test')

    def live_index(self, candidate=2, started=2100.0, state='running'):
        campaign_hash = 'f' * 64
        self.put('operations.json', {'oracle_manifest_sha256': 'a' * 64,
                                     'campaign_identity_sha256': campaign_hash})
        identity = {**self.plan[candidate - 1], 'started_unix': started,
                    'campaign_identity_sha256': campaign_hash}
        aid = export.attempt_id(candidate, identity)
        directory = self.root / 'evidence/telemetry'
        metadata = {}
        for name, content in [('screen/verification.json', self.verification(True)),
                              ('benchmark/samples.json', [self.sample(0, 9.5)])]:
            data = json.dumps(content).encode()
            checksum_value = checksum(data)
            relative = 'blobs/' + checksum_value + '.json'
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            metadata[name] = {'blob': relative, 'sha256': checksum_value, 'bytes': len(data)}
        index = {'candidate': candidate, 'attempt_id': aid, 'identity': identity,
                 'metadata': metadata, 'state': state}
        path = directory / 'attempts' / aid / 'index.json'
        write_json(path, index)
        return path, index

    def test_live_snapshot_does_not_count_or_supply_final_score(self):
        _, index = self.live_index(state='completed')
        # Even an observer's completed state cannot substitute for the complete archive.
        first, second = self.build(), self.build()
        self.assertEqual(first, second)
        rows = [r for r in first['streams']['attempts/results'] if r['attempt_id'] == index['attempt_id']]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['counts_as_completed'])
        self.assertIsNone(rows[0]['score_seconds'])
        self.assertTrue(rows[0]['correctness_screen_passed'])
        self.assertIsNone(rows[0]['correctness_full_passed'])
        self.assertEqual(first['streams']['search/overview'][0]['completed'], 1)
        samples = [r for r in first['streams']['metrics/cold_samples'] if r['attempt_id'] == index['attempt_id']]
        self.assertEqual([r['seconds'] for r in samples], [9.5])

    def test_live_blob_mutation_is_rejected_before_using_its_pass(self):
        _, index = self.live_index()
        item = index['metadata']['screen/verification.json']
        path = self.root / 'evidence/telemetry' / item['blob']
        path.write_bytes(path.read_bytes().replace(b'true', b'null'))
        with self.assertRaises(ValueError):
            self.build()

    def test_live_blob_cannot_escape_frozen_blob_namespace(self):
        index_path, index = self.live_index()
        original = self.root / 'evidence/telemetry' / index['metadata']['screen/verification.json']['blob']
        outside = self.root / 'evidence/not-a-frozen-blob.json'
        outside.write_bytes(original.read_bytes())
        index['metadata']['screen/verification.json']['blob'] = '../not-a-frozen-blob.json'
        write_json(index_path, index)
        with self.assertRaises(ValueError):
            self.build()

    def test_live_source_campaign_and_attempt_identity_must_match(self):
        path, original = self.live_index()
        for section, key, value in [('identity', 'commit', '9' * 40),
                                    ('identity', 'source_sha256', '0' * 64),
                                    ('identity', 'campaign_identity_sha256', '0' * 64),
                                    (None, 'attempt_id', 'invented-attempt')]:
            with self.subTest(key=key):
                index = json.loads(json.dumps(original))
                (index[section] if section else index)[key] = value
                write_json(path, index)
                with self.assertRaises(ValueError):
                    self.build()

    def test_archived_attempt_suppresses_duplicate_live_result_and_metrics(self):
        before = self.build()
        self.live_index(candidate=1, started=1100.0, state='completed')
        after = self.build()
        for stream in ['attempts/results', 'metrics/cold_samples', 'metrics/verification',
                       'metrics/numerical_checks', 'search/overview']:
            self.assertEqual(before['streams'][stream], after['streams'][stream], stream)


class ActualReadOnlyTests(unittest.TestCase):
    def test_paused_real_run_counts_samples_and_raw_text_integrity(self):
        root = Path(__file__).resolve().parents[1] / 'runs/20261010-100'
        if not (root / 'evidence/packages/cloud-artifacts.json').is_file():
            self.skipTest('Actual paused-run fixture is not installed')
        client = export.client_module()
        writes = []
        with patch.object(client, 'LakeClient', side_effect=AssertionError('Network forbidden')):
            with patch.object(export, 'write', side_effect=lambda path, value: writes.append((path, value))):
                first, second = export.Export(root).build(), export.Export(root).build()
        self.assertEqual(first, second)
        self.assertTrue(all('/lake/packages/' in str(path) for path, _ in writes))
        results = {r['candidate']: r for r in first['streams']['attempts/results']}
        self.assertEqual(set(results), {0, 1, 2})
        self.assertEqual(results[1]['decision'], 'discard')
        self.assertEqual(results[2]['state'], 'interrupted')
        self.assertIsNone(results[2]['score_seconds'])
        self.assertFalse(results[2]['counts_as_completed'])
        self.assertIsNone(results[1]['correctness_full_passed'])
        self.assertIsNone(results[2]['correctness_full_passed'])
        self.assertTrue(results[0]['correctness_full_passed'])
        counts = Counter(row['candidate'] for row in first['streams']['metrics/cold_samples'])
        self.assertEqual(counts, {0: 3, 1: 3, 2: 2})
        overview = first['streams']['search/overview'][0]
        self.assertEqual((overview['completed'], overview['total']), (1, 100))
        self.assertIsNone(overview['active'])
        grouped = defaultdict(list)
        for row in first['streams']['raw/text_chunks']:
            grouped[(row['candidate'], row['attempt_id'], row['source_path'], row['content_sha256'])].append(row)
        for (candidate, attempt, logical, _), rows in grouped.items():
            raw = reassemble(rows)
            path = root / 'evidence/detail' / f'{candidate:04d}' / logical if candidate and attempt != 'source' else root / logical
            if path.is_file():
                self.assertEqual(raw, path.read_bytes(), str(path))
        packages = first['streams']['packages/catalog']
        self.assertEqual(len(packages), 4)
        self.assertTrue(all(row['storage_uri'].startswith('gs://') for row in packages))


if __name__ == '__main__':
    unittest.main()
