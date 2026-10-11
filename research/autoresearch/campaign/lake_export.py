"""Build durable AutoGrad lake packages outside native inference.

Only this module's send command contacts the lake. Building payloads never
executes C, CUDA, a reference model, or a benchmark. Binary evidence stays in
private object storage; its complete file inventory and all available text
evidence are represented as lake rows.
"""
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys

EXPERIMENT = 'h3-h200-cold-search-20261010-100'
TRAINING_RUN = 'h3-cuda-autoresearch-20261010-100'
SKILL = Path('/home/homo/.codex/skills/training-lake-reporting')


def client_module():
    sys.path.insert(0, str(SKILL / 'scripts'))
    import lake_client
    return lake_client


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            value.update(part)
    return value.hexdigest()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else None


def attempt_id(candidate, record):
    if record.get('started_unix') is None:
        raise ValueError('An actual attempt must have its recorded start time')
    return f'{candidate:04d}-' + sha(canonical(record['started_unix']).encode())[:16]


def verified_cloud(item, package=None):
    if item.get('verification', {}).get('passed') is not True:
        raise ValueError('Cloud artifact has no successful integrity verification')
    if not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')):
        raise ValueError('Cloud artifact SHA256 is invalid')
    if not str(item.get('generation', '')).isdigit() or int(item['generation']) < 1:
        raise ValueError('Cloud artifact has no immutable generation')
    uri = item.get('storage_uri', '')
    if not uri.startswith('gs://') or '/' + item['sha256'] + '/' not in uri:
        raise ValueError('Cloud artifact URI does not identify its content')
    if package and any(item.get(key) != package.get(key) for key in ('sha256', 'size_bytes')):
        raise ValueError('Cloud artifact differs from package hash or size')
    return item


class Export:
    def __init__(self, root, experiment=EXPERIMENT):
        self.root = Path(root).resolve()
        self.experiment = experiment
        self.streams = defaultdict(list)
        self.seen = set()
        self.texts = set()
        self.client = client_module()

    def add(self, stream, key, **fields):
        row = self.client.event(stream, key, experiment_id=self.experiment,
                                training_run_id=TRAINING_RUN, **fields)
        identity = (stream, row['event_id'])
        if identity not in self.seen:
            self.streams[stream].append(row)
            self.seen.add(identity)
        return row

    def text(self, path, *, candidate=0, attempt='campaign', logical_path=None):
        path = Path(path)
        if not path.is_file():
            return None
        data = path.read_bytes()
        content_hash = sha(data)
        logical_path = logical_path or str(path.relative_to(self.root))
        identity = (candidate, attempt, logical_path, content_hash)
        if identity in self.texts:
            return content_hash
        self.texts.add(identity)
        try:
            decoded = data.decode('utf-8')
            parts = [decoded[index:index + 60000] for index in range(0, len(decoded), 60000)] or ['']
            encoding = 'utf-8'
        except UnicodeDecodeError:
            parts = [base64.b64encode(data[index:index + 45000]).decode('ascii')
                     for index in range(0, len(data), 45000)] or ['']
            encoding = 'base64'
        offset = 0
        for index, part in enumerate(parts):
            length = len(part.encode('utf-8')) if encoding == 'utf-8' else len(base64.b64decode(part))
            self.add('raw/text_chunks', f'{candidate}/{attempt}/{logical_path}/{content_hash}/{index}',
                     candidate=candidate, attempt_id=attempt, source_path=logical_path,
                     content_sha256=content_hash, encoding=encoding, chunk_index=index,
                     chunk_count=len(parts), offset_bytes=offset, chunk_bytes=length,
                     file_bytes=len(data), data=part)
            offset += length
        if offset != len(data):
            raise ValueError('Text chunks do not reconstruct the exact source bytes')
        return content_hash

    def numeric_checks(self, obj, *, candidate, attempt, gate, location='root'):
        if isinstance(obj, list):
            for index, value in enumerate(obj):
                self.numeric_checks(value, candidate=candidate, attempt=attempt, gate=gate,
                                    location=f'{location}/{index}')
        elif isinstance(obj, dict):
            if isinstance(obj.get('passed'), bool) and any(key in obj for key in
                    ('relative_l2', 'normalized_max', 'max_abs', 'max_abs_error', 'max_error', 'reference',
                     'sigma_exact', 'updates_exact', 'roundtrip_exact', 'exact', 'tokens')):
                self.add('metrics/numerical_checks', f'{attempt}/{gate}/{location}',
                         candidate=candidate, attempt_id=attempt, gate=gate, case=location,
                         label=str(obj.get('label', obj.get('case', obj.get('name', location)))),
                         passed=obj['passed'], relative_l2=obj.get('relative_l2'),
                         normalized_max=obj.get('normalized_max'), details=obj)
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    self.numeric_checks(value, candidate=candidate, attempt=attempt, gate=gate,
                                        location=f'{location}/{key}')

    def verification(self, path, *, candidate, attempt, gate):
        report = read(path)
        if report is None:
            return None
        count_before = len(self.streams['metrics/numerical_checks'])
        self.numeric_checks(report, candidate=candidate, attempt=attempt, gate=gate)
        count = len(self.streams['metrics/numerical_checks']) - count_before
        self.add('metrics/verification', f'{attempt}/{gate}', candidate=candidate,
                 attempt_id=attempt, gate=gate, passed=report.get('passed'),
                 scope='complete 6+1 regression' if gate == 'fullgate' else
                       'frozen primitive, sampler and layout units' if gate == 'units' else 'fixed target scenario',
                 seconds=report.get('wall_seconds_including_checks_and_io'), comparison_count=count,
                 report_sha256=digest(path), report=report)
        return report.get('passed')

    def samples(self, path, *, candidate, attempt, phase='preliminary', role='candidate', pair=None):
        records = read(path, [])
        for row in records:
            value = row['seconds']
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('A timing sample must be a measured positive finite duration')
            self.add('metrics/cold_samples', f'{attempt}/{phase}/{pair}/{role}/{row["index"]}',
                     candidate=candidate, attempt_id=attempt, phase=phase, role=role,
                     index=row['index'], pair=pair, seconds=value, seed=row.get('seed'),
                     prompt_sha256=row.get('prompt_sha256'), command=row.get('command'),
                     mp4_sha256=row.get('media', {}).get('sha256'), media=row.get('media'))

    def common(self):
        root = self.root
        protocol = read(root / 'protocol.json')
        plan = read(root / 'plan.json')
        operations = read(root / 'operations.json', {})
        self.add('experiment/identity', 'identity-v1', name='H3 native CUDA / H200 cold-start search',
                 description='100 distinct native backend candidates; fixed PyTorch correctness gate and one cold-process E2E score.',
                 kind='historical_backfill_then_live', method='custom', seed=7,
                 experiment_type='inference_code_optimization', planned_candidates=len(plan),
                 has_training=False, run_path=str(root), parent_experiment=None)
        reporting_plan = read(root / 'artifacts/reporting-plan.json', {})
        self.add('experiment/configuration', 'frozen-protocol', protocol=protocol,
                 protocol_sha256=digest(root / 'protocol.json'), reporting_plan=reporting_plan,
                 custom_definition={'objective': 'minimize median cold-process end-to-end seconds after correctness PASS',
                     'unit': 'seconds', 'correctness': 'hard eligibility constraint',
                     'inference_steps': 20, 'transformer_layers': 50,
                     'training_metrics': 'not applicable: no optimizer or parameter updates',
                     'attempt_identity': 'candidate plus recorded execution start; retries are separate executions',
                     'round_count': 'only terminal candidate decisions count; planned candidates and interrupted attempts do not',
                     'telemetry_boundary': 'external controller and sender; no C/CUDA hooks or HTTP inside timed inference'})
        self.add('runtime/environment', 'campaign-environment', environment=operations,
                 gpu='H200', gpu_count=1, native_runtime='C/C++ and CUDA; no PyTorch in scored process')
        self.add('model/roles', 'roles', roles={'candidate': 'native C/CUDA inference',
                 'reference': 'fixed cached PyTorch tensors', 'trainer': None, 'optimizer': None},
                 checkpoint_revision='42ed227ee7df40d41602854ae760620d6eb651fe')
        self.add('checkpoint/catalog', 'fixed-inference-weights',
                 checkpoint_id='h3-42ed227ee7df40d41602854ae760620d6eb651fe',
                 name='Fixed H3 inference weights',
                 description='Identical pretrained weights for the native candidates and fixed PyTorch reference. No weight updates during this search.',
                 role='initial_and_evaluated', step=None, owner_training_run_id=None,
                 path='/var/tmp/h3-cuda-14228/local-model/FL2VA',
                 revision='42ed227ee7df40d41602854ae760620d6eb651fe',
                 optimizer_state=False, scheduler_state=False, training_resume=False,
                 provenance='Accepted phase-one model, pinned repository revision',
                 checksum=None, checksum_reason='No separate complete weight-file checksum manifest was collected.')
        self.add('data/splits', 'fixed-golden-oracle', source='pinned PyTorch reference', split='regression',
                 snapshot_sha256=operations.get('oracle_manifest_sha256'),
                 snapshot_path=str(root / 'evidence/calibration/oracle-manifest.json'),
                 cases=7, model_steps_per_case=20, selection='entire fixed fixture matrix')
        gaps = [('training/updates', 'not_applicable', 'This is inference code search, with no training or optimizer steps.'),
                ('data/locations', 'unavailable', 'Fixed PyTorch fixtures have a frozen file manifest. No verified Langfuse dataset ID is available; no ID or URL is invented.'),
                ('runtime/resources/per_sample', 'unavailable', 'Per-sample GPU utilization and peak memory were not measured; they are not filled with zero.'),
                ('timing/cold_storage', 'not_applicable', 'Each scored sample uses a new process/model/CUDA state. Shared OS/filesystem page caches are not forcibly evicted.'),
                ('binary_weights', 'not_applicable', 'Model weights are identified by pinned revision and existing storage path; they are not copied into lake JSON.')]
        for field, status, reason in gaps:
            self.add('collection/coverage', field, field=field, status=status, reason=reason)
        for row in plan:
            candidate = row['candidate']
            planned = dict(row)
            planned.setdefault('parent', row.get('suggested_parent', 0))
            planned.setdefault('hypothesis', '')
            self.add('search/candidates', str(candidate), **planned, record_type='candidate_plan')
            source = root / 'sources' / f'{candidate:04d}.cu'
            patch = root / 'artifacts' / f'{candidate:04d}' / 'candidate.patch'
            if digest(source) != row['source_sha256']:
                raise ValueError('Candidate source differs from frozen plan')
            self.add('provenance/code', f'candidate/{candidate}', candidate=candidate,
                     commit=row['commit'], parent_commit=row['parent_commit'], source_path='cuda/h3_gpu_cuda.cu',
                     source_sha256=self.text(source, candidate=candidate, attempt='source'),
                     patch_sha256=self.text(patch, candidate=candidate, attempt='source'))
        for relative in ('protocol.json', 'plan.json', 'operations.json', 'control-snapshot.json',
                         'telemetry-control-snapshot.json', 'lake/telemetry-plan.json', 'pause.json',
                         'evidence/slurm-accounting.txt', 'evidence/calibration/oracle-manifest.json',
                         'evidence/calibration/oracle-prepare.json', 'evidence/calibration/audit.json',
                         'artifacts/reporting-plan.json', 'evidence/preflight/build/build.json',
                         'evidence/packages/manifest.json', 'evidence/packages/reference-packages-manifest.json',
                         'evidence/packages/cloud-artifacts.json', 'evidence/packages/cloud-artifacts-audit.json'):
            self.text(root / relative)
        telemetry_control = read(root / 'telemetry-control-snapshot.json')
        if telemetry_control:
            self.add('provenance/controller', telemetry_control['commit'], **telemetry_control,
                     placement='external observer/collector; frozen native timing and judge unchanged')
        for directory in ('evidence/infrastructure', 'evidence/preflight/baseline'):
            for path in sorted((root / directory).rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.log', '.txt'):
                    self.text(path)
        for index, line in enumerate((root / 'events.jsonl').read_text().splitlines()):
            obj = json.loads(line)
            self.add('raw/ledger_events', str(index), index=index, source_sha256=sha(line.encode()), event=obj)
        accounting = root / 'evidence/slurm-accounting.txt'
        if accounting.is_file():
            names = ('job_id', 'name', 'partition', 'node', 'cpus', 'allocated_resources',
                     'requested_memory', 'submitted_utc', 'started_utc', 'finished_utc',
                     'elapsed', 'state', 'exit_code')
            for line in accounting.read_text().splitlines():
                values = line.rstrip('|').split('|')
                if len(values) != len(names):
                    raise ValueError('Unexpected Slurm accounting field count')
                row = dict(zip(names, values))
                self.add('runtime/slurm', sha(line.encode()), **row, counts_as_search_iteration=False)
        baseline = read(root / 'evidence/preflight/baseline/benchmark.json', {})
        self.samples(root / 'evidence/preflight/baseline/samples.json', candidate=0,
                     attempt='baseline-14325', phase='baseline', role='baseline')
        baseline_correct = self.verification(root / 'evidence/calibration/seed-verification.json', candidate=0,
                                            attempt='baseline-14325', gate='fullgate')
        self.add('attempts/results', 'baseline-14325', candidate=0, attempt_id='baseline-14325',
                 state='baseline', decision='baseline', counts_as_completed=False,
                 correctness_screen_passed=None, correctness_full_passed=baseline_correct,
                 score_seconds=baseline.get('score_seconds') if baseline.get('passed') is True else None, round_seconds=None,
                 reason='Measured native seed baseline, not a search round; correctness is recorded from its fixed fixture gate',
                 commit='c2b3c4bfb979de1923a4e4c0b85d20c914d6c108', benchmark_parent=None)
        references = read(root / 'evidence/packages/reference-text-inventory.json', {'artifacts': []})
        cloud = read(root / 'evidence/packages/cloud-artifacts.json', {'artifacts': []})
        by_hash = {item['sha256']: item for item in cloud['artifacts']}
        for archive in references['artifacts']:
            remote = by_hash.get(archive['archive_sha256'])
            if remote is None:
                raise ValueError('Reference inventory has no verified cloud archive')
            verified_cloud(remote, {'sha256': archive['archive_sha256'], 'size_bytes': archive['archive_size_bytes']})
            for member in archive['members']:
                self.add('artifacts/catalog', f"shared-{archive['archive_sha256']}/{member['path']}",
                         candidate=0, attempt_id='shared-' + archive['role'], path=member['path'],
                         sha256=member['sha256'], bytes=member['size_bytes'],
                         media_type=mimetypes.guess_type(member['path'])[0] or 'application/octet-stream',
                         package_id='sha256:' + archive['archive_sha256'], access_status='private_cloud_archive_member',
                         storage_uri=remote['storage_uri'], download_url=remote['download_url'])
            for member in archive['text_files']:
                path = Path(member['local_path'])
                if not path.is_file() or digest(path) != member['sha256'] or path.stat().st_size != member['size_bytes']:
                    raise ValueError('Reference text differs from frozen inventory')
                self.text(path, candidate=0, attempt='shared-' + archive['role'], logical_path=member['path'])
                if archive['role'] == 'baseline' and member['path'] == 'fullgate/units.json':
                    self.verification(path, candidate=0, attempt='baseline-14325', gate='units')
        return plan

    def packages(self, plan):
        root = self.root
        inventory = read(root / 'evidence/packages/manifest.json', {'packages': []})
        cloud = read(root / 'evidence/packages/cloud-artifacts.json', {'artifacts': []})
        by_hash = {item['sha256']: item for item in cloud['artifacts']}
        by_candidate = {row['candidate']: row for row in plan}
        pause = read(root / 'pause.json', {})
        for package in inventory['packages']:
            number = package['candidate']
            detail = Path(package.get('detail_root', root / 'evidence/detail' / f'{number:04d}'))
            anchored = {member['path']: member for member in package['members']}
            for path in detail.rglob('*'):
                if path.is_file():
                    relative = str(path.relative_to(detail))
                    member = anchored.get(relative)
                    if not member or path.stat().st_size != member['size_bytes'] or digest(path) != member['sha256']:
                        raise ValueError('Extracted evidence is not anchored to the immutable package inventory')
            record = read(detail / 'result.json') or read(detail / 'infrastructure-error.json') or read(detail / 'inflight.json')
            if not record or record.get('commit') != by_candidate[number]['commit']:
                raise ValueError('Attempt package source identity differs from plan')
            aid = attempt_id(number, record)
            terminal = record.get('terminal') is True
            if terminal and record.get('status') not in ('keep', 'discard', 'crash'):
                raise ValueError('Malformed terminal decision')
            is_paused = (not terminal and pause.get('interrupted_candidate') == number and
                record['started_unix'] < datetime.fromisoformat(pause['cancelled_at_utc'].replace('Z', '+00:00')).timestamp())
            state = 'completed' if terminal else package.get('execution_state') or ('interrupted' if is_paused else 'running')
            decision = record.get('status') if terminal else state
            ended = utc(record.get('finished_unix')) if terminal else pause.get('cancelled_at_utc') if is_paused else None
            self.add('attempts/lifecycle', f'{aid}/start', candidate=number, attempt_id=aid,
                     state='started', phase='build', source_time_utc=utc(record['started_unix']))
            if terminal or state in ('interrupted', 'infrastructure_error'):
                self.add('attempts/lifecycle', f'{aid}/{state}', candidate=number, attempt_id=aid,
                         state=state, phase=decision, source_time_utc=ended)
            screen = self.verification(detail / 'screen/verification.json', candidate=number, attempt=aid, gate='screen')
            full = self.verification(detail / 'fullgate/verification.json', candidate=number, attempt=aid, gate='fullgate')
            self.verification(detail / 'screen/units.json', candidate=number, attempt=aid, gate='units')
            self.samples(detail / 'benchmark/samples.json', candidate=number, attempt=aid)
            for path in sorted((detail / 'confirmation').glob('*/samples.json')):
                pair, role = path.parent.name.split('-', 1)
                self.samples(path, candidate=number, attempt=aid, phase='confirmation', role=role, pair=int(pair))
            for index, pair in enumerate(record.get('confirmation', {}).get('pairs', [])):
                self.add('metrics/promotion_pairs', f'{aid}/{index}', candidate=number, attempt_id=aid,
                         index=index, parent_seconds=pair['parent'], candidate_seconds=pair['candidate'],
                         difference_seconds=pair['parent'] - pair['candidate'])
            self.add('attempts/results', f'{aid}/{state}', candidate=number, attempt_id=aid, state=state,
                     decision=decision, counts_as_completed=terminal, correctness_screen_passed=screen,
                     correctness_full_passed=full, score_seconds=record.get('score_seconds') if terminal else None,
                     round_seconds=record['finished_unix'] - record['started_unix'] if terminal else None,
                     reason=record.get('reason') if terminal else package.get('reason') or
                            ('User requested pause to add lake reporting and dashboard' if is_paused else 'Execution in progress'),
                     commit=record['commit'], benchmark_parent=record.get('benchmark_parent'), record=record)
            remote = by_hash.get(package['sha256'], {})
            if remote:
                verified_cloud(remote, package)
            members = []
            for member in package['members']:
                path = detail / member['path']
                if member.get('detail_path'):
                    if not path.is_file():
                        raise ValueError('Extracted artifact required by package inventory is missing')
                    if path.stat().st_size != member['size_bytes'] or digest(path) != member['sha256']:
                        raise ValueError('Extracted artifact differs from immutable package inventory')
                    self.text(path, candidate=number, attempt=aid, logical_path=member['path'])
                members.append({key: member[key] for key in ('path', 'size_bytes', 'sha256')})
            manifest = {'schema_version': 1, 'candidate': number, 'attempt_id': aid, 'state': state,
                        'recorded_start_utc': utc(record['started_unix']), 'recorded_end_utc': ended,
                        'source': by_candidate[number], 'protocol_sha256': digest(root / 'protocol.json'),
                        'oracle_manifest_sha256': read(root / 'operations.json')['oracle_manifest_sha256'],
                        'archive_sha256': package['sha256'], 'archive_bytes': package['size_bytes'],
                        'storage_uri': remote.get('storage_uri'), 'download_url': remote.get('download_url'),
                        'storage_verification': remote.get('verification'), 'artifacts': members,
                        'complete_execution': terminal, 'recorded_artifacts_complete': True,
                        'coverage': 'All recorded files included. An interrupted execution has no invented final score or fullgate.'}
            manifest_hash = sha(canonical(manifest).encode())
            package_id = 'sha256:' + manifest_hash
            write(root / 'lake/packages' / aid / (manifest_hash + '.json'), manifest)
            self.add('packages/catalog', package_id, candidate=number, attempt_id=aid, package_id=package_id,
                     state=state, artifact_count=len(members), total_bytes=package['size_bytes'],
                     manifest_sha256=manifest_hash, archive_sha256=package['sha256'],
                     storage_uri=remote.get('storage_uri'), download_url=remote.get('download_url'),
                     complete=terminal, manifest=manifest)
            for member in members:
                self.add('artifacts/catalog', f'{package_id}/{member["path"]}', candidate=number, attempt_id=aid,
                         path=member['path'], sha256=member['sha256'], bytes=member['size_bytes'],
                         media_type=mimetypes.guess_type(member['path'])[0] or 'application/octet-stream',
                         package_id=package_id, access_status='private_cloud_archive_member' if remote else 'cluster_archive_only',
                         storage_uri=remote.get('storage_uri'), download_url=remote.get('download_url'))
            if not remote:
                self.add('collection/coverage', f'{aid}/cloud', field='packages/cloud_upload', status='pending',
                         reason='Archive retained locally and on the cluster; no verified cloud object is recorded yet.', attempt_id=aid)
        for item in cloud['artifacts']:
            if item.get('candidate') == 0:
                verified_cloud(item)
                self.add('packages/catalog', 'shared-' + item['sha256'], candidate=0,
                         attempt_id='shared-' + item['role'], package_id='sha256:' + item['sha256'], state='reference',
                         artifact_count=None, total_bytes=item['size_bytes'], manifest_sha256=None,
                         archive_sha256=item['sha256'], storage_uri=item['storage_uri'],
                         download_url=item['download_url'], complete=True, manifest=item)

    def overview(self, plan):
        progress = read(self.root / 'evidence/search/progress.json', {})
        pause = read(self.root / 'pause.json', {})
        results = [row for row in self.streams['attempts/results'] if row['counts_as_completed']]
        candidate_ids = {row['candidate'] for row in results}
        if len(candidate_ids) != len(results):
            raise ValueError('More than one terminal result exists for one candidate')
        baseline = read(self.root / 'evidence/preflight/baseline/benchmark.json', {}).get('score_seconds')
        paused = pause.get('state') == 'paused'
        snapshot = {'status': 'paused' if paused else progress.get('status', 'unknown'),
                    'completed': len(results), 'total': len(plan), 'remaining': len(plan) - len(results),
                    'kept': sum(row['decision'] == 'keep' for row in results),
                    'discarded': sum(row['decision'] == 'discard' for row in results),
                    'crashed': sum(row['decision'] == 'crash' for row in results),
                    'active': None if paused else progress.get('active'), 'best_candidate': progress.get('incumbent', 0),
                    'best_seconds': progress.get('incumbent_score', baseline), 'baseline_seconds': baseline,
                    'time_utc': pause.get('cancelled_at_utc') if paused else
                        max((row.get('observed_time_utc') or '' for row in self.streams['attempts/lifecycle']), default='') or None,
                    'source_progress_sha256': sha(canonical(progress).encode())}
        self.add('search/overview', sha(canonical(snapshot).encode()), **snapshot)
        self.add('experiment/status', sha(canonical(snapshot).encode()), **snapshot,
                 reason='User requested telemetry before resuming search' if paused else 'External controller observation')

    def live(self, plan):
        directory = self.root / 'evidence/telemetry'
        planned = {row['candidate']: row for row in plan}
        terminal_attempts = {row['attempt_id'] for row in self.streams['attempts/results']}
        for path in sorted((directory / 'events').glob('*.json')):
            record = read(path)
            self.add('raw/telemetry_events', record['event_id'], event=record)
            if record.get('attempt_id'):
                self.add('attempts/lifecycle', 'observer/' + record['event_id'],
                         candidate=record['candidate'], attempt_id=record['attempt_id'],
                         state=record.get('state'), phase=record.get('phase'),
                         source_time_utc=utc(record.get('source_time_unix')),
                         observed_time_utc=utc(record.get('observed_unix')),
                         event_kind=record['kind'], source_path=record.get('source_path'))
        for path in sorted((directory / 'attempts').glob('*/index.json')):
            index = read(path)
            number, aid = index['candidate'], index['attempt_id']
            identity = index['identity']
            if number not in planned or attempt_id(number, identity) != aid or any(
                    identity.get(key) != planned[number].get(key)
                    for key in ('commit', 'parent_commit', 'source_sha256', 'patch_sha256')):
                raise ValueError('Live observer attempt differs from frozen source identity')
            campaign_hash = read(self.root / 'operations.json').get('campaign_identity_sha256')
            if campaign_hash and identity.get('campaign_identity_sha256') != campaign_hash:
                raise ValueError('Live observer belongs to another frozen campaign')
            if aid in terminal_attempts:
                continue
            blobs = {}
            for name, item in index.get('metadata', {}).items():
                blob = (directory / item['blob']).resolve()
                if not blob.is_relative_to((directory / 'blobs').resolve()) or not blob.is_file():
                    raise ValueError('Live observer metadata blob is missing or outside its namespace')
                if digest(blob) != item['sha256'] or blob.stat().st_size != item['bytes']:
                    raise ValueError('Live observer metadata failed its immutable hash/size check')
                blobs[name] = blob
                self.text(blob, candidate=number, attempt=aid, logical_path=name)
            gates = {}
            for name, gate in (('screen/verification.json', 'screen'), ('screen/units.json', 'units'),
                               ('fullgate/verification.json', 'fullgate')):
                if name in blobs:
                    gates[gate] = self.verification(blobs[name], candidate=number, attempt=aid, gate=gate)
            if 'benchmark/samples.json' in blobs:
                self.samples(blobs['benchmark/samples.json'], candidate=number, attempt=aid)
            for name, blob in blobs.items():
                match = re.fullmatch(r'confirmation/(\d+)-(parent|candidate)/samples.json', name)
                if match:
                    self.samples(blob, candidate=number, attempt=aid, phase='confirmation',
                                 pair=int(match[1]), role=match[2])
            state = index.get('state', 'observed')
            self.add('attempts/results', f'{aid}/observed/{sha(canonical(index).encode())}',
                     candidate=number, attempt_id=aid, state=state, decision=state,
                     counts_as_completed=False, correctness_screen_passed=gates.get('screen'),
                     correctness_full_passed=gates.get('fullgate'), score_seconds=None, round_seconds=None,
                     reason='External observation; complete archived result is required before counting the round',
                     commit=identity['commit'], benchmark_parent=None)

    def build(self):
        plan = self.common()
        self.packages(plan)
        self.live(plan)
        self.overview(plan)
        payload = {'experiment_id': self.experiment,
                   'reporting': {'catalog_version': '1-draft', 'method': 'custom', 'features': [], 'scope': 'metadata'},
                   'streams': {key: value for key, value in sorted(self.streams.items()) if value}}
        self.client.validate_payload(payload)
        list(self.client.chunks(payload['streams']))
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'send', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', default=EXPERIMENT)
    parser.add_argument('--payload', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'build':
        payload = Export(root, args.experiment).build()
        content_hash = sha(canonical(payload).encode())
        target = args.payload or root / 'lake/outbox' / (content_hash + '.json')
        if target.exists() and read(target) != payload:
            raise ValueError('Refusing to replace an immutable outbox payload')
        write(target, payload)
        summary = {'payload': str(target), 'payload_sha256': content_hash,
                   'experiment_id': args.experiment, 'streams': len(payload['streams']),
                   'rows': sum(map(len, payload['streams'].values())),
                   'stream_rows': {key: len(value) for key, value in payload['streams'].items()}}
        write(root / 'lake/latest.json', summary)
        print(canonical(summary))
        return
    target = args.payload or Path(read(root / 'lake/latest.json')['payload'])
    payload = read(target)
    module = client_module()
    contract = module.validate_payload(payload)
    client = module.LakeClient(timeout=60)
    # Only one sender is allowed per experiment. No GPU process imports this code.
    import fcntl
    (root / 'lake').mkdir(exist_ok=True)
    with (root / 'lake/.sender.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {'experiment_id': payload['experiment_id'], 'payload_path': str(target),
                   'payload_sha256': digest(target), 'metric_contract': contract, 'verified': False}
        try:
            if args.command == 'send':
                receipt['upload'] = client.sync(payload['experiment_id'], payload['streams'])
            receipt['streams'] = client.verify(payload['experiment_id'], payload['streams'])
            receipt['verified'] = True
        except Exception as error:
            receipt['error'] = module.redact_error(str(error), client.token)
            raise RuntimeError(receipt['error']) from None
        finally:
            receipt['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
            write(root / 'lake/receipts' / (target.stem + '-' + args.command + '.json'), receipt)
        print(canonical({'verified': True, 'streams': len(receipt['streams']),
                         'rows': sum(row['expected_rows'] for row in receipt['streams'].values())}))


if __name__ == '__main__':
    main()
