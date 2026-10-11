"""Collect completed packages and publish outside the GPU process.

The node worker creates immutable archives before publishing completed rounds.
This controller downloads them from shared storage, verifies every member,
uploads private content-addressed objects, then builds a durable lake outbox.
No inference process imports this module. Network failures retain local files.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from artifact_storage import Gcloud, upload_artifact
from lake_export import Export, attempt_id, canonical, digest, read, sha, verified_cloud, write

HERE = Path(__file__).resolve().parent
REMOTE = '/home/cj/h3-cuda/autoresearch-100/results'


def ssh():
    return ['ssh', '-i', str(Path.home() / '.ssh/msb-build'), '-p', '2222',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cj@129.212.240.50', 'python3 -']


def sync_telemetry(root):
    local = root / 'evidence/telemetry'
    known = [str(path.relative_to(local)) for pattern in ('events/*.json', 'blobs/*.json')
             for path in local.glob(pattern)]
    script = '''import io,json,sys,tarfile
from pathlib import Path
root=Path('/home/cj/h3-cuda/autoresearch-100/telemetry')
known=set(KNOWN)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 for pattern in ('progress.json','observer.json','events/*.json','blobs/*.json','attempts/*/index.json','archives/*.json','closed/*.json'):
  for path in sorted(root.glob(pattern)):
   relative=str(path.relative_to(root))
   if relative in known:continue
   if path.is_symlink() or not path.is_file():raise ValueError('Unsafe telemetry file')
   data=path.read_bytes()
   if len(data)>2<<20:raise ValueError('Telemetry exceeds small-file limit')
   json.loads(data)
   entry=tarfile.TarInfo(relative);entry.size=len(data);out.addfile(entry,io.BytesIO(data))
'''.replace('KNOWN', repr(known))
    result = subprocess.run(ssh(), input=script.encode(), capture_output=True, timeout=120, check=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or '..' in parts or member.name.startswith('/') or member.size > 2 << 20:
                raise ValueError('Unsafe telemetry transfer member')
            accepted = (member.name in ('progress.json', 'observer.json') or
                        len(parts) == 2 and parts[0] in ('events', 'blobs', 'archives', 'closed') and parts[1].endswith('.json') or
                        len(parts) == 3 and parts[0] == 'attempts' and parts[2] == 'index.json')
            if not accepted:
                raise ValueError('Unexpected telemetry transfer member')
            data = archive.extractfile(member).read()
            json.loads(data)
            if parts[0] == 'blobs' and sha(data) != Path(parts[1]).stem:
                raise ValueError('Telemetry blob hash differs from its immutable name')
            target = local / member.name
            if target.exists() and parts[0] in ('events', 'blobs') and target.read_bytes() != data:
                raise ValueError('Immutable telemetry event/blob changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.pending')
            with temporary.open('wb') as output:
                output.write(data); output.flush(); os.fsync(output.fileno())
            temporary.replace(target)


def download(number, target, remote_path=None):
    if type(number) is not int or not 1 <= number <= 100:
        raise ValueError('Invalid candidate')
    remote_path = remote_path or (REMOTE + '/' + format(number, '04d') + '/evidence.tar')
    if not re.fullmatch(r'/home/cj/h3-cuda/autoresearch-100/(results/[0-9]{4}|telemetry-packages/[0-9]{4}-[0-9a-f]{16})/evidence\.tar', remote_path):
        raise ValueError('Archive download path is outside the campaign namespace')
    # Read a published immutable file. Never package live node files here.
    script = f'''import hashlib,json,sys
from pathlib import Path
p=Path({remote_path!r})
if not p.is_file():sys.exit(75)
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
print(json.dumps(dict(size_bytes=p.stat().st_size,sha256=h.hexdigest())),flush=True)
with p.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):sys.stdout.buffer.write(b)
'''
    temporary = target.with_suffix('.pending')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.download.log').open('wb') as errors:
        process = subprocess.Popen(ssh(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        process.stdin.write(script.encode())
        process.stdin.close()
        metadata_line = process.stdout.readline(4096)
        if not metadata_line:
            code = process.wait(timeout=60)
            if code == 75:
                return None
            raise RuntimeError('Archive download failed before metadata')
        metadata = json.loads(metadata_line)
        value, size = hashlib.sha256(), 0
        with temporary.open('wb') as output:
            for part in iter(lambda: process.stdout.read(8 << 20), b''):
                output.write(part)
                value.update(part)
                size += len(part)
            output.flush()
            os.fsync(output.fileno())
        code = process.wait(timeout=60)
    if code or size != metadata['size_bytes'] or value.hexdigest() != metadata['sha256']:
        raise ValueError('Archive transfer checksum or size mismatch')
    temporary.replace(target)
    write(target.with_suffix('.transfer.json'), metadata)
    return metadata


def _inspect_into_empty(path, detail):
    members, names, compact_bytes = [], set(), 0
    detail.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'r:') as archive:
        for entry in archive:
            if entry.isdir():
                continue
            name = str(PurePosixPath(entry.name))
            if not entry.isfile() or name.startswith('/') or '..' in PurePosixPath(name).parts or name in names:
                raise ValueError('Unsafe or duplicate archive member')
            names.add(name)
            value = hashlib.sha256()
            text_file = Path(name).suffix in ('.json', '.log', '.txt', '.jsonl')
            output_path = detail / name
            if text_file:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            output = output_path.with_suffix(output_path.suffix + '.pending').open('wb') if text_file else None
            size = 0
            with archive.extractfile(entry) as source:
                for part in iter(lambda: source.read(1 << 20), b''):
                    value.update(part)
                    size += len(part)
                    if output:
                        output.write(part)
            if output:
                output.flush()
                os.fsync(output.fileno())
                output.close()
                Path(output.name).replace(output_path)
                compact_bytes += size
            if size != entry.size:
                raise ValueError('Truncated archive member')
            row = dict(path=name, size_bytes=size, sha256=value.hexdigest())
            if text_file:
                row['detail_path'] = str(output_path)
            members.append(row)
    return dict(members=members, file_count=len(members),
                file_payload_bytes=sum(row['size_bytes'] for row in members),
                compact_file_count=sum('detail_path' in row for row in members), compact_bytes=compact_bytes)


def inspect_archive(path, detail):
    detail.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.extract-', dir=detail.parent))
    try:
        result = _inspect_into_empty(path, temporary)
        for member in result['members']:
            if 'detail_path' in member:
                member['detail_path'] = str(detail / member['path'])
        if detail.exists() or detail.is_symlink():
            existing = [detail, *detail.rglob('*')]
            if any(p.is_symlink() for p in existing) or not detail.is_dir():
                raise ValueError('Retained extraction contains a symbolic link or non-directory root')
            wanted = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
            found = {str(p.relative_to(detail)): digest(p) for p in detail.rglob('*') if p.is_file()}
            if wanted != found:
                raise ValueError('Existing extraction differs from the complete immutable archive')
        else:
            temporary.replace(detail)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def collect(root, upload=True):
    inventory_path = root / 'evidence/packages/manifest.json'
    cloud_path = root / 'evidence/packages/cloud-artifacts.json'
    inventory = read(inventory_path, {'schema_version': 1, 'packages': []})
    cloud = read(cloud_path, {'schema_version': 1, 'artifacts': []})
    known_attempts = set()
    for package in inventory['packages']:
        detail = Path(package.get('detail_root', root / 'evidence/detail' / f"{package['candidate']:04d}"))
        record = read(detail / 'result.json') or read(detail / 'inflight.json')
        known_attempts.add(attempt_id(package['candidate'], record))
    for result_path in sorted((root / 'evidence/search').glob('[0-9][0-9][0-9][0-9]/result.json')):
        record = read(result_path)
        number = record['candidate']
        aid = attempt_id(number, record)
        if aid in known_attempts:
            continue
        target = root / 'evidence/packages' / (aid + '-evidence.tar')
        transferred = read(target.with_suffix('.transfer.json')) if target.is_file() else download(number, target)
        if transferred is None:
            continue
        if digest(target) != transferred['sha256'] or target.stat().st_size != transferred['size_bytes']:
            raise ValueError('Retained archive differs from transfer receipt')
        detail = root / 'evidence/detail' / aid
        package = dict(candidate=number, attempt_id=aid, execution_state='completed',
                       local_path=str(target), shared_path=f'{REMOTE}/{number:04d}/evidence.tar',
                       detail_root=str(detail), **transferred, **inspect_archive(target, detail))
        if read(detail / 'result.json') != record:
            raise ValueError('Archive terminal result differs from synchronized ledger evidence')
        inventory['packages'].append(package)
        write(inventory_path, inventory)
        known_attempts.add(aid)
    for request_path in sorted((root / 'evidence/telemetry/closed').glob('*.json')):
        request = read(request_path)
        number, aid = request['candidate'], request['attempt_id']
        if aid in known_attempts:
            continue
        if request.get('producer_stopped') is not True or request.get('counts_as_completed') is not False:
            raise ValueError('Interrupted archive has no stopped-producer confirmation')
        target = root / 'evidence/packages' / (aid + '-interrupted-evidence.tar')
        transferred = read(target.with_suffix('.transfer.json')) if target.is_file() else download(number, target, request['source_path'])
        if transferred is None:
            continue
        if any(transferred[key] != request[key] for key in ('sha256', 'size_bytes')) or digest(target) != request['sha256']:
            raise ValueError('Interrupted archive differs from its node packaging receipt')
        detail = root / 'evidence/detail' / aid
        package = dict(candidate=number, attempt_id=aid, execution_state=request['execution_state'],
                       reason=request['reason'], local_path=str(target), shared_path=request['source_path'],
                       detail_root=str(detail), **transferred, **inspect_archive(target, detail))
        record = read(detail / 'inflight.json')
        if attempt_id(number, record) != aid or (detail / 'result.json').exists():
            raise ValueError('Interrupted archive does not match its nonterminal execution')
        inventory['packages'].append(package)
        write(inventory_path, inventory)
    if upload:
        for package in inventory['packages']:
            found = next((item for item in cloud['artifacts'] if item['sha256'] == package['sha256']), None)
            if found:
                verified_cloud(found, package)
                continue
            receipt_dir = root / 'lake/uploads' / (package['sha256'] + '-' + str(time.time_ns()))
            item = {key: package[key] for key in ('candidate', 'local_path', 'sha256', 'size_bytes')}
            item['role'] = 'candidate_execution'
            receipt = upload_artifact(item, Gcloud(receipt_dir))
            cloud['artifacts'].append(receipt)
            cloud['status'] = 'PASS'
            write(cloud_path, cloud)
    return inventory


def persist_payload(root):
    payload = Export(root).build()
    identity = sha(canonical(payload).encode())
    target = root / 'lake/outbox' / (identity + '.json')
    if not target.exists():
        write(target, payload)
    elif read(target) != payload:
        raise ValueError('Existing immutable outbox payload differs from its content hash')
    write(root / 'lake/latest.json', dict(payload=str(target), payload_sha256=identity,
          experiment_id=payload['experiment_id'], streams=len(payload['streams']),
          rows=sum(map(len, payload['streams'].values())),
          stream_rows={key: len(value) for key, value in payload['streams'].items()}))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--no-sync', action='store_true')
    parser.add_argument('--no-upload', action='store_true')
    parser.add_argument('--send', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error('Polling interval must be at least 30 seconds')
    root = args.root.resolve()
    (root / 'lake').mkdir(exist_ok=True)
    with (root / 'lake/.collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        while True:
            try:
                if not args.no_sync:
                    subprocess.run([sys.executable, str(HERE / 'sync_results.py'), '--root', str(root)],
                                   check=True, capture_output=True, timeout=300)
                    sync_telemetry(root)
                collect(root, upload=not args.no_upload)
                target = persist_payload(root)
                if args.send:
                    subprocess.run([sys.executable, str(HERE / 'lake_export.py'), 'send', '--root', str(root),
                                    '--payload', str(target)], check=True, timeout=1800)
                write(root / 'lake/collector-status.json', dict(state='verified' if args.send else 'packaged',
                      payload=str(target), observed_unix=time.time(), pid=os.getpid()))
                failures = 0
            except Exception as error:
                failures += 1
                write(root / 'lake/collector-status.json', dict(state='error', error=str(error),
                      consecutive_failures=failures, observed_unix=time.time(), pid=os.getpid()))
                if not args.watch or failures >= 3:
                    raise
            if not args.watch:
                return
            progress = read(root / 'evidence/search/progress.json', {})
            if progress.get('status') in ('complete', 'infrastructure_error') or read(root / 'pause.json', {}).get('state') == 'paused':
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
