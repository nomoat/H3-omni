"""Package stopped, interrupted executions outside the frozen GPU worker.

Call only after the entire producer process group has stopped. Normal terminal
archives are already produced by the worker. No HTTP or model execution occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from telemetry_observer import atomic_write, canonical, attempt_id


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            value.update(part)
    return value.hexdigest()


def package_stopped(stage, outbox, destination):
    stage, outbox, destination = map(Path, (stage, outbox, destination))
    results = (stage / 'results').resolve()
    for path in sorted((outbox / 'archives').glob('*.json')):
        request = json.loads(path.read_text())
        if request.get('kind') != 'interrupted_directory' or request.get('safe_to_collect') is not True:
            continue
        source = Path(request['source_path'])
        if source.is_symlink() or not source.resolve().is_relative_to(results):
            raise ValueError('Interrupted directory is outside producer results')
        record = json.loads((source / 'inflight.json').read_text())
        aid = attempt_id(request['candidate'], record)
        if aid != request['attempt_id'] or record['started_unix'] != request['expected_started_unix']:
            raise ValueError('A replacement execution owns this directory')
        index = json.loads((outbox / 'attempts' / aid / 'index.json').read_text())
        if (index.get('candidate') != request['candidate'] or index.get('attempt_id') != aid or
                index.get('state') not in ('interrupted', 'infrastructure_error') or
                any(record.get(key) != index.get('identity', {}).get(key) for key in
                    ('started_unix', 'commit', 'parent_commit', 'source_sha256', 'patch_sha256', 'campaign_identity_sha256'))):
            raise ValueError('Stopped archive source differs from observer identity or state')
        if (source / 'result.json').exists():
            # A terminal write awaiting archive publication is replayed by the
            # frozen worker. Do not relabel that execution as interrupted.
            continue
        target = destination / aid / 'evidence.tar'
        receipt_path = outbox / 'closed' / (aid + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if target.is_file() and target.stat().st_size == receipt['size_bytes'] and digest(target) == receipt['sha256']:
                continue
            raise ValueError('Previously packaged interrupted archive is missing or changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tar.pending')
        subprocess.run(['tar', '-cf', str(temporary), '-C', str(source), '.'], check=True, timeout=600)
        if json.loads((source / 'inflight.json').read_text()) != record:
            raise ValueError('Interrupted source identity changed while packaging')
        content_hash = digest(temporary)
        temporary.replace(target)
        receipt = dict(schema_version=1, candidate=request['candidate'], attempt_id=aid,
                       execution_state=index['state'], source_path=str(target), source_directory=str(source),
                       sha256=content_hash, size_bytes=target.stat().st_size,
                       counts_as_completed=False, producer_stopped=True,
                       reason='Execution stopped before a terminal result; full recorded evidence retained')
        atomic_write(receipt_path, canonical(receipt) + b'\n', immutable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--outbox', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--producer-stopped', action='store_true')
    args = parser.parse_args()
    if not args.producer_stopped:
        parser.error('The producer process group must have stopped before packaging')
    package_stopped(args.stage, args.outbox, args.destination)


if __name__ == '__main__':
    main()
