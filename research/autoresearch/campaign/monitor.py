"""Pull an active campaign's durable records and refresh its local report.

This observer never submits/cancels GPU work or changes a candidate or judge.
It exits on completion or a job/infrastructure failure, preserving the last
observed status for a subsequent interactive review.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def observe(root, job):
    checked = datetime.datetime.now(datetime.timezone.utc).isoformat()
    completed = subprocess.run(
        [sys.executable, str(HERE / 'sync_results.py'), '--root', str(root)],
        capture_output=True, text=True, timeout=240)
    if completed.returncode:
        raise RuntimeError('Result sync failed: ' + completed.stderr[-4000:])
    report = HERE / 'report.py'
    if report.is_file():
        rendered = subprocess.run(
            [sys.executable, str(report), '--root', str(root)],
            capture_output=True, text=True, timeout=120)
        if rendered.returncode:
            raise RuntimeError('Report refresh failed: ' + rendered.stderr[-4000:])
    status = subprocess.run(
        ['ssh', '-i', str(Path.home() / '.ssh/msb-build'), '-p', '2222',
         '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cj@129.212.240.50',
         f'sacct -X -j {job} --noheader --parsable2 --format=JobIDRaw,State,Elapsed,ExitCode'],
        capture_output=True, text=True, timeout=60, check=True)
    rows = [line.split('|') for line in status.stdout.splitlines() if line.strip()]
    matching = [row for row in rows if row[0] == str(job)]
    if len(matching) != 1 or len(matching[0]) < 4:
        raise RuntimeError('Scheduler did not return the requested job identity')
    row = matching[0]
    progress_path = root / 'evidence/search/progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    terminal = sorted((root / 'evidence/search').glob('[0-9][0-9][0-9][0-9]/result.json'))
    finished = [json.loads(path.read_text()) for path in terminal]
    numbers = [record['candidate'] for record in finished if record.get('terminal') is True]
    state = dict(checked_utc=checked, observer_pid=os.getpid(), job=job,
                 scheduler_state=row[1], scheduler_elapsed=row[2], scheduler_exit=row[3],
                 worker_status=progress.get('status', 'not_yet_observed'),
                 completed=len(numbers), completed_candidates=numbers,
                 active=progress.get('active'), phase=progress.get('phase'),
                 incumbent=progress.get('incumbent'),
                 incumbent_score_seconds=progress.get('incumbent_score'))
    if progress.get('error'):
        state['worker_error'] = progress['error']
    write_json(root / 'monitoring.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.job <= 0 or args.interval < 30:
        parser.error('job must be positive; interval must be at least 30 seconds')
    root = args.root.resolve()
    if not (root / 'plan.json').is_file():
        parser.error('root must contain the frozen campaign plan')
    with (root / '.monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        failures = 0
        while True:
            try:
                state = observe(root, args.job)
                failures = 0
                signature = tuple(state.get(key) for key in
                    ('scheduler_state', 'worker_status', 'completed', 'active', 'phase', 'incumbent'))
                if signature != previous or args.once:
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    previous = signature
                if args.once:
                    return 0
                if state['worker_status'] == 'complete':
                    return 0 if state['completed'] == len(json.loads((root / 'plan.json').read_text())) else 2
                if state['worker_status'] == 'infrastructure_error':
                    return 2
                if state['scheduler_state'].split()[0] not in (
                    'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'):
                    return 2
            except Exception as error:
                failures += 1
                value = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             observer_error=str(error), consecutive_failures=failures,
                             job=args.job, observer_pid=os.getpid())
                write_json(root / 'monitor-error.json', value)
                print(json.dumps(value, ensure_ascii=False), flush=True)
                if args.once or failures >= 5:
                    return 2
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
