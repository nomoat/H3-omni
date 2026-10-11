#!/usr/bin/env python3
"""Read-only remote observer for the explicit two-sample, 300-second v2 protocol.

Writes evidence/search-fast, fast-events.jsonl, monitoring-fast.json and
report-fast.{json,html}; never changes the v1 journal, worker, judge or GPU job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tarfile
import time

from ledger import atomic_write

PROTOCOL_ID = 'h3-cold2-300s-v2'
PROTOCOL = dict(round_seconds=300, evaluation_seconds=295, samples=2,
                screen='frozen_extended', fullgate='final_winner_only',
                keep_rule='median<parent*0.995 and every sample<parent')
LIVE = {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
REMOTE_SCRIPT = """import io,json,tarfile,sys
from pathlib import Path
root=Path('/home/cj/h3-cuda/autoresearch-100/results-fast')
paths=[root/'progress.json',root/'active.json']+sorted(root.glob('[0-9][0-9][0-9][0-9]/result.json'))+sorted(root.glob('[0-9][0-9][0-9][0-9]/archive.json'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for path in paths:
  if not path.is_file():continue
  if path.stat().st_size>4194304:raise ValueError('Compact result exceeds size limit')
  data=path.read_bytes();value=json.loads(data)
  if path.name=='result.json' and value.get('terminal') is not True:continue
  info=tarfile.TarInfo(str(path.relative_to(root)));info.size=len(data)
  archive.addfile(info,io.BytesIO(data))
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()+b'\n')


def load_plan(root, check_git=True):
    plan = json.loads((root/'plan.json').read_bytes())
    require(isinstance(plan, list) and plan, 'Frozen plan is missing or empty')
    ids = [entry['candidate'] for entry in plan]
    require(all(type(n) is int and n > 0 for n in ids) and ids == sorted(set(ids)),
            'Plan candidate IDs are not a unique ordered sequence')
    require(len({entry['source_sha256'] for entry in plan}) == len(plan), 'Duplicate candidate source hashes')
    for entry in plan:
        source = root/'sources'/f'{entry["candidate"]:04d}.cu'
        require(digest(source.read_bytes()) == entry['source_sha256'], 'Local source differs from frozen plan')
    if check_git:
        run = subprocess.run(['git', '-C', str(root/'repo'), 'for-each-ref',
                              '--format=%(refname) %(objectname)', 'refs/candidates/'],
                             capture_output=True, text=True, check=True, timeout=30)
        refs = dict(line.split() for line in run.stdout.splitlines())
        require(all(refs.get(f'refs/candidates/{entry["candidate"]:04d}') == entry['commit']
                    for entry in plan), 'Candidate Git refs differ from frozen plan')
    return {entry['candidate']:entry for entry in plan}


def validate_progress(root, progress, plan):
    require(isinstance(progress, dict), 'Progress must be an object')
    identity = progress.get('campaign_identity')
    require(isinstance(identity, dict), 'Fast progress has no campaign identity')
    require(identity.get('protocol_id') == PROTOCOL_ID and progress.get('protocol_id') == PROTOCOL_ID,
            'Fast progress uses a different protocol')
    require(progress.get('campaign_identity_sha256') == digest(encoded(identity)), 'Campaign identity hash differs')
    require(identity.get('plan_sha256') == digest((root/'plan.json').read_bytes()), 'Campaign plan hash differs')
    protocol = json.loads((root/'protocol.json').read_bytes())
    require(identity.get('prompt_sha256') == protocol['prompt_sha256'] and identity.get('limit') == len(plan),
            'Campaign fixed prompt or limit differs')
    require(all(identity.get('protocol', {}).get(k) == v for k,v in PROTOCOL.items()), 'Fast protocol parameters differ')
    local_protocol = root/'protocol-fast.json'
    if local_protocol.exists():
        declared = json.loads(local_protocol.read_bytes())
        require(declared.get('version') == 2 and declared.get('candidate_samples') == 2
                and declared.get('round_timeout_seconds') == 300 and declared.get('round_work_seconds') == 295
                and declared.get('promotion_pairs') == 0 and declared.get('seed') == protocol['seed']
                and declared.get('prompt_sha256') == protocol['prompt_sha256'],
                'Local fast protocol declaration differs')
    baseline_path = root/'fast-baseline.json'
    require(baseline_path.is_file() and identity.get('baseline_sha256') == digest(baseline_path.read_bytes()),
            'Two-sample baseline manifest hash differs')
    baseline = json.loads(baseline_path.read_bytes())
    samples = baseline.get('samples')
    require(isinstance(samples,list) and len(samples)==2 and all(positive(v) for v in samples)
            and baseline.get('score_seconds') == statistics.median(samples)
            and progress.get('baseline_samples') == samples and progress.get('baseline_score') == baseline['score_seconds'],
            'Two-sample baseline score differs')
    sources = baseline.get('provenance', [])
    require(len(sources)==2, 'Baseline must identify exactly two source measurements')
    for index, source in enumerate(sources):
        name = f'{index:02d}-parent-benchmark.json'
        require(Path(source.get('path','')).name == name, 'Unexpected baseline proof path')
        raw = (root/'evidence/fast-inputs'/name).read_bytes()
        proof = json.loads(raw)
        require(digest(raw) == source.get('sha256') and proof.get('passed') is True
                and proof.get('samples') == [samples[index]] and len(proof.get('records',[])) == 1,
                'Baseline proof hash or measured sample differs')
        measurement = proof['records'][0]
        require(measurement.get('seed') == protocol['seed'] and measurement.get('prompt_sha256') == protocol['prompt_sha256']
                and measurement.get('seconds') == samples[index], 'Baseline proof uses a different prompt/seed')
    completed = progress.get('completed', [])
    legacy = progress.get('legacy_candidates', [])
    require(isinstance(completed, list) and all(type(n) is int for n in completed)
            and completed == list(plan)[:len(completed)], 'Completed rounds must form the unique plan prefix')
    require(isinstance(legacy, list) and legacy == sorted(set(legacy))
            and all(n in completed for n in legacy), 'Legacy candidate list is inconsistent')
    require(legacy in ([], [1]), 'Only the explicitly completed v1 candidate 1 may be carried forward')
    if legacy:
        old_path = root/'evidence/search/0001/result.json'
        old_raw = old_path.read_bytes()
        old = json.loads(old_raw)
        require(old.get('terminal') is True and old.get('status') in ('keep','discard','crash')
                and all(old.get(k) == v for k,v in plan[1].items()), 'Legacy terminal identity differs')
        manifest_path = root/'fast-legacy-manifest.json'
        require(manifest_path.is_file() and identity.get('legacy_sha256') == digest(manifest_path.read_bytes()),
                'Legacy manifest hash differs')
        carried = json.loads(manifest_path.read_bytes()).get('candidates', [])
        require([entry.get('candidate') for entry in carried] == legacy, 'Legacy manifest candidate list differs')
        source = carried[0]
        require(Path(source.get('path','')).name == 'legacy-0001-result.json', 'Unexpected legacy proof path')
        proof = (root/'evidence/fast-inputs/legacy-0001-result.json').read_bytes()
        require(source.get('sha256') == digest(proof) == digest(old_raw), 'Legacy source proof differs')
    require(progress.get('active') is None or progress['active'] in plan, 'Active candidate is outside frozen plan')
    return identity


def validate_record(record, entry, progress):
    require(isinstance(record, dict) and record.get('terminal') is True and
            record.get('status') in ('keep','discard','crash'), 'Result is not a complete terminal result')
    require(all(record.get(k) == v for k,v in entry.items()), 'Terminal source/commit identity differs from plan')
    require(record.get('protocol_id') == PROTOCOL_ID and
            record.get('campaign_identity_sha256') == progress['campaign_identity_sha256'],
            'Terminal protocol or campaign identity differs')
    require((record.get('correctness') is None or type(record.get('correctness')) is bool) and record.get('fullgate') is False
            and record.get('provisional') is True, 'v2 correctness scope must be explicitly provisional')
    require(positive(record.get('started_unix')) and positive(record.get('finished_unix'))
            and record['finished_unix'] >= record['started_unix'], 'Attempt timing is invalid')
    attempts = record.get('attempts')
    require(isinstance(attempts, list) and attempts and all(isinstance(a, dict) and
            a.get('stage') in ('build','run') and a.get('outcome') in ('success','candidate_error','budget_exceeded')
            for a in attempts), 'Result has missing or infrastructure attempts')
    screen = record.get('screen', {}).get('passed')
    require(screen is None or type(screen) is bool, 'Screen result is invalid')
    require(record.get('correctness') is screen, 'Correctness differs from the recorded screen result')
    samples = record.get('samples', [])
    require(isinstance(samples, list) and len(samples) <= 2 and all(positive(v) for v in samples),
            'Fast samples must contain at most two finite positive values')
    score = record.get('score_seconds')
    if score is not None:
        require(positive(score) and len(samples) == 2 and
                math.isclose(score, statistics.median(samples), rel_tol=1e-10, abs_tol=1e-10),
                'Fast score differs from the two-sample median')
    budget = record.get('budget_exceeded', False)
    require(type(budget) is bool, 'Budget status must be boolean')
    if budget:
        require(record.get('reason') == 'budget_exceeded' and record['status'] != 'keep' and score is None,
                'Budget timeout must not be a kept/scored accuracy failure')
    parent = record.get('benchmark_parent')
    after = record.get('state_after')
    require(type(parent) is int and parent >= 0 and isinstance(after, dict)
            and set(after) == {'incumbent','incumbent_score'} and positive(after['incumbent_score'])
            and after['incumbent'] == (entry['candidate'] if record['status']=='keep' else parent),
            'Invalid durable incumbent transition')
    if record['status'] == 'keep':
        require(screen is True and len(samples) == 2 and positive(score) and not budget
                and after['incumbent_score'] == score, 'Kept candidate lacks two samples and screen PASS')
    require(isinstance(record.get('reason'), str), 'Result has no decision reason')


def install_archive(root, raw, plan):
    require(len(raw) <= 64 << 20, 'Compact sync exceeds size limit')
    local = root/'evidence/search-fast'
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        for member in archive:
            require(member.isfile() and member.size <= 4 << 20 and
                    (member.name in ('progress.json','active.json') or re.fullmatch(r'[0-9]{4}/(?:result|archive)\.json', member.name)),
                    'Unexpected compact evidence member')
            require(member.name not in payloads, 'Duplicate compact evidence member')
            payloads[member.name] = archive.extractfile(member).read()
    previous_path = local/'progress.json'
    previous = json.loads(previous_path.read_bytes()) if previous_path.exists() else None
    progress = json.loads(payloads['progress.json']) if 'progress.json' in payloads else previous
    if progress is None:
        require(not payloads, 'Results have no campaign identity')
        return None
    validate_progress(root, progress, plan)
    if previous is not None:
        require(previous['campaign_identity_sha256'] == progress['campaign_identity_sha256'],
                'Remote campaign identity changed after initial observation')
        require(progress['completed'][:len(previous['completed'])] == previous['completed'],
                'Remote completed prefix went backward')
    for name, payload in payloads.items():
        if name == 'progress.json':
            continue
        value = json.loads(payload)
        if name == 'active.json':
            require(value.get('protocol_id') == PROTOCOL_ID and value.get('candidate') in plan
                    and positive(value.get('started_unix')), 'Active observation identity differs')
            continue
        number = int(name.split('/')[0])
        require(number in plan and number not in progress.get('legacy_candidates', []),
                'Result is unknown or relabels a legacy execution')
        if name.endswith('/archive.json'):
            require(value.get('candidate') == number and value.get('status') in
                    ('pending','complete','failed','deferred_after_archive_error'), 'Archive metadata identity differs')
            continue
        validate_record(value, plan[number], progress)
        target = local/name
        require(not target.exists() or target.read_bytes() == payload, 'Immutable terminal result changed')
    # Result files first; progress is the final local publication marker.
    for name in sorted(payloads, key=lambda name: name == 'progress.json'):
        atomic_write(local/name, payloads[name])
    return progress


def replay(root, progress, plan):
    """One hash-linked v2 event per immutable result; crash-safe append/replay."""
    journal = root/'fast-events.jsonl'
    raw = journal.read_bytes() if journal.exists() else b''
    require(not raw or raw.endswith(b'\n'), 'Incomplete fast journal tail; preserve and repair explicitly')
    events, seen, previous = [], {}, '0'*64
    for index, line in enumerate(raw.splitlines(), 1):
        event = json.loads(line)
        value = {k:v for k,v in event.items() if k != 'event_hash'}
        require(event.get('seq') == index and event.get('prev_hash') == previous
                and digest(encoded(value)) == event.get('event_hash'), 'Fast journal hash chain differs')
        number = event['candidate']
        require(number not in seen and event.get('protocol_id') == PROTOCOL_ID and
                event.get('campaign_identity_sha256') == progress['campaign_identity_sha256'],
                'Fast journal has duplicate or foreign records')
        events.append(event);seen[number] = event;previous = event['event_hash']
    records = []
    baseline = progress.get('baseline_score')
    incumbent, incumbent_score = 0, baseline
    for number in progress['completed']:
        if number in progress.get('legacy_candidates', []):
            continue
        path = root/'evidence/search-fast'/f'{number:04d}/result.json'
        require(path.is_file(), 'Committed fast result is not yet synchronized')
        payload = path.read_bytes();record = json.loads(payload)
        validate_record(record, plan[number], progress)
        require(record['benchmark_parent'] == incumbent, 'Benchmark parent breaks incumbent replay')
        if positive(incumbent_score):
            if record['status'] == 'keep':
                require(record['score_seconds'] < incumbent_score*.995 and
                        all(v < incumbent_score for v in record['samples']), 'Keep rule was not satisfied')
            else:
                require(record['state_after']['incumbent_score'] == incumbent_score,
                        'Discard changed incumbent score')
        incumbent = record['state_after']['incumbent']
        incumbent_score = record['state_after']['incumbent_score']
        records.append(record)
        if number in seen:
            require(seen[number]['raw_result_sha256'] == digest(payload), 'Journal evidence hash differs')
            continue
        event = dict(seq=len(events)+1, prev_hash=previous, kind='terminal_v2', candidate=number,
                     protocol_id=PROTOCOL_ID, campaign_identity_sha256=progress['campaign_identity_sha256'],
                     raw_result_path=str(path.relative_to(root)), raw_result_sha256=digest(payload),
                     commit=record['commit'], source_sha256=record['source_sha256'],
                     status=record['status'], budget_exceeded=record.get('budget_exceeded', False),
                     score_seconds=record.get('score_seconds'))
        event['event_hash'] = digest(encoded(event))
        with journal.open('ab') as stream:
            stream.write(encoded(event)+b'\n');stream.flush();os.fsync(stream.fileno())
        events.append(event);seen[number] = event;previous = event['event_hash']
    require(set(seen) == {r['candidate'] for r in records}, 'Journal has results absent from the completed prefix')
    if records:
        require(progress['incumbent'] == incumbent and progress['incumbent_score'] == incumbent_score,
                'Progress incumbent differs from durable replay')
    return records


def summarize(root, progress, records, scheduler, total):
    progress = progress or {}
    legacy = progress.get('legacy_candidates', [])
    counts = Counter(record['status'] for record in records)
    kept = [record for record in records if record['status']=='keep']
    best = min(kept, key=lambda r:r['score_seconds']) if kept else None
    phase = progress.get('phase')
    active_path = root/'evidence/search-fast/active.json'
    if active_path.exists():
        active = json.loads(active_path.read_bytes())
        if progress.get('status') == 'running' and active.get('candidate') == progress.get('active'):
            phase = active.get('phase',phase)
    archives = {}
    for path in (root/'evidence/search-fast').glob('[0-9][0-9][0-9][0-9]/archive.json'):
        archives[int(path.parent.name)] = json.loads(path.read_bytes())['status']
    value = dict(checked_utc=datetime.now(timezone.utc).isoformat(), protocol_id=PROTOCOL_ID,
                 correctness_scope='frozen_extended_screen; full 6+1 gate deferred to final winner',
                 score='median of exactly two cold process E2E seconds', round_budget_seconds=300,
                 **scheduler, worker_status=progress.get('status','not_yet_observed'),
                 completed=len(legacy)+len(records), total=total, legacy_completed=len(legacy),
                 v2_completed=len(records), remaining=total-len(legacy)-len(records),
                 active=progress.get('active'), phase=phase, counts=dict(counts), archive_status=archives,
                 budget_exceeded=sum(r.get('budget_exceeded',False) for r in records),
                 provisional_best_candidate=best['candidate'] if best else None,
                 provisional_best_seconds=best['score_seconds'] if best else None,
                 incumbent=progress.get('incumbent'), incumbent_score_seconds=progress.get('incumbent_score'),
                 baseline_seconds=progress.get('baseline_score'), records=records)
    value['round_budget_overruns'] = [r['candidate'] for r in records if r.get('round_seconds',r['finished_unix']-r['started_unix']) > 300]
    if progress.get('error'):
        value['worker_error'] = progress['error']
    write_json(root/'report-fast.json', value)
    write_json(root/'monitoring-fast.json', {k:v for k,v in value.items() if k!='records'})
    escape = lambda value: html.escape(str(value))
    rows = []
    for r in records:
        screen = r.get('screen',{}).get('passed')
        accuracy = 'PASS' if screen is True else ('FAIL' if screen is False else 'not finished')
        state = 'over budget' if r.get('budget_exceeded') else r['status']
        rows.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in (
            r['candidate'],state,accuracy,r.get('score_seconds'),r.get('samples',[]),
            round(r['finished_unix']-r['started_unix'],3),r['benchmark_parent'],
            r['commit'][:12],r['parent_commit'][:12],r['reason']))+
            f'<td><a href="evidence/search-fast/{r["candidate"]:04d}/result.json">raw result</a></td></tr>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H3 fast search</title>
<style>body{{font:15px system-ui;margin:24px;color:#17212f}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}p{{max-width:1100px}}code{{background:#eee}}</style>
<h1>H3 · two-sample / 300 s protocol</h1><p>Completed <b>{value['completed']} / {total}</b>; new in v2: {len(records)}; kept from the old protocol: {len(legacy)}.</p>
<p>Slurm {escape(scheduler.get('scheduler_state'))}; worker {escape(value['worker_status'])}; running {escape(value['active'])} / {escape(value['phase'])}.</p>
<p>Provisional best candidate: {escape(value['provisional_best_candidate'])}; cold-start E2E: {escape(value['provisional_best_seconds'])} s. Each score is the median of two samples in separate processes.</p>
<p>In this round, keep only requires a PASS of the frozen extended screen; the full 6+1 regression is for the final winner. Every v2 keep is provisional. Over budget is shown separately and is not an accuracy FAIL. Old-protocol results are not converted to two-sample results.</p>
<p>Last read: {escape(value['checked_utc'])}. Source parent is the Git parent of the source; benchmark parent is the measured incumbent.</p>
<table><thead><tr>{''.join('<th>'+name+'</th>' for name in ('candidate','verdict','screen','E2E s','raw samples','round s','benchmark parent','source commit','source parent','reason','evidence'))}</tr></thead><tbody>{''.join(rows)}</tbody></table></html>'''
    atomic_write(root/'report-fast.html', page.encode())
    return {k:v for k,v in value.items() if k!='records'}


def ssh_args():
    return ['ssh','-i',str(Path.home()/'.ssh/msb-build'),'-p','2222','-o','BatchMode=yes',
            '-o','ConnectTimeout=15','cj@129.212.240.50']


def fetch_remote():
    run = subprocess.run(ssh_args()+['python3 -'], input=REMOTE_SCRIPT.encode(),
                         capture_output=True, timeout=90, check=True)
    return run.stdout


def scheduler_status(job):
    require(type(job) is int and job > 0, 'Invalid Slurm job')
    run = subprocess.run(ssh_args()+[f'sacct -X -j {job} --noheader --parsable2 --format=JobIDRaw,State,Elapsed,ExitCode'],
                         capture_output=True, text=True, check=True, timeout=45)
    rows = [line.split('|') for line in run.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == str(job)]
    require(len(rows)==1 and len(rows[0])>=4, 'Scheduler job identity is missing or ambiguous')
    row = rows[0]
    return dict(job=job,scheduler_state=row[1],scheduler_elapsed=row[2],scheduler_exit=row[3])


def observe(root, job):
    plan = load_plan(root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        remote = pool.submit(fetch_remote)
        slurm = pool.submit(scheduler_status, job)
        data, scheduler = remote.result(), slurm.result()
    progress = install_archive(root, data, plan)
    records = replay(root, progress, plan) if progress else []
    return summarize(root, progress, records, scheduler, len(plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--job', required=True, type=int)
    parser.add_argument('--interval', default=60, type=int)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.job <= 0 or args.interval < 10:
        parser.error('job must be positive; interval must be at least 10 seconds')
    root = args.root.resolve()
    with (root/'.monitor-fast.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        while True:
            try:
                state = observe(root, args.job)
                failures = 0
                print(json.dumps(state, ensure_ascii=False), flush=True)
                if args.once:
                    return 0
                if state['worker_status'] == 'complete':
                    return 0 if state['completed'] == state['total'] else 2
                if state['worker_status'] == 'infrastructure_error' or state['scheduler_state'].split()[0] not in LIVE:
                    return 2
            except Exception as error:
                failures += 1
                value = dict(job=args.job, checked_utc=datetime.now(timezone.utc).isoformat(),
                             observer_error=str(error), consecutive_failures=failures)
                write_json(root/'monitor-fast-error.json', value)
                print(json.dumps(value, ensure_ascii=False), flush=True)
                if args.once or failures >= 5:
                    return 2
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
