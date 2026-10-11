"""Protocol v2: frozen screen, two cold samples, 300-second candidate budget.

Keep decisions are provisional until the final winner passes the frozen promote
judge outside this search. The v1 worker, runtime and judge are not changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

PROTOCOL_ID = 'h3-cold2-300s-v2'
PROTOCOL = dict(round_seconds=300, evaluation_seconds=295, samples=2,
                screen='frozen_extended', fullgate='final_winner_only',
                keep_rule='median<parent*0.995 and every sample<parent')
ACCEPTED = Path('/var/tmp/h3-cuda-14274')
WEIGHTS = Path('/var/tmp/h3-cuda-14228/local-model/FL2VA')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def kill_group(process):
    # Kill descendants even if their leader has already exited after SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=1)


def supervise(command, seconds, **kwargs):
    """One session/process group contains build, judge, CLI and media children."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return dict(returncode=process.wait(timeout=max(0.01, seconds)), budget_exceeded=False)
    except subprocess.TimeoutExpired:
        kill_group(process)
        return dict(returncode=process.returncode, budget_exceeded=True)
    except BaseException:
        kill_group(process)
        raise


def decision(samples, parent_score):
    require(len(samples) == 2 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
            'A v2 score requires exactly two finite positive samples')
    score = statistics.median(samples)
    return score, score < parent_score * .995 and all(value < parent_score for value in samples)


def load_frozen(stage):
    directory = stage / 'seed/research/autoresearch/campaign'
    sys.path.insert(0, str(directory))
    runtime = importlib.import_module('runtime')
    frozen = importlib.import_module('worker')
    require(Path(runtime.__file__).resolve() == (directory / 'runtime.py').resolve(), 'Imported runtime is not the frozen runtime')
    require(Path(frozen.__file__).resolve() == (directory / 'worker.py').resolve(), 'Imported classifier is not the frozen worker')
    return directory, runtime, frozen


def validate_baseline(path, seed, prompt_hash):
    baseline = read_json(path)
    require(len(baseline.get('samples', [])) == len(baseline.get('provenance', [])) == 2,
            'Baseline needs two recorded parent samples and two provenance files')
    paths = set()
    for value, source in zip(baseline['samples'], baseline['provenance']):
        proof = Path(source['path']).resolve()
        require(proof not in paths and digest(proof) == source['sha256'], 'Baseline provenance changed or repeated')
        paths.add(proof)
        report = read_json(proof)
        require(report.get('passed') is True and report.get('samples') == [value] and len(report.get('records', [])) == 1,
                'Baseline source is not one complete parent measurement')
        record = report['records'][0]
        require(record.get('seconds') == value and record.get('seed') == 7 and record.get('prompt_sha256') == prompt_hash,
                'Baseline request does not match the fixed prompt/seed')
        require(Path(record['command'][0]).resolve() == (seed / 'build-cuda/h3').resolve(), 'Baseline source is not the accepted seed executable')
    score, _ = decision(baseline['samples'], float('inf'))
    return baseline, score


def validate_legacy(path, plan):
    legacy = read_json(path)
    rows = legacy.get('candidates', [])
    require([row['candidate'] for row in rows] == [1], 'Only historical v1 candidate 1 may carry over')
    row = rows[0]
    require(digest(row['path']) == row['sha256'], 'Legacy terminal bytes changed')
    result = read_json(row['path'])
    require(all(result.get(key) == value for key, value in plan[0].items()), 'Legacy candidate differs from frozen plan')
    require(result.get('terminal') is True and result.get('status') == 'discard' and
            result.get('state_after', {}).get('incumbent') == 0, 'Legacy carryover must be the discarded first candidate')
    return legacy


def evaluate_round(payload):
    stage, folder = Path(payload['stage']), Path(payload['folder'])
    directory, runtime, frozen = load_frozen(stage)
    entry = payload['entry']; number = entry['candidate']
    root = stage / 'candidates' / f'{number:04d}'
    record = dict(attempts=[], correctness=None, fullgate=False, provisional=True, score_seconds=None,
                  phase='build', status='running')

    def persist():
        write_json(folder / 'evaluation.json', record)
        write_json(Path(payload['shared']) / 'active.json', dict(candidate=number, protocol_id=PROTOCOL_ID,
                   phase=record['phase'], started_unix=payload['started_unix']))
        print(json.dumps(dict(event='candidate_phase', candidate=number, phase=record['phase'])), flush=True)

    try:
        persist()
        seed = stage / 'seed'
        if not root.exists():
            shutil.copytree(seed, root, ignore=shutil.ignore_patterns('build-cuda', 'research', '__pycache__'))
        require(all((root / name).is_file() and digest(root / name) == wanted
                    for name, wanted in payload['seed_hashes'].items()
                    if not name.startswith('research/') and name != 'cuda/h3_gpu_cuda.cu'),
                'Candidate changed code outside the frozen CUDA surface')
        source = stage / 'sources' / f'{number:04d}.cu'
        require(digest(source) == entry['source_sha256'], 'Candidate source hash differs')
        shutil.copy2(source, root / 'cuda/h3_gpu_cuda.cu')
        try:
            built = runtime.build(root, ACCEPTED, folder / 'build')
        except subprocess.CalledProcessError:
            require(not any(marker in frozen.log_tail(folder / 'build/build.log').lower() for marker in frozen.INFRA_MARKERS),
                    'Build infrastructure failure')
            record['attempts'].append(dict(stage='build', outcome='candidate_error'))
            record.update(status='crash', reason='candidate compilation failed')
        else:
            require(built['reused_objects'] == payload['immutable_objects'], 'Accepted immutable objects changed')
            require(all(digest(root / 'build-cuda' / name) == wanted for name, wanted in payload['accepted_hashes'].items()
                        if name != 'h3_gpu_cuda.o'), 'A reused CLI/model object or tokenizer changed')
            record.update(build=built)
            record['attempts'].append(dict(stage='build', outcome='success'))
            record['phase'] = 'correctness_screen'; persist()
            output = folder / 'screen'; started = time.monotonic()
            command = [sys.executable, str(directory / 'verify.py'), 'screen', '--oracle', payload['oracle'],
                       '--oracle-sha256', payload['oracle_sha256'], '--candidate-root', str(root), '--weights', str(WEIGHTS),
                       '--diffusers', str(ACCEPTED / 'diffusers'), '--output', str(output)]
            with (folder / 'screen.log').open('w') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if not (output / 'verification.json').is_file():
                if process.returncode in (-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGFPE, -signal.SIGILL):
                    raise frozen.CandidateFailure('Candidate screen process terminated by signal')
                raise RuntimeError('Screen produced no complete verification report')
            screen = frozen.classify_verification(read_json(output / 'verification.json'), output)
            require(not (process.returncode and screen['passed']), 'Screen process failed after claiming PASS')
            record['screen'] = dict(passed=screen['passed'], seconds=time.monotonic() - started, report=str(output / 'verification.json'))
            record['correctness'] = screen['passed']
            record['attempts'].append(dict(stage='run', outcome='success' if screen['passed'] else 'candidate_error'))
            if not screen['passed']:
                record.update(status='discard', reason='fixed numerical screen failed', error=screen.get('error'))
            else:
                record['phase'] = 'cold_benchmark'; persist()
                report = frozen.benchmark_checked(root, WEIGHTS, stage / 'prompt.txt', folder / 'benchmark', payload['prompt_sha256'], samples=2)
                score, keep = decision(report['samples'], payload['parent_score'])
                record.update(status='keep' if keep else 'discard', score_seconds=score, samples=report['samples'],
                              reason='provisional cold E2E improvement' if keep else 'two-sample improvement guard not met')
    except frozen.CandidateFailure as error:
        record['attempts'].append(dict(stage='run', outcome='candidate_error', phase=record['phase']))
        record.update(status='crash', reason=str(error), error=str(error), score_seconds=None)
    except BaseException as error:
        record.update(status='infrastructure_error', error=str(error), traceback=traceback.format_exc())
    persist()
    return 0 if record['status'] != 'infrastructure_error' else 2


def check_terminal(result, entry, campaign_hash):
    require(all(result.get(key) == value for key, value in entry.items()), 'Terminal candidate identity changed')
    require(result.get('terminal') is True and result.get('campaign_identity_sha256') == campaign_hash and
            result.get('protocol_id') == PROTOCOL_ID and result.get('status') in ('keep', 'discard', 'crash'), 'Invalid v2 terminal')
    require(result.get('fullgate') is False and result.get('provisional') is True, 'Search results cannot claim final acceptance')
    if result.get('budget_exceeded'):
        require(result['status'] == 'discard' and result.get('reason') == 'budget_exceeded' and result.get('score_seconds') is None,
                'Budget discard must not carry an accepted score')
    if result['status'] == 'keep':
        score, keep = decision(result['samples'], result['parent_score_seconds'])
        require(keep and score == result['score_seconds'] and result.get('screen', {}).get('passed') is True and
                not result.get('budget_exceeded'), 'Provisional keep lacks valid screen/timing evidence')
    expected = dict(incumbent=entry['candidate'] if result['status'] == 'keep' else result['benchmark_parent'],
                    incumbent_score=result['score_seconds'] if result['status'] == 'keep' else result['parent_score_seconds'])
    require(result.get('state_after') == expected, 'Terminal state transition differs')


class Archives:
    """One background archive at a time; at most 99 jobs in this finite campaign."""
    def __init__(self, shared):
        self.shared = shared
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = []
        self.failed = False

    def submit(self, number, folder):
        require(len(self.futures) < 100, 'Archive queue bound exceeded')
        self.futures.append(self.pool.submit(self.archive, number, folder))

    def archive(self, number, folder):
        target = self.shared / f'{number:04d}'
        state = dict(candidate=number, local_raw=str(folder), status='pending')
        try:
            if self.failed:
                state['status'] = 'deferred_after_archive_error'
                return
            temporary = target / 'evidence.tar.pending'
            outcome = supervise(['tar', '-cf', str(temporary), '-C', str(folder), '.'], 120)
            require(not outcome['budget_exceeded'] and outcome['returncode'] == 0, 'Archive failed or exceeded 120 seconds')
            temporary.replace(target / 'evidence.tar')
            state.update(status='complete', bytes=(target / 'evidence.tar').stat().st_size)
        except Exception as error:
            self.failed = True
            state.update(status='failed', error=str(error))
        finally:
            write_json(target / 'archive.json', state)

    def finish(self):
        self.pool.shutdown(wait=True)
        for future in self.futures:
            future.result()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '_round':
        return evaluate_round(read_json(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage', 'shared', 'oracle', 'baseline-json', 'legacy-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--oracle-sha256', required=True)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    stage, shared = args.stage.resolve(), args.shared.resolve()
    directory, runtime, frozen = load_frozen(stage)
    seed = stage / 'seed'; results = stage / 'results-fast'
    results.mkdir(exist_ok=True); shared.mkdir(parents=True, exist_ok=True)
    plan = read_json(stage / 'plan.json')[:args.limit]
    require(2 <= args.limit <= 100 and len(plan) == args.limit and [row['candidate'] for row in plan] == list(range(1, args.limit + 1)), 'Plan is not the fixed unique 1..limit prefix')
    bodies = set()
    for entry in plan:
        source = stage / 'sources' / f'{entry["candidate"]:04d}.cu'
        require(digest(source) == entry['source_sha256'], 'Frozen candidate source changed')
        body = source.read_bytes()
        if body.startswith(b'// H3_AUTORESEARCH_GENERATED '):
            body = body.partition(b'\n')[2]
        require(body not in bodies and body != (seed / 'cuda/h3_gpu_cuda.cu').read_bytes(), 'Repeated or unchanged candidate body')
        bodies.add(body)
    prompt_hash = digest(stage / 'prompt.txt')
    require(digest(args.oracle / 'manifest.json') == args.oracle_sha256, 'Oracle manifest changed')
    require(read_json(args.oracle / 'manifest.json')['configuration']['prompt'] == (stage / 'prompt.txt').read_text().strip(), 'Prompt differs from fixed oracle')
    baseline, baseline_score = validate_baseline(args.baseline_json, seed, prompt_hash)
    legacy = validate_legacy(args.legacy_manifest, plan)
    preflight = read_json(stage / 'preflight/build/build.json')
    old_progress = read_json(stage / 'results/progress.json')
    old_terminal = read_json(legacy['candidates'][0]['path'])
    require(identity_hash(old_progress['campaign_identity']) == old_terminal['campaign_identity_sha256'], 'Legacy accepted-object provenance differs')
    accepted_hashes = old_progress['campaign_identity']['reused_objects']
    require(all(digest(ACCEPTED / 'build-cuda' / name) == wanted for name, wanted in accepted_hashes.items()),
            'Accepted build objects changed since the original campaign')
    require(all(digest(seed / 'build-cuda' / name) == wanted for name, wanted in preflight['binaries'].items()), 'Seed binaries differ from accepted preflight')
    seed_hashes = {str(path.relative_to(seed)): digest(path) for path in sorted(seed.rglob('*'))
                   if path.is_file() and not any(part in ('build-cuda', '.git', '.deps', '__pycache__', 'evidence') for part in path.relative_to(seed).parts)
                   and (path.suffix in ('.c', '.h', '.cu', '.cuh', '.cpp', '.hpp', '.inc', '.py', '.json') or path.name == 'Makefile')}
    harness = {name: digest(directory / name) for name in ('worker.py', 'runtime.py', 'verify.py', 'frozen_primitives.py')}
    identity = dict(protocol_id=PROTOCOL_ID, protocol=PROTOCOL, plan_sha256=digest(stage / 'plan.json'), limit=args.limit,
                    prompt_sha256=prompt_hash, oracle_sha256=args.oracle_sha256, baseline_sha256=digest(args.baseline_json),
                    legacy_sha256=digest(args.legacy_manifest), harness=harness, worker_fast_sha256=digest(Path(__file__)),
                    seed_sha256=identity_hash(seed_hashes), accepted_objects=accepted_hashes)
    campaign_hash = identity_hash(identity)
    state = dict(protocol_id=PROTOCOL_ID, campaign_identity=identity, campaign_identity_sha256=campaign_hash,
                 status='running', completed=[1], legacy_candidates=[1], legacy_manifest=legacy,
                 baseline_score=baseline_score, baseline_samples=baseline['samples'], incumbent=0, incumbent_score=baseline_score,
                 final_validation_required=True, score='median of exactly two cold-process E2E seconds')
    existing = results / 'progress.json'
    if existing.exists():
        prior = read_json(existing)
        require(prior['campaign_identity_sha256'] == campaign_hash, 'Fast campaign identity changed on resume')
        completed = prior['completed']
        require(1 <= len(completed) <= args.limit and completed == list(range(1, len(completed) + 1)), 'Completed rounds are not a unique bounded prefix')
        after = dict(incumbent=0, incumbent_score=baseline_score)
        for number in completed[1:]:
            terminal = read_json(results / f'{number:04d}/result.json')
            check_terminal(terminal, plan[number - 1], campaign_hash)
            require(terminal['benchmark_parent'] == after['incumbent'] and terminal['parent_score_seconds'] == after['incumbent_score'], 'Completed terminal chain differs')
            after = terminal['state_after']
        require(all(prior[key] == after[key] for key in after), 'Progress differs from its durable terminal state')
        state.update(after); state['completed'] = completed

    def persist():
        write_json(results / 'progress.json', state)
        write_json(shared / 'progress.json', state)

    archives = Archives(shared)
    try:
        persist()
        for entry in plan[1:]:
            number = entry['candidate']; folder = results / f'{number:04d}'
            if (folder / 'result.json').exists():
                prior = read_json(folder / 'result.json'); check_terminal(prior, entry, campaign_hash)
                if number not in state['completed']:
                    require(prior['benchmark_parent'] == state['incumbent'] and prior['parent_score_seconds'] == state['incumbent_score'], 'Terminal replay parent differs')
                    write_json(shared / f'{number:04d}/result.json', prior)
                    state.update(prior['state_after']); state['completed'].append(number); persist()
                archives.submit(number, folder)
                continue
            require(number not in state['completed'], 'Completed round lacks terminal result')
            if folder.exists():
                retries = results / 'retries'; retries.mkdir(exist_ok=True)
                folder.rename(retries / f'{number:04d}-{time.time_ns()}')
            folder.mkdir()
            require(harness == {name: digest(directory / name) for name in harness} and
                    digest(Path(__file__)) == identity['worker_fast_sha256'] and digest(stage / 'plan.json') == identity['plan_sha256'],
                    'Frozen harness, worker or plan changed')
            state.update(active=number, phase='evaluation'); persist()
            started_unix, started = time.time(), time.monotonic()
            record = dict(entry, terminal=False, protocol_id=PROTOCOL_ID, campaign_identity_sha256=campaign_hash,
                          started_unix=started_unix, benchmark_parent=state['incumbent'], parent_score_seconds=state['incumbent_score'],
                          provisional=True, fullgate=False, correctness=None, score_seconds=None, attempts=[])
            payload = dict(stage=str(stage), shared=str(shared), folder=str(folder), entry=entry, oracle=str(args.oracle.resolve()),
                           oracle_sha256=args.oracle_sha256, prompt_sha256=prompt_hash, seed_hashes=seed_hashes,
                           immutable_objects=preflight['reused_objects'], accepted_hashes=accepted_hashes,
                           parent_score=state['incumbent_score'], started_unix=started_unix)
            write_json(folder / 'input.json', payload)
            outcome = supervise([sys.executable, str(Path(__file__).resolve()), '_round', str(folder / 'input.json')],
                                PROTOCOL['evaluation_seconds'] - (time.monotonic() - started))
            if (folder / 'evaluation.json').exists():
                record.update(read_json(folder / 'evaluation.json'))
            if outcome['budget_exceeded']:
                record.update(status='discard', reason='budget_exceeded', budget_exceeded=True, score_seconds=None)
                record['attempts'].append(dict(stage='run' if record.get('build') else 'build', outcome='budget_exceeded', phase=record.get('phase')))
            else:
                require(outcome['returncode'] == 0 and record.get('status') in ('keep', 'discard', 'crash'),
                        'Round infrastructure error: ' + str(record.get('error', outcome)))
                record['budget_exceeded'] = False
            after = dict(incumbent=number if record['status'] == 'keep' else state['incumbent'],
                         incumbent_score=record['score_seconds'] if record['status'] == 'keep' else state['incumbent_score'])
            record.update(terminal=True, finished_unix=time.time(), round_seconds=time.monotonic() - started,
                          evaluation_limit_seconds=295, round_limit_seconds=300, state_after=after)
            check_terminal(record, entry, campaign_hash)
            write_json(folder / 'result.json', record)
            # The small shared terminal is durable before advancing the prefix.
            write_json(shared / f'{number:04d}/result.json', record)
            state.update(after); state['completed'].append(number)
            state['last_result'] = {key: record.get(key) for key in ('candidate', 'status', 'correctness', 'score_seconds', 'reason', 'budget_exceeded', 'round_seconds')}
            persist()
            archives.submit(number, folder)
            print(json.dumps(dict(event='trial_complete', **state['last_result'])), flush=True)
        state.update(status='evaluations_complete', active=None, phase='archive_drain'); persist()
    except BaseException as error:
        state.update(status='infrastructure_error', error=str(error)); persist()
        raise
    finally:
        archives.finish()
    state.update(status='complete', phase='finished', archive_error=archives.failed); persist()
    print(json.dumps(state), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
