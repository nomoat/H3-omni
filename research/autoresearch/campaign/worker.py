"""GPU worker: fixed judge, fixed cold-start score, durable per-trial records.

Candidate Git commits are created in the local ledger before submission. This
worker consumes their immutable source hashes and never edits the judge.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from runtime import benchmark, build, digest, write_json
import runtime as runtime_module


class InfrastructureError(RuntimeError):
    pass


class CandidateFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise InfrastructureError(message)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def numeric_failure(report):
    """Only explicit comparison rows, never a top-level default FAIL."""
    for row in report.get('comparisons', []) + report.get('tokenizer', []):
        if row.get('passed') is False:
            return True
    for name in ('cases', 'layout_cases'):
        for row in report.get(name, []):
            if numeric_failure(row):
                return True
            if row.get('passed') is False and any(k in row for k in
                    ('relative_l2', 'normalized_max', 'max_abs', 'max_error', 'max_abs_error', 'finite')):
                return True
    return any(numeric_failure(report[k]) for k in ('primitives', 'host')
               if isinstance(report.get(k), dict))


INFRA_MARKERS = ('no space left on device', 'disk quota exceeded', 'read-only file system',
                 'input/output error', 'no such file or directory', 'cannot open', 'permission denied',
                 'driver version is insufficient', 'no cuda-capable device', 'initialization error',
                 'system not yet initialized', 'system has unsupported display driver',
                 'modulenotfounderror', 'importerror', 'frozen oracle', 'hash differs', 'harness changed')
NATIVE_MARKERS = ('cuda: out of memory', 'cuda: an illegal memory access',
                  'cuda: invalid configuration argument', 'cuda: too many resources requested',
                  'cuda: invalid argument', 'cublas status ', 'cudnn:',
                  'requested cublaslt fp32 bias algorithm rank is unavailable',
                  'cublaslt heuristic returned an unusable algorithm')


def log_tail(path):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        return stream.read().decode(errors='replace')


def classify_verification(report, output):
    if report.get('passed') is True:
        return report
    details = str(report.get('error', '')).lower()
    if any(marker in details for marker in INFRA_MARKERS):
        raise InfrastructureError('Judge infrastructure failure: ' + details)
    if numeric_failure(report):
        return report
    unit_path = output / 'units.json'
    if unit_path.is_file():
        units = json.loads(unit_path.read_text())
        unit_rows=[row for group in ('primitives','host') for category in ('cases','layout_cases')
                   for row in units.get(group,{}).get(category,[])]
        if numeric_failure(units) or any(row.get('passed') is False for row in unit_rows):
            return report
    details += '\n' + log_tail(output.with_suffix('.log')).lower()
    for child in report.get('subprocesses', []):
        if child.get('log'):
            details += '\n' + log_tail(Path(child['log'])).lower()
    if any(marker in details for marker in INFRA_MARKERS):
        raise InfrastructureError('Judge infrastructure failure; see ' + str(output))
    if any(marker in details for marker in NATIVE_MARKERS):
        return report
    if any(child.get('returncode') in (-signal.SIGSEGV,-signal.SIGABRT,-signal.SIGFPE,-signal.SIGILL)
           for child in report.get('subprocesses', [])):
        return report
    raise InfrastructureError('Judge FAIL lacks explicit numeric/native candidate evidence: ' + str(report.get('error')))


def benchmark_checked(root, weights, prompt, output, expected_prompt_hash, samples=3):
    require(digest(prompt) == expected_prompt_hash, 'Frozen benchmark prompt changed')
    try:
        return benchmark(root, weights, prompt, output, samples=samples)
    except subprocess.TimeoutExpired as error:
        # A timeout alone cannot distinguish a candidate hang from a stalled
        # filesystem or a lost GPU. Stop without consuming a round.
        raise InfrastructureError('Ambiguous benchmark timeout; inspect/retry the same candidate') from error
    except subprocess.CalledProcessError as error:
        details = '\n'.join(log_tail(p) for p in sorted(output.glob('native-*.log'))).lower()
        details += '\n' + str(error.stderr or '').lower()
        if any(marker in details for marker in INFRA_MARKERS):
            raise InfrastructureError('Benchmark infrastructure failure; see ' + str(output)) from error
        command = error.cmd if isinstance(error.cmd, list) else []
        native = bool(command) and Path(command[0]).resolve() == (root / 'build-cuda/h3').resolve()
        if native and (error.returncode in (-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGFPE, -signal.SIGILL)
                       or any(marker in details for marker in NATIVE_MARKERS)):
            raise CandidateFailure('Native candidate crashed; see ' + str(output)) from error
        if not native and any(marker in details for marker in
                ('invalid data found when processing input', 'moov atom not found', 'matches no streams')):
            raise CandidateFailure('Candidate MP4 is malformed or lacks audio') from error
        raise InfrastructureError('Unclassified benchmark subprocess failure; see ' + str(output)) from error
    except AssertionError as error:
        # The fixed prompt was checked above; runtime's remaining assertion
        # checks decoded frame count, dimensions and the presence of audio.
        raise CandidateFailure('Candidate MP4 failed the fixed media contract') from error


def check_terminal(prior, entry, campaign_hash):
    require(all(prior.get(k) == value for k, value in entry.items()), 'Terminal candidate identity differs from plan')
    require(prior.get('campaign_identity_sha256') == campaign_hash, 'Terminal campaign identity differs')
    require(prior.get('terminal') is True and prior.get('status') in ('keep', 'discard', 'crash'), 'Malformed terminal result')
    require(isinstance(prior.get('state_after'), dict), 'Terminal result has no durable state transition')
    after=prior['state_after']
    require(set(after)=={'incumbent','incumbent_score'} and
            after['incumbent']==(entry['candidate'] if prior['status']=='keep' else prior.get('benchmark_parent')) and
            isinstance(after['incumbent_score'],(float,int)) and math.isfinite(after['incumbent_score']) and after['incumbent_score']>0,
            'Terminal state transition is inconsistent')
    if prior['status'] == 'keep':
        require(prior.get('correctness') is True and prior.get('fullgate') is True and
                prior.get('confirmation', {}).get('passed') is True and
                isinstance(prior.get('score_seconds'), (float, int)) and
                math.isfinite(prior['score_seconds']) and prior['score_seconds'] > 0,
                'Kept terminal lacks correctness/timing evidence')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--shared',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--oracle-sha256',required=True)
    p.add_argument('--limit',type=int,default=100)
    args=p.parse_args()
    stage=args.stage.resolve();shared=args.shared.resolve()
    seed=stage/'seed';prompt=stage/'prompt.txt'
    weights=Path('/var/tmp/h3-cuda-14228/local-model/FL2VA')
    accepted=Path('/var/tmp/h3-cuda-14274')
    judge=seed/'research/autoresearch/campaign/verify.py'
    all_entries=json.loads((stage/'plan.json').read_text())
    require(1 <= args.limit <= 100 and len(all_entries) >= args.limit, 'Requested round limit exceeds the frozen plan')
    plan=all_entries[:args.limit]
    numbers=[entry['candidate'] for entry in plan]
    require(all(type(n) is int and n > 0 for n in numbers) and numbers == sorted(set(numbers)), 'Candidate IDs must be unique, positive and ordered')
    source_hashes=set();body_hashes=set()
    for entry in plan:
        source=stage/'sources'/f'{entry["candidate"]:04d}.cu'
        actual=digest(source)
        require(actual == entry['source_sha256'], 'Candidate source hash differs from the frozen plan')
        body=source.read_bytes()
        if body.startswith(b'// H3_AUTORESEARCH_GENERATED '):
            body=body.partition(b'\n')[2]
        body_hash=hashlib.sha256(body).hexdigest()
        require(actual not in source_hashes and body_hash not in body_hashes, 'Duplicate candidate source/body')
        require(body != (seed/'cuda/h3_gpu_cuda.cu').read_bytes(), 'Candidate is the unchanged seed')
        source_hashes.add(actual);body_hashes.add(body_hash)
    preflight_build=json.loads((stage/'preflight/build/build.json').read_text())
    immutable_objects=preflight_build['reused_objects']
    expected_harness={str(f.relative_to(judge.parent)):digest(f) for f in sorted(judge.parent.glob('*.py'))}
    # Freeze the actual imported worker/runtime, not only copies beside judge.
    expected_harness['@executed_worker']=digest(Path(__file__))
    expected_harness['@imported_runtime']=digest(Path(runtime_module.__file__))
    seed_hashes={str(f.relative_to(seed)):digest(f) for f in sorted(seed.rglob('*'))
                 if f.is_file() and not any(part in ('build-cuda','.git','.deps','__pycache__','evidence')
                                            for part in f.relative_to(seed).parts)
                 and (f.suffix in ('.c','.h','.cu','.cuh','.cpp','.hpp','.inc','.py','.json') or f.name == 'Makefile')}
    accepted_hashes={name:digest(accepted/'build-cuda'/name) for name in
                     sorted({s+'.o' for s in immutable_objects} | {'main.o','h3_cli.o','linenoise.o','h3_gpu_cuda.o','libh3_tokenizer.so'})}
    require(all(accepted_hashes[s+'.o'] == h for s,h in immutable_objects.items()), 'Accepted objects differ from preflight')
    require(all(digest(seed/'build-cuda'/name)==wanted for name,wanted in preflight_build['binaries'].items()),
            'Seed binaries differ from the preflight build')
    prompt_hash=digest(prompt)
    prompt_text=prompt.read_text().strip()
    require(bool(prompt_text) and '\n' not in prompt_text and '\t' not in prompt_text, 'Fixed prompt must be one nonempty line')
    require(digest(args.oracle/'manifest.json') == args.oracle_sha256, 'Frozen oracle manifest differs')
    require(json.loads((args.oracle/'manifest.json').read_text())['configuration']['prompt'] == prompt_text,
            'Benchmark prompt differs from the fixed oracle prompt')
    campaign_identity=dict(plan_sha256=digest(stage/'plan.json'),limit=args.limit,prompt_sha256=prompt_hash,
                           seed_sha256=identity_hash(seed_hashes),reused_objects=accepted_hashes,
                           harness=expected_harness,oracle_sha256=args.oracle_sha256,
                           preflight_build_sha256=digest(stage/'preflight/build/build.json'),
                           preflight_baseline_sha256=digest(stage/'preflight/baseline/benchmark.json'))
    campaign_hash=identity_hash(campaign_identity)
    shared.mkdir(parents=True,exist_ok=True)
    results=stage/'results';results.mkdir(exist_ok=True)
    progress_file=results/'progress.json'
    if progress_file.exists():
        state=json.loads(progress_file.read_text())
        require(state.get('campaign_identity_sha256') == campaign_hash, 'Resume campaign identity changed')
        require(state['completed'] == numbers[:len(state['completed'])], 'Completed rounds are not a unique plan prefix')
        for number in state['completed']:
            prior=json.loads((results/f'{number:04d}'/'result.json').read_text())
            check_terminal(prior,next(e for e in plan if e['candidate']==number),campaign_hash)
        if state['completed']:
            require(all(state.get(k) == v for k,v in prior['state_after'].items()), 'Progress disagrees with last durable terminal')
        state['status']='running';state.pop('error',None)
    else:
        state=dict(status='running',completed=[],incumbent=0,incumbent_score=None,
                   oracle_sha256=args.oracle_sha256,harness_sha256=expected_harness,
                   campaign_identity_sha256=campaign_hash,campaign_identity=campaign_identity,
                   score='median cold-process fixed-prompt seed-7 E2E seconds',
                   file_cache_policy='OS/filesystem caches are not forcibly cleared on the shared node')

    def persist():
        write_json(progress_file,state)
        write_json(shared/'progress.json',state)

    def verify(root, output, mode):
        output.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(judge),mode,'--oracle',str(args.oracle),'--oracle-sha256',args.oracle_sha256,
             '--candidate-root',str(root),'--weights',str(weights),'--diffusers',str(accepted/'diffusers'),'--output',str(output)]
        started=time.monotonic()
        with output.with_suffix('.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        if not (output/'verification.json').exists():
            detail=log_tail(output.with_suffix('.log')).lower()
            if result.returncode in (-signal.SIGSEGV,-signal.SIGABRT,-signal.SIGFPE,-signal.SIGILL) and not any(m in detail for m in INFRA_MARKERS):
                return dict(passed=False,error='Candidate evaluation process terminated by signal',
                            returncode=result.returncode,wall_seconds_including_checks_and_io=time.monotonic()-started)
            raise InfrastructureError('Judge produced no complete verification report; see '+str(output))
        report=json.loads((output/'verification.json').read_text())
        if result.returncode and report.get('passed'):
            raise RuntimeError('judge process failed after claiming PASS')
        return classify_verification(report,output)

    def root_for(number):
        return seed if number==0 else stage/'candidates'/f'{number:04d}'

    def archive(number,folder):
        target=shared/f'{number:04d}'
        target.mkdir(exist_ok=True)
        # Compact result becomes visible first. Full tensors/logs follow.
        shutil.copy2(folder/'result.json',target/'result.json')
        tmp=target/'evidence.tar.pending'
        subprocess.run(['tar','-cf',str(tmp),'-C',str(folder),'.'],check=True,timeout=600)
        tmp.rename(target/'evidence.tar')

    def finish_trial(number,folder,record,state_after):
        record.update(terminal=True,finished_unix=time.time(),state_after=state_after)
        write_json(folder/'result.json',record)
        archive(number,folder)
        # Publish the new incumbent only after its terminal evidence is durable.
        state.update(state_after)
        state['completed'].append(number)
        state['last_result']={k:record.get(k) for k in ('candidate','status','correctness','fullgate','score_seconds','reason')}
        persist()
        print(json.dumps(dict(event='trial_complete',**state['last_result'])),flush=True)

    if state['incumbent_score'] is None:
        baseline=results/'baseline';baseline.mkdir(exist_ok=True)
        gate_path=baseline/'fullgate'
        full=json.loads((gate_path/'verification.json').read_text()) if (gate_path/'verification.json').exists() else verify(seed,gate_path,'promote')
        require(full.get('candidate_binary_sha256')==preflight_build['binaries'], 'Baseline gate refers to different binaries')
        if not full['passed']:
            raise RuntimeError('Frozen judge rejects accepted seed: '+str(full.get('error')))
        base=json.loads((stage/'preflight/baseline/benchmark.json').read_text())
        require(base.get('passed') is True and len(base.get('samples',[])) == 3 and
                math.isfinite(base['score_seconds']) and base['score_seconds'] > 0 and
                all(r['seed']==7 and r['prompt_sha256']==hashlib.sha256(prompt_text.encode()).hexdigest()
                    for r in base.get('records',[])) and len(base.get('records',[]))==3,
                'Preflight benchmark does not match the fixed cold-start request')
        state['incumbent_score']=base['score_seconds']
        state['baseline_score']=base['score_seconds']
        state['baseline_samples']=base['samples']
        state['baseline_fullgate']=str(gate_path/'verification.json')
        state['baseline_fullgate_seconds']=full['wall_seconds_including_checks_and_io']
        persist()

    for entry in plan:
        number=entry['candidate']
        if number in state['completed']:
            continue
        # A harness mutation is an infrastructure error, never a failed trial.
        current_harness={str(f.relative_to(judge.parent)):digest(f) for f in sorted(judge.parent.glob('*.py'))}
        current_harness.update({'@executed_worker':digest(Path(__file__)), '@imported_runtime':digest(Path(runtime_module.__file__))})
        require(expected_harness==current_harness, 'Frozen worker/runtime/judge changed')
        require(digest(stage/'plan.json')==campaign_identity['plan_sha256'] and digest(prompt)==prompt_hash,
                'Frozen plan or prompt changed')
        folder=results/f'{number:04d}'
        if folder.exists() and not (folder/'result.json').exists():
            retries=results/'retries';retries.mkdir(exist_ok=True)
            folder.rename(retries/f'{number:04d}-{time.time_ns()}')
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists():
            prior=json.loads((folder/'result.json').read_text())
            if prior.get('terminal'):
                check_terminal(prior,entry,campaign_hash)
                require(prior['benchmark_parent']==state['incumbent'], 'Terminal replay parent differs from current incumbent')
                archive(number,folder)
                state.update(prior['state_after'])
                state['completed'].append(number);persist();continue
        root=root_for(number)
        if not root.exists():
            shutil.copytree(seed,root,ignore=shutil.ignore_patterns('build-cuda','research','__pycache__'))
        require(all((root/name).is_file() and digest(root/name)==wanted for name,wanted in seed_hashes.items()
                    if not name.startswith('research/') and name != 'cuda/h3_gpu_cuda.cu'),
                'Candidate changed a source/header outside the declared mutation surface')
        source=stage/'sources'/f'{number:04d}.cu'
        require(digest(source)==entry['source_sha256'], 'Candidate source changed after plan validation')
        shutil.copy2(source,root/'cuda/h3_gpu_cuda.cu')
        record=dict(entry,terminal=False,status='running',correctness=False,fullgate=False,score_seconds=None,
                    benchmark_parent=state['incumbent'],campaign_identity_sha256=campaign_hash,
                    started_unix=time.time(),attempts=[])
        state_after={k:state[k] for k in ('incumbent','incumbent_score')}
        state['active']=number;state['phase']='build';persist()
        write_json(folder/'inflight.json',record)
        try:
            try:
                built=build(root,accepted,folder/'build')
                if built['reused_objects'] != immutable_objects:
                    raise RuntimeError('Immutable accepted production objects changed')
                require(all(digest(root/'build-cuda'/name)==wanted for name,wanted in accepted_hashes.items()
                            if name != 'h3_gpu_cuda.o'), 'A reused CLI/model object or tokenizer changed')
            except subprocess.CalledProcessError as error:
                require(not any(m in log_tail(folder/'build/build.log').lower() for m in INFRA_MARKERS),
                        'Compiler/linker infrastructure failure; inspect build log')
                record['attempts'].append(dict(stage='build',outcome='candidate_error'))
                record.update(status='crash',reason='candidate compilation failed',error=str(error))
            else:
                record['attempts'].append(dict(stage='build',outcome='success'))
                record['build']=built
                state['phase']='correctness_screen';persist()
                screen=verify(root,folder/'screen','screen')
                record['screen']=dict(passed=screen['passed'],seconds=screen['wall_seconds_including_checks_and_io'],report=str(folder/'screen/verification.json'))
                record['attempts'].append(dict(stage='run',outcome='success' if screen['passed'] else 'candidate_error'))
                if not screen['passed']:
                    record.update(status='discard',reason='fixed numerical gate failed',error=screen.get('error'))
                else:
                    record['correctness']=True
                    state['phase']='cold_benchmark';persist()
                    score=benchmark_checked(root,weights,prompt,folder/'benchmark',prompt_hash,samples=3)
                    record.update(score_seconds=score['score_seconds'],samples=score['samples'])
                    if score['score_seconds'] >= state['incumbent_score'] * 0.995:
                        record.update(status='discard',reason='no preliminary E2E improvement beyond 0.5%; fullgate not run')
                    else:
                        state['phase']='full_correctness_gate';persist()
                        full=verify(root,folder/'fullgate','promote')
                        record['fullgate']=full['passed']
                        record['fullgate_report']=str(folder/'fullgate/verification.json')
                        if not full['passed']:
                            record.update(status='discard',correctness=False,reason='full fixed fixture matrix failed',error=full.get('error'))
                        else:
                            state['phase']='paired_confirmation';persist()
                            parent_root=root_for(state['incumbent'])
                            pairs=[]
                            for pair in range(5):
                                values={}
                                for role in (('parent','candidate') if pair%2==0 else ('candidate','parent')):
                                    rr=parent_root if role=='parent' else root
                                    out=folder/'confirmation'/f'{pair:02d}-{role}'
                                    try:
                                        trial=benchmark_checked(rr,weights,prompt,out,prompt_hash,samples=1)
                                    except CandidateFailure as error:
                                        if role=='parent':
                                            raise InfrastructureError('Already accepted incumbent failed during confirmation') from error
                                        raise
                                    values[role]=trial['score_seconds']
                                pairs.append(values)
                                write_json(folder/'confirmation/pairs.json',pairs)
                            differences=[v['parent']-v['candidate'] for v in pairs]
                            improvement=statistics.median(differences)
                            parent_score=statistics.median(v['parent'] for v in pairs)
                            candidate_score=statistics.median(v['candidate'] for v in pairs)
                            mad=statistics.median(abs(d-improvement) for d in differences)
                            noise_floor=max(0.005*parent_score,2*mad/math.sqrt(5))
                            keep=improvement>noise_floor and sum(d>0 for d in differences)>=4 and candidate_score < parent_score * .995
                            record.update(score_seconds=candidate_score,confirmation=dict(pairs=pairs,median_improvement_seconds=improvement,noise_floor_seconds=noise_floor,passed=keep),status='keep' if keep else 'discard',reason='confirmed cold E2E improvement' if keep else 'improvement not stable in paired confirmation')
                            if keep:
                                state_after.update(incumbent=number,incumbent_score=candidate_score)
                            else:
                                state_after['incumbent_score']=parent_score
            finish_trial(number,folder,record,state_after)
        except CandidateFailure as error:
            record['attempts'].append(dict(stage='run',phase=state['phase'],outcome='candidate_error'))
            record.update(status='crash',correctness=False,reason=str(error),error=str(error),score_seconds=None)
            finish_trial(number,folder,record,state_after)
        except Exception as error:
            # Unknown failures are NOT counted as search rounds. Preserve and stop.
            record.update(terminal=False,status='infrastructure_error',error=str(error),traceback=traceback.format_exc())
            write_json(folder/'infrastructure-error.json',record)
            state.update(status='infrastructure_error',error=str(error));persist()
            raise
    state.update(status='complete',active=None,phase='finished');persist()
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
