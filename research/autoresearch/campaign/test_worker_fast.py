"""CPU regressions for the v2 timing budget, evidence and keep guard."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import worker_fast as worker


class FastWorkerTests(unittest.TestCase):
    def test_keep_requires_both_samples_and_half_percent_median(self):
        self.assertEqual(worker.decision([98.0, 99.0], 100.0), (98.5, True))
        self.assertFalse(worker.decision([98.0, 100.0], 100.0)[1])
        self.assertFalse(worker.decision([99.4, 99.6], 100.0)[1])
        for values in ([1], [1, 2, 3], [float('nan'), 1], [0, 1]):
            with self.assertRaises(RuntimeError):
                worker.decision(values, 100.0)

    def test_supervisor_kills_grandchild_after_leader_exits_on_term(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / 'grandchild.pid'
            code = (
                'import os,signal,sys,time\n'
                'pid=os.fork()\n'
                'if pid == 0:\n'
                ' signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                ' open(sys.argv[1],"w").write(str(os.getpid()))\n'
                ' time.sleep(60)\n'
                'else:\n'
                ' signal.signal(signal.SIGTERM,lambda *args:sys.exit(0))\n'
                ' time.sleep(60)\n'
            )
            started = time.monotonic()
            result = worker.supervise([sys.executable, '-c', code, str(pidfile)], .4)
            self.assertTrue(result['budget_exceeded'])
            self.assertLess(time.monotonic() - started, 2)
            pid = int(pidfile.read_text())
            for _ in range(20):
                stat = Path(f'/proc/{pid}/stat')
                if not stat.exists() or stat.read_text().split()[2] == 'Z':
                    break
                time.sleep(.01)
            else:
                os.kill(pid, signal.SIGKILL)
                self.fail('grandchild survived its process group budget')

    def test_supervisor_accepts_normal_completion(self):
        result = worker.supervise([sys.executable, '-c', 'pass'], 1)
        self.assertEqual(result, {'returncode': 0, 'budget_exceeded': False})

    def test_baseline_proof_requires_two_distinct_accepted_parent_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = {'samples': [85.0, 84.0], 'provenance': []}
            for index, seconds in enumerate(baseline['samples']):
                path = root / f'{index}.json'
                report = {'passed': True, 'samples': [seconds], 'records': [
                    {'seconds': seconds, 'seed': 7, 'prompt_sha256': 'prompt', 'command': [str(root / 'seed/build-cuda/h3')]}]}
                worker.write_json(path, report)
                baseline['provenance'].append({'path': str(path), 'sha256': worker.digest(path)})
            path = root / 'baseline.json'; worker.write_json(path, baseline)
            self.assertEqual(worker.validate_baseline(path, root / 'seed', 'prompt')[1], 84.5)
            baseline['provenance'][1] = baseline['provenance'][0]
            worker.write_json(path, baseline)
            with self.assertRaises(RuntimeError):
                worker.validate_baseline(path, root / 'seed', 'prompt')

    def test_legacy_counts_only_frozen_first_discard_and_keeps_old_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {'candidate': 1, 'source_sha256': 'source'}
            result = dict(entry, terminal=True, status='discard', state_after={'incumbent': 0}, samples=[86, 87, 88], round_seconds=363)
            worker.write_json(root / 'result.json', result)
            manifest = {'candidates': [{'candidate': 1, 'path': str(root / 'result.json'), 'sha256': worker.digest(root / 'result.json')}]}
            worker.write_json(root / 'legacy.json', manifest)
            self.assertEqual(worker.validate_legacy(root / 'legacy.json', [entry]), manifest)
            result['status'] = 'keep'; worker.write_json(root / 'result.json', result)
            with self.assertRaises(RuntimeError):
                worker.validate_legacy(root / 'legacy.json', [entry])

    def test_timeout_discard_replays_without_claiming_numeric_failure(self):
        entry = {'candidate': 2, 'source_sha256': 'source'}
        result = dict(entry, terminal=True, protocol_id=worker.PROTOCOL_ID, campaign_identity_sha256='id',
                      status='discard', correctness=None, score_seconds=None, budget_exceeded=True,
                      fullgate=False, provisional=True,
                      reason='budget_exceeded', benchmark_parent=0, parent_score_seconds=85,
                      state_after={'incumbent': 0, 'incumbent_score': 85})
        worker.check_terminal(result, entry, 'id')
        result.update(status='keep', samples=[80, 81], score_seconds=80.5,
                      screen={'passed': True}, state_after={'incumbent': 2, 'incumbent_score': 80.5})
        with self.assertRaises(RuntimeError):
            worker.check_terminal(result, entry, 'id')


if __name__ == '__main__':
    unittest.main()
