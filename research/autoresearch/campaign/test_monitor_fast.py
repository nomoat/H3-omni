"""CPU-only checks for the separate fast protocol observer and journal."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import monitor_fast as fast


class FastMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = []
        for number in (1,2):
            body = f'candidate {number}'.encode()
            self.write(f'sources/{number:04d}.cu', body)
            self.entries.append(dict(candidate=number,commit=f'commit{number}',parent_commit='baseline',
                               source_sha256=fast.digest(body),patch_sha256=f'patch{number}', config={'threads':number}))
        self.write('plan.json', self.entries)
        self.write('protocol.json', {'prompt_sha256':'p'*64, 'seed':7})
        provenance=[]
        for index in range(2):
            name=f'{index:02d}-parent-benchmark.json'
            self.write('evidence/fast-inputs/'+name,dict(passed=True,samples=[100],
                       records=[dict(seconds=100,seed=7,prompt_sha256='p'*64)]))
            provenance.append(dict(path='/remote/'+name,
                              sha256=fast.digest((self.root/'evidence/fast-inputs'/name).read_bytes())))
        self.write('fast-baseline.json',dict(samples=[100,100],score_seconds=100,provenance=provenance))
        identity = dict(protocol_id=fast.PROTOCOL_ID,protocol=fast.PROTOCOL,
                        plan_sha256=fast.digest((self.root/'plan.json').read_bytes()),
                        prompt_sha256='p'*64,limit=2,
                        baseline_sha256=fast.digest((self.root/'fast-baseline.json').read_bytes()))
        self.progress = dict(protocol_id=fast.PROTOCOL_ID,campaign_identity=identity,
                             campaign_identity_sha256=fast.digest(fast.encoded(identity)),
                             completed=[],legacy_candidates=[],active=1,phase='build',
                             status='running',baseline_score=100,baseline_samples=[100,100],incumbent=0,incumbent_score=100)
        self.plan = fast.load_plan(self.root, check_git=False)

    def write(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else fast.encoded(value)+b'\n')

    def result(self, number=1, **changes):
        value = dict(self.plan[number],protocol_id=fast.PROTOCOL_ID,
                     campaign_identity_sha256=self.progress['campaign_identity_sha256'],
                     terminal=True,status='keep',correctness=True,fullgate=False,provisional=True,
                     started_unix=100,finished_unix=299,attempts=[dict(stage='run',outcome='success')],
                     screen=dict(passed=True),samples=[90,92],score_seconds=91,
                     benchmark_parent=0,state_after=dict(incumbent=number,incumbent_score=91),reason='faster')
        value.update(changes)
        return value

    def archive(self, results, progress=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:') as archive:
            payloads = {'progress.json':fast.encoded(progress or self.progress)+b'\n'}
            payloads.update({f'{r["candidate"]:04d}/result.json':fast.encoded(r)+b'\n' for r in results})
            for name, raw in payloads.items():
                member=tarfile.TarInfo(name);member.size=len(raw)
                archive.addfile(member,io.BytesIO(raw))
        return output.getvalue()

    def test_sync_replay_deduplicates_and_does_not_touch_v1(self):
        self.write('events.jsonl', b'v1 stays exact\n')
        self.write('status.json', b'v1 status exact\n')
        result = self.result()
        self.progress.update(completed=[1],incumbent=1,incumbent_score=91)
        for _ in range(2):
            progress = fast.install_archive(self.root,self.archive([result]),self.plan)
            records = fast.replay(self.root,progress,self.plan)
            self.assertEqual(len(records),1)
        self.assertEqual(len((self.root/'fast-events.jsonl').read_text().splitlines()),1)
        self.assertEqual((self.root/'events.jsonl').read_bytes(),b'v1 stays exact\n')
        self.assertEqual((self.root/'status.json').read_bytes(),b'v1 status exact\n')

    def test_timeout_is_budget_and_unfinished_screen_not_accuracy_failure(self):
        result = self.result(status='discard',correctness=None,screen={},samples=[90],score_seconds=None,
                             budget_exceeded=True,reason='budget_exceeded',
                             state_after=dict(incumbent=0,incumbent_score=100))
        self.progress['completed']=[1]
        progress=fast.install_archive(self.root,self.archive([result]),self.plan)
        records=fast.replay(self.root,progress,self.plan)
        report=fast.summarize(self.root,progress,records,dict(job=9,scheduler_state='RUNNING'),2)
        page=(self.root/'report-fast.html').read_text()
        self.assertEqual(report['budget_exceeded'],1)
        self.assertIn('<td>over budget</td><td>not finished</td>',page)
        self.assertIsNone(report['provisional_best_seconds'])

    def test_keep_requires_two_samples_and_matching_score(self):
        for changes in ({'samples':[90]}, {'score_seconds':90}, {'screen':{'passed':False},'correctness':False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fast.validate_record(self.result(**changes),self.plan[1],self.progress)

    def test_keep_rule_rejects_non_improvement(self):
        result=self.result(samples=[99,100.1],score_seconds=99.55,
                           state_after=dict(incumbent=1,incumbent_score=99.55))
        self.progress.update(completed=[1],incumbent=1,incumbent_score=99.55)
        progress=fast.install_archive(self.root,self.archive([result]),self.plan)
        with self.assertRaisesRegex(ValueError,'Keep rule'):
            fast.replay(self.root,progress,self.plan)

    def test_terminal_rewrite_and_protocol_change_rejected(self):
        result=self.result()
        fast.install_archive(self.root,self.archive([result]),self.plan)
        with self.assertRaisesRegex(ValueError,'Immutable terminal'):
            fast.install_archive(self.root,self.archive([dict(result,reason='changed')]),self.plan)
        changed=dict(self.progress,protocol_id='old-protocol')
        with self.assertRaisesRegex(ValueError,'different protocol'):
            fast.install_archive(self.root,self.archive([],changed),self.plan)

    def test_wrong_source_and_unknown_terminal_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity differs'):
            fast.install_archive(self.root,self.archive([self.result(source_sha256='wrong')]),self.plan)
        self.write('sources/0001.cu',b'changed')
        with self.assertRaisesRegex(ValueError,'source differs'):
            fast.load_plan(self.root,check_git=False)

    def test_uncommitted_terminal_does_not_count(self):
        progress=fast.install_archive(self.root,self.archive([self.result()]),self.plan)
        records=fast.replay(self.root,progress,self.plan)
        self.assertEqual(records,[])
        self.assertFalse((self.root/'fast-events.jsonl').exists())

    def test_journal_tampering_and_parent_transition_rejected(self):
        result=self.result()
        self.progress.update(completed=[1],incumbent=1,incumbent_score=91)
        progress=fast.install_archive(self.root,self.archive([result]),self.plan)
        fast.replay(self.root,progress,self.plan)
        journal=self.root/'fast-events.jsonl'
        journal.write_text(journal.read_text().replace('"status":"keep"','"status":"discard"'))
        with self.assertRaisesRegex(ValueError,'hash chain'):
            fast.replay(self.root,progress,self.plan)

    def test_legacy_cannot_be_relabelled_as_v2(self):
        legacy=dict(self.result(),protocol_id='legacy-v1')
        self.write('evidence/search/0001/result.json',legacy)
        self.write('evidence/fast-inputs/legacy-0001-result.json',legacy)
        self.write('fast-legacy-manifest.json',dict(candidates=[dict(candidate=1,path='/remote/legacy-0001-result.json',
                   sha256=fast.digest(fast.encoded(legacy)+b'\n'))]))
        self.progress['campaign_identity']['legacy_sha256']=fast.digest((self.root/'fast-legacy-manifest.json').read_bytes())
        self.progress['campaign_identity_sha256']=fast.digest(fast.encoded(self.progress['campaign_identity']))
        self.progress.update(completed=[1],legacy_candidates=[1])
        with self.assertRaisesRegex(ValueError,'relabels a legacy'):
            fast.install_archive(self.root,self.archive([self.result()]),self.plan)


if __name__=='__main__':
    unittest.main()
