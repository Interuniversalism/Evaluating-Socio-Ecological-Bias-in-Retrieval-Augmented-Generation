"""Offline tests. Fake Ollama only; never contacts a server."""
import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import typing
from unittest.mock import patch
import zipfile
import collect_judgments as c

class FakeOllama:
    def __init__(self,text='0.8',fail_first=False,done_reason='stop',tokens=20,version='0.40.16'):
        self.lock=threading.Lock();self.calls=[];self.active=0;self.peak=0;self.text=text
        self.fail_first=fail_first;self.done_reason=done_reason;self.tokens=tokens;self.version=version
        self.models=[dict(m,details={'format':'gguf'}) for m in json.loads((c.HERE/'baseline_models.json').read_text())['models'].values()]
        self.contexts={}
    def __call__(self,base,endpoint,payload=None,timeout=30):
        if endpoint=='tags':return {'models':self.models}
        if endpoint=='show':return {'model_info':{'test.context_length':131072},'template':'{{ .Prompt }}'}
        if endpoint=='version':return {'version':self.version}
        if endpoint=='ps':return {'models':[dict(m,context_length=self.contexts.get(m['name'],8192)) for m in self.models]}
        if endpoint=='generate':
            with self.lock:
                self.calls.append(payload);n=len(self.calls);self.active+=1;self.peak=max(self.peak,self.active)
                self.contexts[payload['model']]=payload['options']['num_ctx']
            time.sleep(.02)
            with self.lock:self.active-=1
            if self.fail_first and n==1:raise OSError('simulated network failure')
            text=self.text(payload) if callable(self.text) else self.text
            return {'response':text,'thinking':'reasoning retained','done':True,'done_reason':self.done_reason,
                    'prompt_eval_count':self.tokens,'eval_count':3}
        raise AssertionError(endpoint)

class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.zip=self.root/'input.zip'
        self.protocol=json.loads((c.HERE/'protocol.json').read_text())
        with zipfile.ZipFile(self.zip,'w') as z:
            for combo,domain in [('Mistral-Gemma','ecological'),('Mistral-nomic','social')]:
                buf=io.StringIO();w=csv.writer(buf);w.writerow(['question','answer','top_k_docs','context_relevance_dual','context_relevance_Answer_dual','response_relevancy_llm'])
                for i in range(3):w.writerow([f'\nQuestion {i}  ',f' Answer {i}\n',' CHUNK1 ||| CHUNK2 ||| CHUNK3 ||| CHUNK4 ',.5,.6,.8])
                z.writestr(f'root/{combo}/{domain}/context variation/data.csv',buf.getvalue())
    def tearDown(self):self.tmp.cleanup()
    def run_cli(self,server,out='run',action='run',extra=()):
        args=[action,'--zip',str(self.zip),'--out',str(self.root/out),*extra]
        with patch.object(c.core,'api',server),contextlib.redirect_stdout(io.StringIO()):return c.main(args)
    def rows(self,name,out='run'):
        with (self.root/out/name).open() as f:return list(csv.DictReader(f))
    def test_complete_parallel_resume_and_offline_export(self):
        server=FakeOllama();self.assertEqual(self.run_cli(server),0)
        self.assertEqual(len(server.calls),108);self.assertGreater(server.peak,1);self.assertLessEqual(server.peak,4)
        self.assertTrue(all(p['options']=={'num_ctx':8192,'temperature':0.} for p in server.calls))
        self.assertTrue(all('Question: Scoring request' in p['prompt'] for p in server.calls))
        self.assertEqual(len(self.rows('call_scores.csv')),108);self.assertEqual(len(self.rows('per_judge_scores.csv')),36)
        self.assertEqual(len(self.rows('legacy_scores_long.csv')),54)
        self.assertTrue(all(float(r['score'])==.8 for r in self.rows('per_judge_scores.csv')))
        self.assertEqual(self.run_cli(server),0);self.assertEqual(len(server.calls),108)
        def forbidden(*args,**kwargs):raise AssertionError('offline export contacted server')
        self.assertEqual(self.run_cli(forbidden,action='export'),0)
        events=[json.loads(l) for l in (self.root/'run/calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(events),216)
        self.assertEqual({e['job_id'] for e in events if e['event']=='request'}, {e['job_id'] for e in events if e['event']=='result'})
        self.assertTrue(all(e['raw_response']['thinking']=='reasoning retained' for e in events if e['event']=='result'))
    def test_input_preservation_and_balanced_pilot(self):
        items,files=c.load_inputs(self.zip)
        self.assertTrue(items[0]['question'].startswith('\n'));self.assertTrue(items[0]['answer'].endswith('\n'))
        self.assertEqual(items[0]['chunks'],['CHUNK1','CHUNK2','CHUNK3','CHUNK4'])
        selected=c.select_pilot(items);self.assertEqual(len(selected),4)
        self.assertEqual([i['item_id'] for i in selected],[i['item_id'] for i in c.select_pilot(items)])
        self.assertEqual(len(c.all_jobs(selected,['deepseek','llama'])),72)
    def test_actual_reference_inner_and_outer_prompts(self):
        raw=(c.HERE/'coauthor_reference.txt').read_text()
        tree=ast.parse('\n'.join('' if s.startswith('!') else s for s in raw.splitlines()))
        reference={}
        for n in tree.body:
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name) and target.id in c.METRICS.values():reference[target.id]=ast.literal_eval(n.value)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CollabEvalJudge')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_rubric_prompt');fn.decorator_list=[]
        ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'reference_wrapper','exec'),ns)
        item=c.load_inputs(self.zip)[0][0]
        for metric,var in c.METRICS.items():
            for k in c.chunk_indices(item,metric):
                inner=reference[var].format(question=item['question'],answer=item['answer'],chunk=item['chunks'][k-1] if k else '')
                expected=ns['build_rubric_prompt']('Scoring request','Candidate output','',inner)
                self.assertEqual(c.make_prompt(self.protocol,item,metric,k),expected)
    def test_failure_retry_preserves_legacy_and_fills_validated(self):
        server=FakeOllama(fail_first=True);self.assertEqual(self.run_cli(server),3)
        legacy=(self.root/'run/legacy_scores_wide.csv').read_bytes()
        self.assertEqual(sum(r['score']=='' for r in self.rows('per_judge_scores.csv')),1)
        self.assertEqual(self.run_cli(server),3);self.assertEqual(len(server.calls),108)
        self.assertEqual(self.run_cli(server,action='retry'),0);self.assertEqual(len(server.calls),109)
        self.assertEqual((self.root/'run/legacy_scores_wide.csv').read_bytes(),legacy)
        self.assertTrue(any(r['validated_attempt']=='2' for r in self.rows('call_scores.csv')))
        self.assertTrue(all(float(r['score'])==.8 for r in self.rows('per_judge_scores.csv')))
    def test_max_calls_and_config_guard(self):
        server=FakeOllama();self.assertEqual(self.run_cli(server,extra=['--max-calls','3']),0)
        self.assertEqual(len(server.calls),3)
        with self.assertRaisesRegex(ValueError,'configuration/input/model changed'):
            self.run_cli(server,extra=['--num-ctx','16384'])
        self.assertEqual(len(server.calls),3)
        self.assertEqual(self.run_cli(server),0);self.assertEqual(len(server.calls),108)
    def test_baseline_and_context_checks(self):
        with self.assertRaisesRegex(ValueError,'baseline mismatch'):self.run_cli(FakeOllama(version='different'))
        self.assertFalse((self.root/'run/calls.jsonl').exists())
        server=FakeOllama(tokens=8180)
        self.assertEqual(self.run_cli(server,extra=['--stop-after-failures','2']),3)
        self.assertLessEqual(len(server.calls),5)
        observed=[r for r in self.rows('call_scores.csv') if r['status']!='pending']
        self.assertTrue(all(r['status']=='context_boundary_warning' and r['score']=='' for r in observed))
    def test_truncated_response_does_not_enter_validated_scores(self):
        server=FakeOllama(done_reason='length')
        self.assertEqual(self.run_cli(server,extra=['--max-calls','1']),0)
        r=next(r for r in self.rows('call_scores.csv') if r['status']!='pending')
        self.assertEqual(r['status'],'incomplete_generation');self.assertEqual(r['score'],'');self.assertEqual(float(r['legacy_score']),.8)
    def test_serial_parallel_same_prompts_and_scores(self):
        a=FakeOllama();b=FakeOllama()
        self.assertEqual(self.run_cli(a,out='serial',extra=['--workers','1']),0)
        self.assertEqual(self.run_cli(b,out='parallel',extra=['--workers','4']),0)
        payload_key=lambda p:(p['model'],p['prompt'])
        self.assertEqual(sorted(a.calls,key=payload_key),sorted(b.calls,key=payload_key))
        for name in ['scores_wide.csv','legacy_scores_wide.csv','per_judge_summary.csv']:
            self.assertEqual((self.root/'serial'/name).read_bytes(),(self.root/'parallel'/name).read_bytes())
    def test_parsers_and_reference_rounding(self):
        self.assertEqual(c.parse_legacy('There are 4 claims. Score: 0.8')[0],1.)
        self.assertEqual(c.parse_explicit('There are 4 claims. Score: 0.8')[0],.8)
        self.assertIsNone(c.parse_explicit('8/10')[0]);self.assertEqual(c.parse_legacy('8/10')[0],1.)
        self.assertEqual(c.parse_legacy('no numeric output')[0],0.)
        self.assertIsNone(c.parse_explicit('no numeric output')[0])
        self.assertIsNone(c.parse_explicit('Score: 0.4\nScore: 0.8')[0])
        self.assertIsNone(c.compatibility_mean([[.8,None]]))
        self.assertEqual(c.compatibility_mean([[.3333,.3334]]),.3333)
    def test_legacy_aggregation_against_actual_coauthor_pipeline(self):
        try:import numpy as np
        except ImportError:self.skipTest('NumPy needed only to exercise the original co-author implementation')
        from concurrent.futures import ThreadPoolExecutor,as_completed
        from types import SimpleNamespace
        def answer(payload):
            first=payload['model'].startswith('deepseek')
            prompt=payload['prompt']
            if 'Answer Quality Judge' in prompt:return '0.3333' if first else '0.3334'
            if 'CHUNK2' in prompt:return 'There are 4 claims. Score: 0.6' if first else '0.8'
            if 'CHUNK3' in prompt:return 'No numeric reply' if first else '0.7'
            return '0.357' if first else '0.8'
        class OriginalLLM:
            def __init__(self,**kwargs):self.model=kwargs['model']
            def invoke(self,prompt):return answer({'model':self.model,'prompt':prompt})
        raw=(c.HERE/'coauthor_reference.txt').read_text()
        tree=ast.parse('\n'.join('' if s.startswith('!') else s for s in raw.splitlines()))
        selected={'Agent','CollabEvalJudge','CollabEvalAdapter','JudgeRecorder','clamp01',
            'context_relevance_dual','context_relevance_Answer_dual','response_relevancy_llm','evaluate_all_metrics','judge_scores'}
        defs=[]
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in selected:defs.append(n)
            if isinstance(n,ast.If):defs.extend(q for q in n.body if isinstance(q,ast.ClassDef) and q.name=='JudgeRecorder')
        ns={'np':np,'re':c.re,'Ollama':OriginalLLM,'ThreadPoolExecutor':ThreadPoolExecutor,'as_completed':as_completed,
            'trange':lambda n,**kw:range(n),**{k:getattr(typing,k) for k in ['List','Dict','Any','Optional','Callable']},
            **{k:self.protocol[k] for k in c.METRICS.values()}}
        exec(compile(ast.Module(body=defs,type_ignores=[]),'original_coauthor_definitions','exec'),ns)
        judge=ns['CollabEvalJudge']([{'model':'deepseek-r1'},{'model':'llama3.2'}],rounds=1,aggregation='weighted',weights=[.5,.5])
        item=c.load_inputs(self.zip)[0][0]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics,_=ns['evaluate_all_metrics'](item['question'],item['answer'],[SimpleNamespace(page_content=x) for x in item['chunks']],
                ns['CollabEvalAdapter'](judge),debug=False,debug_output_path=str(self.root/'reference_debug.json'))
        server=FakeOllama(text=answer)
        self.assertEqual(self.run_cli(server,extra=['--stop-after-failures','1000']),3)
        row=self.rows('legacy_scores_wide.csv')[0]
        names={'context_relevance':'context_relevance_dual','faithfulness':'context_relevance_Answer_dual','answer_relevance':'response_relevancy_llm'}
        for metric,ref in names.items():
            self.assertEqual(float(row['ensemble__'+metric]),metrics[ref])
            per=ns['judge_scores'](metrics,ref,['deepseek-r1','llama3.2'],4)
            self.assertEqual(float(row['deepseek__'+metric]),per['deepseek-r1'])
            self.assertEqual(float(row['llama__'+metric]),per['llama3.2'])
    def test_output_lock(self):
        with c.output_lock(self.root/'locked'):
            with self.assertRaisesRegex(ValueError,'Another collector'):
                with c.output_lock(self.root/'locked'):pass
    def test_partial_journal_refused(self):
        p=self.root/'broken.jsonl';p.write_text('{"run_id":')
        with self.assertRaisesRegex(ValueError,'partial record'):c.Journal(p,'x',{})

if __name__=='__main__':unittest.main()
