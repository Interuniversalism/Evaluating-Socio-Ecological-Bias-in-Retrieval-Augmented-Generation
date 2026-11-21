#!/usr/bin/env python3
"""Auditable B200 recollection of the supplied CollabEval chunk protocol.

Python 3.10+ standard library. Run audit first; run/export/retry use a separate
output directory. No model pulling, dependency installation or answer generation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics as st
import sys
import threading
import time
import zipfile
from urllib.error import HTTPError, URLError
import collector_core as core

HERE=Path(__file__).resolve().parent
METRICS={'context_relevance':'PROMPT_IS_RELEVANT',
         'faithfulness':'PROMPT_IS_RELEVANT_chunk_answer',
         'answer_relevance':'PROMPT_RELEVANCY_SCORE'}
META=('item_id','source_file','source_row','generator','embedder','domain','experiment')
PARSER_VERSION='explicit_declared_score_v2_with_legacy_first_number_v1'
NUMBER=r'(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)'
END=r'(?=[ \t]*(?:[.,](?:\s|$)|\(|$))'
RULES={
 'explicit_rating': re.compile(r'\bI (?:would )?(?:give (?:this|the) (?:generated answer|retrieved context|answer) a score of|(?:rate|score) (?:this|the) (?:generated answer|retrieved context|answer) (?:as|a))[ \t]+('+NUMBER+r')'+END,re.I),
 'score_label': re.compile(r'\b(?:The score is[ \t]*:?|Score[ \t]*:)[ \t]*('+NUMBER+r')'+END,re.I),
 'grounding_label': re.compile(r'^The generated answer is[ \t]+('+NUMBER+r')[ \t]*\((?:Moderately|Partially|Fully|Weakly) grounded\)',re.I)}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def parse_explicit(text):
    if not isinstance(text,str):return None,[]
    clean=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    if '<think>' in clean or '</think>' in clean:return None,[]
    evidence=[]; value=core.parse_score(text)
    if value is not None:evidence.append({'rule':'leading_scalar','value':value,'text':clean.splitlines()[0]})
    for name,pattern in RULES.items():
        for m in pattern.finditer(clean):evidence.append({'rule':name,'value':float(m[1]),'text':m[0]})
    values={e['value'] for e in evidence}
    if len(values)!=1 or not all(math.isfinite(v) and 0<=v<=1 for v in values):return None,evidence
    return next(iter(values)),evidence

def parse_legacy(text):
    # Deliberately mirrors Agent.score_numeric + None-to-zero in supplied source.
    m=re.search(r'\d*\.?\d+',text or '')
    if not m:return 0.0,'no_number_zero'
    value=float(m[0])
    return max(0.,min(1.,value)),'first_number_clamped' if value>1 else 'first_number'

def load_inputs(path):
    items,files=core.read_archive(path)
    mapping={f['source_file']:f for f in files}
    for i in items:
        m=mapping[i['source_file']];raw=i['original_fields']
        # Original pandas loop passes question and answer strings WITHOUT strip().
        i['question']=raw[m['question_column']];i['answer']=raw[m['answer_column']]
        i['chunks']=[c.strip() for c in raw[m['context_column']].split('|||')]
        if not i['chunks'] or any(not c for c in i['chunks']):
            raise ValueError(f'Empty chunk in {i["source_file"]} row {i["source_row"]}; review input rather than score a placeholder.')
        i['n_chunks']=len(i['chunks'])
        # item_id stays bound to the original raw question/answer/context and row.
    return items,files

def select_pilot(items,seed=20261118,per_file=2):
    by=defaultdict(list)
    for i in items:by[i['source_file']].append(i)
    selected=set()
    for name,rows in sorted(by.items()):
        local_seed=int(core.digest([seed,name]),16)
        for i in random.Random(local_seed).sample(rows,min(per_file,len(rows))):selected.add(i['item_id'])
    return [i for i in items if i['item_id'] in selected]

def chunk_indices(item,metric):return [0] if metric=='answer_relevance' else range(1,item['n_chunks']+1)
def make_prompt(protocol,item,metric,chunk_index):
    inner=protocol[METRICS[metric]].format(question=item['question'],answer=item['answer'],
        chunk='' if chunk_index==0 else item['chunks'][chunk_index-1])
    return protocol['outer_prompt_template'].format(question='Scoring request',candidate='Candidate output',context='',rubric=inner)
def call_id(item,judge,metric,chunk_index):
    return core.digest(['collabeval_chunk_wrapped_v1',item['item_id'],judge,metric,chunk_index,1])
def all_jobs(items,judges):
    return {call_id(i,j,m,k):{'item_id':i['item_id'],'judge':j,'metric':m,'chunk_index':k,'repetition':1}
        for i in items for j in judges for m in METRICS for k in chunk_indices(i,m)}

@contextmanager
def output_lock(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.collector.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another collector is writing this output directory.')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

class Journal:
    """First completed result for legacy; first valid result for scientific scores.

    An explicit retry may fill failed validated scores. It never replaces the
    original legacy result. Unanswered requests remain visible on resume.
    """
    def __init__(self,path,run_id,expected):
        self.path=Path(path);self.run_id=run_id;self.expected=expected
        self.lock=threading.RLock();self.first={};self.latest={};self.success={}
        self.attempts={};self.requests={};self.results=set()
        if self.path.exists():
            with self.path.open('rb') as f:
                for line in f:
                    if not line.endswith(b'\n'):raise ValueError('Journal ends in a partial record; preserve it and repair before resuming.')
                    e=json.loads(line);self.accept(e,hashlib.sha256(line.rstrip(b'\r\n')).hexdigest())
    def accept(self,e,event_hash):
        key=e['job_id'];a=e['attempt'];identity=self.expected.get(key)
        if e['run_id']!=self.run_id or identity is None:raise ValueError('Journal job/run identity mismatch.')
        if any(e[k]!=v for k,v in identity.items()):raise ValueError('Journal identity fields changed.')
        pair=(key,a)
        if e['event']=='request':
            if pair in self.requests or key in self.success:raise ValueError('Duplicate request or request after valid acceptance.')
            if a!=self.attempts.get(key,0)+1:raise ValueError('Nonsequential journal attempt.')
            self.requests[pair]=e;self.attempts[key]=a
        elif e['event']=='result':
            if pair not in self.requests or pair in self.results:raise ValueError('Unmatched/duplicate result.')
            self.results.add(pair);e=dict(e,_event_sha256=event_hash)
            self.first.setdefault(key,e);self.latest[key]=e
            if e['status']=='ok':
                if key in self.success:raise ValueError('Duplicate valid result.')
                self.success[key]=e
        else:raise ValueError('Unknown journal event.')
    def append(self,event):
        with self.lock:
            e={'run_id':self.run_id,'time':core.now(),**event}
            line=json.dumps(e,ensure_ascii=False,allow_nan=False)
            with self.path.open('a',encoding='utf-8') as f:
                f.write(line+'\n');f.flush();os.fsync(f.fileno())
            self.accept(e,hashlib.sha256(line.encode()).hexdigest())
    def snapshot(self):
        with self.lock:return dict(first=dict(self.first),latest=dict(self.latest),success=dict(self.success),attempts=dict(self.attempts))

def api(*args,**kwargs):return core.api(*args,**kwargs)
def execute_call(journal,item,judge,model,metric,index,protocol,options,host,timeout):
    key=call_id(item,judge,metric,index)
    if key in journal.success:return
    attempt=journal.attempts.get(key,0)+1
    common={'job_id':key,**journal.expected[key],'model_digest':model['digest'],'attempt':attempt}
    payload={'model':model['name'],'prompt':make_prompt(protocol,item,metric,index),
             'stream':False,'keep_alive':'5m','options':options}
    journal.append({**common,'event':'request','payload':payload})
    try:raw=api(host,'generate',payload,timeout)
    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as e:
        journal.append({**common,'event':'result','status':'transport_error','score':None,
            'legacy_score':0.,'legacy_parse_status':'transport_error_zero','error':str(e)})
        return
    if not isinstance(raw,dict):raw={'invalid_api_response':raw}
    raw_text=raw.get('response')
    legacy,legacy_status=parse_legacy(raw_text if isinstance(raw_text,str) else '')
    parsed,evidence=parse_explicit(raw.get('response'))
    issues=[]
    if parsed is None:issues.append('invalid_score')
    if not raw.get('done') or raw.get('done_reason')=='length':issues.append('incomplete_generation')
    tokens=raw.get('prompt_eval_count')
    # This is a warning-boundary check, not a proof that all inputs were retained.
    if not isinstance(tokens,int) or isinstance(tokens,bool) or tokens<=0:issues.append('missing_prompt_token_evidence')
    elif tokens>=options['num_ctx']-32:issues.append('context_boundary_warning')
    try:
        running=api(host,'ps').get('models',[])
        matches=[m for m in running if m.get('digest')==model['digest']]
        raw['collector_loaded_model_check']=matches
        if not matches or max(m.get('context_length',0) for m in matches)<options['num_ctx']:issues.append('loaded_model_check_failed')
    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as e:
        raw['collector_loaded_model_check_error']=str(e);issues.append('loaded_model_check_failed')
    status='ok' if not issues else issues[-1]
    journal.append({**common,'event':'result','status':status,'validation_issues':issues,
        'score':parsed if not issues else None,'parsed_score':parsed,'parse_evidence':evidence,
        'legacy_score':legacy,'legacy_parse_status':legacy_status,'raw_response':raw})

def compatibility_mean(per_chunk):
    if not per_chunk or not all(all(v is not None for v in p) for p in per_chunk):return None
    # Match the reference's sequential accumulation of formatted committee scores.
    total=0.0
    for pair in per_chunk:total+=float(f'{sum(pair)/2:.4f}')
    return total/len(per_chunk)

def export_results(out,items,judges,journal):
    snap=journal.snapshot();first,valid,latest=snap['first'],snap['success'],snap['latest']
    chunks=[];long=[];legacy_long=[];wide=[];legacy_wide=[];unresolved=[];comparisons=[]
    for item in items:
        meta={k:item[k] for k in META};row=dict(meta);oldrow=dict(meta)
        for metric in METRICS:
            indices=list(chunk_indices(item,metric));per={};oldper={};column_values={};old_values={};bad=0
            for judge in judges:
                vv=[];lv=[];hashes=[];ids=[];attempts=[]
                for index in indices:
                    key=call_id(item,judge,metric,index);ids.append(key)
                    good=valid.get(key);initial=first.get(key);last=latest.get(key)
                    value=good['score'] if good else None;legacy=initial['legacy_score'] if initial else None
                    vv.append(value);lv.append(legacy)
                    if good:hashes.append(good['_event_sha256']);attempts.append(good['attempt'])
                    if initial and initial['status']!='ok':bad+=1
                    chunks.append({**meta,'judge':judge,'metric':metric,'chunk_index':index,'job_id':key,
                        'score':value,'status':good['status'] if good else last['status'] if last else 'pending',
                        'validated_attempt':good['attempt'] if good else None,
                        'result_sha256':good['_event_sha256'] if good else None,
                        'legacy_score':legacy,'legacy_first_status':initial['status'] if initial else 'pending',
                        'legacy_first_result_sha256':initial['_event_sha256'] if initial else None,
                        'legacy_parse_status':initial.get('legacy_parse_status') if initial else None})
                    if not good:unresolved.append({'job_id':key,**journal.expected[key],
                        'status':last['status'] if last else 'pending','attempts':snap['attempts'].get(key,0)})
                per[judge]=core.strict_mean(vv)
                oldper[judge]=sum(lv)/len(lv) if lv and all(v is not None for v in lv) else None
                column_values[judge]=vv;old_values[judge]=lv
                aggregate_id=core.digest(['collabeval_row_metric_v1',item['item_id'],judge,metric,1])
                long.append({**meta,'judge':judge,'metric':metric,'repetition':1,'score':per[judge],
                    'status':'ok' if per[judge] is not None else 'incomplete','job_id':aggregate_id,
                    'score_origin':'collabeval_chunk_recollection','n_expected_calls':len(indices),
                    'n_valid_calls':sum(v is not None for v in vv),'evidence_call_ids':json.dumps(ids),
                    'evidence_result_hashes':json.dumps(hashes),'evidence_attempts':json.dumps(attempts),
                    'legacy_score':oldper[judge]})
                row[f'{judge}__{metric}']=per[judge];oldrow[f'{judge}__{metric}']=oldper[judge]
                legacy_long.append({**meta,'judge':judge,'metric':metric,'score':oldper[judge],
                    'interpretation':'legacy_first_result_including_failure_zeros','n_expected_calls':len(indices),
                    'n_observed_calls':sum(v is not None for v in lv)})
            row[f'ensemble__{metric}']=core.strict_mean(list(per.values()))
            row[f'{metric}__complete']=row[f'ensemble__{metric}'] is not None
            pairs=[[old_values[j][k] for j in judges] for k in range(len(indices))]
            legacy=compatibility_mean(pairs)
            oldrow[f'ensemble__{metric}']=legacy
            oldrow[f'{metric}__observed']=legacy is not None
            oldrow[f'{metric}__first_result_failed_calls']=bad
            legacy_long.append({**meta,'judge':'mean','metric':metric,'score':legacy,
                'interpretation':'legacy_first_result_rounded_chunk_committee_then_average',
                'n_expected_calls':len(indices)*len(judges),
                'n_observed_calls':sum(v is not None for pair in pairs for v in pair)})
            historical_column={'faithfulness':'context_relevance_Answer_dual',
                'context_relevance':'context_relevance_dual','answer_relevance':'response_relevancy_llm'}[metric]
            raw_historical=item['original_fields'].get(historical_column)
            if raw_historical not in (None,''):
                try:historical=float(raw_historical)
                except ValueError:historical=None
                if historical is not None and math.isfinite(historical):
                    scientific=row[f'ensemble__{metric}']
                    comparisons.append({**meta,'metric':metric,'archived_score':historical,
                        'legacy_score':legacy,'validated_score':scientific,
                        'legacy_minus_archived':legacy-historical if legacy is not None else None,
                        'validated_minus_archived':scientific-historical if scientific is not None else None})
        wide.append(row);legacy_wide.append(oldrow)
    for name,data in [('call_scores.csv',chunks),('per_judge_scores.csv',long),('legacy_scores_long.csv',legacy_long),
        ('scores_wide.csv',wide),('legacy_scores_wide.csv',legacy_wide),('historical_comparison.csv',comparisons)]:
        core.write_csv(out/name,data)
    summary=[]
    for kind,table in [('validated',wide),('legacy',legacy_wide)]:
        groups=defaultdict(list)
        for r in table:
            for metric in METRICS:
                for judge in [*judges,'ensemble']:
                    key=tuple(r[k] for k in ('generator','embedder','domain','experiment'))+(metric,judge)
                    groups[key].append(r[f'{judge}__{metric}'])
        for key,values in sorted(groups.items()):
            observed=[v for v in values if v is not None]
            summary.append(dict(zip(('generator','embedder','domain','experiment','metric','judge'),key),
                score_kind=kind,n_expected=len(values),n_available=len(observed),
                mean=st.mean(observed) if observed else None,sample_sd=st.stdev(observed) if len(observed)>1 else None))
    core.write_csv(out/'per_judge_summary.csv',summary)
    historical_summary=[]
    for metric in METRICS:
        subset=[r for r in comparisons if r['metric']==metric]
        for kind in ['legacy','validated']:
            matched=[r for r in subset if r[kind+'_score'] is not None]
            if matched:
                differences=[r[kind+'_minus_archived'] for r in matched]
                historical_summary.append({'metric':metric,'score_kind':kind,'n_archived_expected':len(subset),
                    'n_matched':len(matched),'archived_mean_on_matched_rows':st.mean(r['archived_score'] for r in matched),
                    'new_mean_on_matched_rows':st.mean(r[kind+'_score'] for r in matched),
                    'mean_signed_difference':st.mean(differences),'mean_absolute_difference':st.mean(abs(x) for x in differences),
                    'root_mean_squared_difference':math.sqrt(st.mean(x*x for x in differences))})
    core.write_csv(out/'historical_comparison_summary.csv',historical_summary)
    agreement=[]
    for experiment in sorted({r['experiment'] for r in wide}):
        for metric in METRICS:
            pairs=[(r[f'{judges[0]}__{metric}'],r[f'{judges[1]}__{metric}']) for r in wide if r['experiment']==experiment]
            differences=[x-y for x,y in pairs if x is not None and y is not None]
            if differences:agreement.append({'experiment':experiment,'metric':metric,'judge_1':judges[0],'judge_2':judges[1],
                'n_expected_pairs':len(pairs),'n_complete_pairs':len(differences),
                'mean_signed_difference':st.mean(differences),'mean_absolute_difference':st.mean(abs(x) for x in differences)})
    core.write_csv(out/'descriptive_agreement.csv',agreement)
    core.atomic_json(out/'remaining_jobs.json',unresolved)
    status={'expected_model_calls':len(journal.expected),'completed_model_calls':len(first),
        'valid_model_calls':len(valid),'pending_model_calls':len(journal.expected)-len(first),
        'failed_model_calls':len(first)-len(valid),'expected_metric_judgments':len(long),
        'valid_metric_judgments':sum(r['score'] is not None for r in long),
        'complete':len(valid)==len(journal.expected),'collection_complete':len(first)==len(journal.expected),
        'parser_disagreements_on_first_results':sum(e.get('score') is not None and e['score']!=e['legacy_score'] for e in first.values()),
        'updated_at':core.now()}
    core.atomic_json(out/'status.json',status)
    return status

def code_hashes():return {n:file_hash(HERE/n) for n in ['collect_judgments.py','collector_core.py']}
def verify_protocol(protocol):
    if protocol['version']!='collabeval_chunk_wrapped_v1' or protocol['weights']!=[.5,.5]:raise ValueError('Unsupported protocol version or weights.')
    if protocol['source_sha256']!=file_hash(HERE/'coauthor_reference.txt'):raise ValueError('Co-author source hash differs from extracted protocol.')

def scheduler(args,items,models,protocol,options,journal,progress):
    state={'submitted':0,'completed':0,'stop_reason':None};failures=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for judge,model in models.items():
            def needed(item,metric,k):
                key=call_id(item,judge,metric,k)
                return key in journal.latest and key not in journal.success if args.action=='retry' else key not in journal.first
            jobs=iter((i,m,k) for i in items for m in METRICS for k in chunk_indices(i,m) if needed(i,m,k))
            pending={}
            def fill():
                while len(pending)<args.workers and state['stop_reason'] is None:
                    if args.max_calls and state['submitted']>=args.max_calls:state['stop_reason']='max_calls';return
                    job=next(jobs,None)
                    if job is None:return
                    item,m,k=job
                    f=pool.submit(execute_call,journal,item,judge,model,m,k,protocol,options,args.host,args.timeout)
                    pending[f]=job;state['submitted']+=1
            fill()
            while pending:
                try:done,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    state['stop_reason']='interrupted';print('Stopping submissions; logging active calls before exit.',flush=True);continue
                for f in done:
                    item,m,k=pending.pop(f);f.result();state['completed']+=1
                    good=call_id(item,judge,m,k) in journal.success
                    failures=0 if good else failures+1
                    if failures>=args.stop_after_failures:state['stop_reason']='consecutive_failures'
                    print(f'{len(journal.first)}/{len(journal.expected)} calls recorded; {len(journal.success)} valid; '
                        f'{judge} {m} chunk={k} {item["source_file"]} row={item["source_row"]}',flush=True)
                    progress()
                fill()
            if state['stop_reason']:break
    return state

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','run','export','retry'])
    p.add_argument('--zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    p.add_argument('--baseline',type=Path,default=HERE/'baseline_models.json')
    p.add_argument('--allow-baseline-change',action='store_true',help='Explicit new experiment if installed model/version differs from B200 baseline.')
    p.add_argument('--host',default='http://127.0.0.1:11434')
    p.add_argument('--pilot',action='store_true',help='Two random rows per input file, fixed selection seed.')
    p.add_argument('--selection-seed',type=int,default=20261118)
    p.add_argument('--num-ctx',type=int,default=8192)
    p.add_argument('--num-predict',type=int,default=None,help='Omitted by default, matching supplied source. Override defines a new run.')
    p.add_argument('--seed',type=int,default=None,help='Generation seed omitted by default; separate from pilot selection seed.')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--max-calls',type=int);p.add_argument('--stop-after-failures',type=int,default=5)
    a=p.parse_args(argv)
    if min(a.workers,a.num_ctx,a.stop_after_failures)<1 or a.timeout<=0 or (a.max_calls is not None and a.max_calls<1):p.error('Counts and timeout must be positive.')
    if a.num_predict is not None and a.num_predict<1:p.error('--num-predict must be positive or omitted.')
    protocol=json.loads(a.protocol.read_text());verify_protocol(protocol)
    items,files=load_inputs(a.zip);selected=select_pilot(items,a.selection_seed) if a.pilot else items
    judges=list(protocol['JUDGES']);expected=all_jobs(selected,judges)
    audit={'archive_rows':len(items),'csv_files':len(files),'selected_rows':len(selected),
        'expected_model_calls':len(expected),'expected_metric_judgments':len(selected)*6,
        'zip_sha256':file_hash(a.zip),'protocol_sha256':core.digest(protocol),
        'pilot':a.pilot,'selection_seed':a.selection_seed if a.pilot else None,
        'chunk_counts':{str(k):v for k,v in Counter(i['n_chunks'] for i in selected).items()},
        'historical_scored_selected_rows':sum('context_relevance_dual' in i['original_fields'] for i in selected)}
    with output_lock(a.out):
        if a.action=='audit':
            core.atomic_json(a.out/'audit.json',audit)
            core.write_csv(a.out/'input_inventory.csv',files)
            core.write_csv(a.out/'selected_items.csv',[{k:i[k] for k in META} for i in selected])
            print(json.dumps(audit,indent=2));return 0
        saved_path=a.out/'run_config.json'
        if a.action in ['export','retry'] and not saved_path.exists():raise ValueError(f'{a.action} requires an existing run.')
        if a.action=='export':
            saved=json.loads(saved_path.read_text());cfg=saved['config']
            for k in ('zip_sha256','protocol_sha256'):
                if cfg[k]!=audit[k]:raise ValueError('Export input/protocol mismatch.')
            if cfg['code_sha256']!=code_hashes():raise ValueError('Collector code changed; use the original package to export.')
            if cfg['item_ids']!=[i['item_id'] for i in selected]:raise ValueError('Export selection differs; include original --pilot/--selection-seed.')
            if core.digest(cfg)!=saved['run_id']:raise ValueError('Run config hash is inconsistent.')
            for job in expected.values():job['model_digest']=cfg['models'][job['judge']]['digest']
            journal=Journal(a.out/'calls.jsonl',saved['run_id'],expected)
            print(json.dumps(export_results(a.out,selected,judges,journal),indent=2));return 0
        models,missing=core.resolve_models(a.host,protocol['JUDGES'])
        if missing:raise ValueError('Required local models missing: '+', '.join(missing)+'. Restore the pinned versions; no automatic pull was performed.')
        if len({m['digest'] for m in models.values()})!=2:raise ValueError('Judges must resolve to different checkpoints.')
        for model in models.values():
            if a.num_ctx>model['max_context']:raise ValueError('Requested context exceeds model capability.')
        version=api(a.host,'version');baseline=json.loads(a.baseline.read_text())
        differences=[]
        if version!=baseline['ollama_version']:differences.append('Ollama version')
        for j,m in models.items():
            if m['digest']!=baseline['models'][j]['digest']:differences.append(j+' model digest')
        if differences and not a.allow_baseline_change:raise ValueError('B200 baseline mismatch: '+', '.join(differences)+'. Use --allow-baseline-change only for an explicitly different experiment.')
        options={'num_ctx':a.num_ctx,'temperature':0.0}
        if a.seed is not None:options['seed']=a.seed
        if a.num_predict is not None:options['num_predict']=a.num_predict
        cfg={**audit,'code_sha256':code_hashes(),'item_ids':[i['item_id'] for i in selected],
            'models':models,'ollama_version':version,'options':options,'host':a.host,'workers':a.workers,
            'keep_alive':'5m','parser':PARSER_VERSION,'repeats':1,'context_mode':'per_chunk',
            'aggregation':'validated complete equal means; legacy first result, rounded chunk committee means',
            'baseline_sha256':file_hash(a.baseline),'baseline_differences':differences,
            'allow_baseline_change':a.allow_baseline_change,'retry_policy':'explicit retry only; first valid score accepted, first-result legacy frozen',
            'transport':'direct Ollama generate API, stream=false; model-specific defaults not overridden',
            'scheduler':'bounded_threads_judge_blocks_v1'}
        if saved_path.exists():
            prior=json.loads(saved_path.read_text())
            if core.digest(prior['config'])!=prior['run_id']:raise ValueError('Saved configuration hash is inconsistent.')
        run_id=core.bind_run(a.out,cfg)
        for job in expected.values():job['model_digest']=models[job['judge']]['digest']
        journal=Journal(a.out/'calls.jsonl',run_id,expected)
        item_text=''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in selected)
        items_path=a.out/'items.jsonl'
        if items_path.exists() and items_path.read_text()!=item_text:raise ValueError('Saved item records changed.')
        if not items_path.exists():items_path.write_text(item_text)
        core.write_csv(a.out/'input_inventory.csv',files)
        core.write_csv(a.out/'selected_items.csv',[{k:i[k] for k in META} for i in selected])
        profile={'time':core.now(),'launcher_mode':os.environ.get('COLLABEVAL_SERVER_MODE','unspecified'),
            'client_environment_declarations':{k:os.environ.get(k) for k in ['OLLAMA_NUM_PARALLEL','OLLAMA_FLASH_ATTENTION','OLLAMA_KV_CACHE_TYPE','OLLAMA_MAX_LOADED_MODELS','OLLAMA_MODELS']},
            'note':'Existing-server settings cannot be verified from client environment; declarations are not measurements.'}
        core.atomic_json(a.out/'latest_execution_profile.json',profile)
        started=time.monotonic();last_export=0.;last_count=-1
        def progress(force=False):
            nonlocal last_export,last_count
            n=len(journal.latest)
            if force or n-last_count>=100 or time.monotonic()-last_export>=60:
                status=export_results(a.out,selected,judges,journal);last_export=time.monotonic();last_count=n
                return status
            core.atomic_json(a.out/'progress.json',{'recorded_calls':len(journal.first),'valid_calls':len(journal.success),
                'expected_calls':len(expected),'updated_at':core.now()})
        progress(True);state={}
        try:state=scheduler(a,selected,models,protocol,options,journal,progress)
        finally:
            status=progress(True)
            core.atomic_json(a.out/'performance.json',{'wall_seconds':time.monotonic()-started,'action':a.action,**state})
            print(json.dumps(status),flush=True)
        if state.get('stop_reason')=='interrupted':return 130
        if state.get('stop_reason')=='max_calls':return 0
        return 0 if status['complete'] else 3

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as e:
        print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)
