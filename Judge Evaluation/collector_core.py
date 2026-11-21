#!/usr/bin/env python3
"""Auditable, resumable whole-context reevaluation using the co-author's prompts.

Python 3.10+ standard library only. See RUN_GUIDE.txt. No model downloads,
cloud fallback, source edits, or generation of replacement answers occur here.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import threading
import time
from types import SimpleNamespace
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import zipfile

METRICS = {
    'faithfulness': 'PROMPT_CTX_ANSWER_ALL',
    'context_relevance': 'PROMPT_CTX_QUESTION_ALL',
    'answer_relevance': 'PROMPT_ANSWER_QUESTION',
}
ALIASES = {
    'question': ['question', 'query', 'prompt', 'questions'],
    'answer': ['answer', 'response', 'answers', 'generated_answer'],
    'context': ['top_k_docs', 'retrieved_docs', 'retrieved_context',
                'retrieved_contexts', 'contexts', 'context', 'docs'],
}

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def write_csv(path, records):
    records = list(records)
    if not records:
        return
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, path)

def parse_score(text):
    """Accept a scalar or an explicit leading score declaration, never any number."""
    if not isinstance(text, str):
        return None
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    number = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
    if re.fullmatch(number, clean):
        numeric = clean
    else:
        first_line = clean.splitlines()[0] if clean else ''
        # A score declaration must occupy the first line, apart from a textual
        # parenthesized rubric label. Later explanatory examples are not scores.
        match = re.fullmatch(
            r'(?:I (?:would )?score\b|(?:Score|Rating)\s*:)'
            r'(?P<prefix>[^\d+\-\n]{0,120}?)(?P<number>' + number + r')'
            r'(?:\s*\([^\d\n]*\))?\.?\s*', first_line, re.I)
        if not match or re.search(r'\b(from|between|scale|not|never)\b', match['prefix'], re.I):
            return None
        numeric = match['number']
    value = float(numeric)
    return value if math.isfinite(value) and 0 <= value <= 1 else None

def strict_mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None

def read_archive(path):
    """Read only CSV members of this ZIP, without extracting or skipping bad rows."""
    csv.field_size_limit(10_000_000)
    items, files = [], []
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.csv')
                       and not any(p.startswith('.') or p == '__MACOSX' for p in PurePosixPath(n).parts))
        if not names or len(names) != len(set(names)):
            raise ValueError('Empty archive or duplicate CSV member names')
        common = os.path.commonpath(names)
        if len(names) == 1:
            common = str(PurePosixPath(names[0]).parent)
        for name in names:
            rel = str(PurePosixPath(name).relative_to(common))
            parts = PurePosixPath(rel).parts
            if len(parts) < 4:
                raise ValueError(f'Cannot infer experiment metadata from {rel}')
            combo = re.sub(r'\s+', '', parts[0]).lower()
            gen, emb = combo.split('-', 1)
            domain = parts[1].lower()
            variation = parts[2].lower()
            experiment = 'prompt' if 'prompt' in variation else 'retrieval' if 'context' in variation else ''
            if domain not in ('social', 'ecological') or not experiment:
                raise ValueError(f'Ambiguous metadata in {rel}')
            reader = csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')), strict=True)
            headers = reader.fieldnames or []
            normalized = [re.sub(r'\s+', '_', h.strip().lower()) for h in headers]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f'Duplicate column names in {rel}')
            mapping = {}
            for key, candidates in ALIASES.items():
                hits = [headers[normalized.index(c)] for c in candidates if c in normalized]
                if len(hits) != 1:
                    raise ValueError(f'{rel}: expected one {key} column, found {hits}')
                mapping[key] = hits[0]
            n = 0
            for row_number, row in enumerate(reader, 1):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f'{rel} record {row_number}: malformed CSV row')
                raw = {k: row[v] for k, v in mapping.items()}
                if any(not v.strip() for v in raw.values()):
                    raise ValueError(f'{rel} record {row_number}: missing evaluation input')
                # Exactly the normalization in the co-author's default all-context path.
                chunks = [c.strip() for c in raw['context'].strip().split('|||') if c.strip()]
                content_hash = digest(raw)
                items.append({
                    'item_id': digest([rel, row_number, content_hash]),
                    'source_file': rel, 'source_row': row_number,
                    'generator': gen, 'embedder': emb, 'domain': domain,
                    'experiment': experiment, 'batch_id': row.get('batch_id', ''),
                    'batch_files': row.get('batch_files', ''), 'input_sha256': content_hash,
                    'question': raw['question'].strip(), 'answer': raw['answer'].strip(),
                    'context': '\n\n'.join(chunks), 'n_chunks': len(chunks),
                    'original_fields': row,
                })
                n += 1
            files.append({'source_file': rel, 'generator': gen, 'embedder': emb,
                          'domain': domain, 'experiment': experiment, 'rows': n,
                          **{f'{k}_column': v for k, v in mapping.items()}})
    return items, files

def make_prompt(protocol, item, metric):
    return protocol[METRICS[metric]].format(question=item['question'], answer=item['answer'], contexts=item['context'])

def pilot_items(items):
    """One real item per column schema, plus longest inputs for all three metrics."""
    selected, schemas = {}, set()
    for item in items:
        schema = tuple(item['original_fields'])
        if schema not in schemas:
            selected[item['item_id']] = item
            schemas.add(schema)
    for fields in [('context', 'answer'), ('context', 'question'), ('answer', 'question')]:
        item = max(items, key=lambda i: sum(len(i[f].encode()) for f in fields))
        selected[item['item_id']] = item
    return list(selected.values())

def api(base, endpoint, payload=None, timeout=30):
    req = Request(base.rstrip('/') + '/api/' + endpoint,
                  data=None if payload is None else json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)

def resolve_models(base, judges):
    tags = api(base, 'tags')['models']
    lookup = {m['name']: m for m in tags}
    resolved, missing = {}, []
    for label, requested in judges.items():
        name = requested if ':' in requested else requested + ':latest'
        if name not in lookup:
            missing.append(name)
            continue
        entry = lookup[name]
        if not entry.get('digest') or not entry.get('details', {}).get('format'):
            raise ValueError(f'{name}: a pinned local model is required; cloud entries are not accepted')
        show = api(base, 'show', {'model': name})
        lengths = [int(v) for k, v in show.get('model_info', {}).items() if k.endswith('.context_length')]
        if not lengths:
            raise ValueError(f'{name}: cannot determine supported context length')
        resolved[label] = {'requested': requested, 'name': name, 'digest': entry['digest'],
                           'details': entry.get('details'), 'show': show, 'max_context': min(lengths)}
    return resolved, missing

def context_budget(prompt, model, minimum, output_limit):
    """Conservative byte-level token upper bound for the two local BPE models.

    Includes the full model template/system text and 1,024 wrapper tokens.
    It intentionally overallocates compared with a character/token heuristic.
    """
    overhead = sum(len(str(model['show'].get(k, '')).encode()) for k in ('template', 'system')) + 1024
    bound = len(prompt.encode()) + overhead + output_limit
    capacity = minimum
    while capacity < bound:
        capacity *= 2
    if capacity > model['max_context']:
        raise ValueError(f'Conservative input/output bound requires {capacity} tokens; model supports {model["max_context"]}')
    return capacity, bound

def bind_run(out, config):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'run_config.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['config'] != config:
            raise ValueError('Run configuration/input/model changed. Use a new output directory; previous results were not reused.')
        return old['run_id']
    run_id = digest(config)
    atomic_json(path, {'run_id': run_id, 'created_at': now(), 'config': config})
    return run_id

class Journal:
    def __init__(self, path, run_id):
        self.lock = threading.RLock()
        self.path, self.run_id = Path(path), run_id
        self.success, self.attempts, self.latest = {}, {}, {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                event = json.loads(line)  # Stop on damaged logs; never silently discard data.
                if event['run_id'] != run_id:
                    raise ValueError('Journal contains another run')
                self.accept(event)

    def accept(self, event):
        key = event['job_id']
        self.attempts[key] = max(self.attempts.get(key, 0), event['attempt'])
        if event['event'] == 'result':
            self.latest[key] = event
            if event['status'] == 'ok':
                if key in self.success:
                    raise ValueError(f'Duplicate accepted judgment: {key}')
                self.success[key] = event

    def append(self, event):
        event = {'run_id': self.run_id, 'time': now(), **event}
        # Workers share one serialized, durable journal writer. Each job is
        # submitted only once; retries for a job stay within its worker.
        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            self.accept(event)

    def snapshot(self):
        # Export a consistent snapshot without blocking workers during CSV I/O.
        with self.lock:
            return SimpleNamespace(success=dict(self.success), latest=dict(self.latest))

def job_id(item, label, metric, repeat):
    return digest([item['item_id'], label, metric, repeat])

def execute_job(journal, item, label, model, metric, repeat, prompt, options,
                base, timeout, transport=api):
    key = job_id(item, label, metric, repeat)
    if key in journal.success:
        return False
    identity = {'job_id': key, 'item_id': item['item_id'], 'judge': label,
                'metric': metric, 'repetition': repeat, 'model_digest': model['digest']}
    payload = {'model': model['name'], 'prompt': prompt, 'stream': False,
               'keep_alive': '30m', 'options': options}
    for network_try in range(3):
        attempt = journal.attempts.get(key, 0) + 1
        common = {**identity, 'attempt': attempt}
        journal.append({**common, 'event': 'request', 'payload': payload})
        try:
            response = transport(base, 'generate', payload, timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            journal.append({**common, 'event': 'result', 'status': 'transport_error',
                            'score': None, 'error': str(e)})
            if network_try < 2:
                time.sleep(2 * (network_try + 1))
                continue
            return True
        score = parse_score(response.get('response'))
        status = 'ok' if score is not None else 'invalid_score'
        if not response.get('done') or response.get('done_reason') == 'length':
            status, score = 'incomplete_generation', None
        prompt_tokens = response.get('prompt_eval_count')
        if not isinstance(prompt_tokens, int) or prompt_tokens >= options['num_ctx'] - 32:
            status, score = 'context_check_failed', None
        if transport is api:
            # Check the server honored the context allocation and still served the
            # pinned local model. Keep failed verification in the same raw event.
            try:
                running = api(base, 'ps').get('models', [])
                matches = [m for m in running if m.get('digest') == model['digest']]
                response['collector_loaded_model_check'] = matches
                if not matches or max(m.get('context_length', 0) for m in matches) < options['num_ctx']:
                    status, score = 'loaded_model_check_failed', None
            except (HTTPError, URLError, TimeoutError, OSError) as e:
                response['collector_loaded_model_check_error'] = str(e)
                status, score = 'loaded_model_check_failed', None
        journal.append({**common, 'event': 'result', 'status': status,
                        'score': score, 'raw_response': response})
        return True
    return True

def export_results(out, items, labels, repeats, journal):
    wide, long, groups, gaps = [], [], {}, {}
    for item in items:
        meta = {k: item[k] for k in ('item_id', 'source_file', 'source_row', 'generator', 'embedder', 'domain', 'experiment')}
        row = dict(meta)
        for metric in METRICS:
            per = {}
            for label in labels:
                values = []
                for rep in range(1, repeats + 1):
                    key = job_id(item, label, metric, rep)
                    success = journal.success.get(key)
                    value = success['score'] if success else None
                    status = 'ok' if success else journal.latest.get(key, {}).get('status', 'pending')
                    long.append({**meta, 'judge': label, 'metric': metric,
                                 'repetition': rep, 'score': value, 'status': status, 'job_id': key})
                    values.append(value)
                per[label] = strict_mean(values)
                row[f'{label}__{metric}'] = per[label]
            aggregate = strict_mean(list(per.values()))
            row[f'ensemble__{metric}'] = aggregate
            row[f'{metric}__complete'] = aggregate is not None
            for label, value in {**per, 'ensemble': aggregate}.items():
                group = (item['generator'], item['embedder'], item['domain'], item['experiment'], metric, label)
                groups.setdefault(group, []).append(value)
            if len(labels) == 2 and aggregate is not None:
                difference = per[labels[0]] - per[labels[1]]
                gaps.setdefault((item['experiment'], metric), []).append(difference)
        wide.append(row)
    summary = []
    for key, values in sorted(groups.items()):
        valid = [v for v in values if v is not None]
        summary.append(dict(zip(('generator', 'embedder', 'domain', 'experiment', 'metric', 'judge'), key),
                            n_expected=len(values), n_valid=len(valid),
                            mean=statistics.mean(valid) if valid else None,
                            sample_sd=statistics.stdev(valid) if len(valid) > 1 else None))
    agreement = [{'experiment': k[0], 'metric': k[1], 'judge_1': labels[0], 'judge_2': labels[1],
                  'n_pairs': len(v), 'mean_signed_difference': statistics.mean(v),
                  'mean_absolute_difference': statistics.mean(abs(x) for x in v)} for k, v in sorted(gaps.items())]
    write_csv(out / 'per_judge_scores.csv', long)
    write_csv(out / 'scores_wide.csv', wide)
    write_csv(out / 'per_judge_summary.csv', summary)
    write_csv(out / 'descriptive_agreement.csv', agreement)
    expected = len(items) * len(labels) * len(METRICS) * repeats
    status = {'expected_metric_judgments': expected, 'valid_metric_judgments': len(journal.success),
              'failed_or_pending': expected - len(journal.success), 'complete': len(journal.success) == expected,
              'updated_at': now()}
    atomic_json(out / 'status.json', status)
    return status

def run_parallel(args, selected, models, protocol, capacities, journal, progress, state):
    """Bound in-flight work to workers; finish one judge before starting the next.

    On three consecutive failed completions, stop submitting and drain all
    in-flight calls into the journal. Ctrl+C similarly drains active work.
    """
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for label, model in models.items():
            if args.only_judge and label != args.only_judge:
                continue
            jobs = iter((item, metric, rep)
                        for item in selected for metric in METRICS
                        for rep in range(1, args.repeats + 1)
                        if job_id(item, label, metric, rep) not in journal.success)
            pending = {}

            def fill():
                while len(pending) < args.workers and state['stop_reason'] is None:
                    if args.max_calls is not None and state['submitted'] >= args.max_calls:
                        state['stop_reason'] = 'max_calls'
                        return
                    job = next(jobs, None)
                    if job is None:
                        return
                    item, metric, rep = job
                    options = {'num_ctx': capacities[label], 'num_predict': args.num_predict,
                               'temperature': args.temperature, 'seed': args.seed + rep - 1}
                    future = pool.submit(execute_job, journal, item, label, model, metric, rep,
                                         make_prompt(protocol, item, metric), options,
                                         args.host, args.timeout, api)
                    pending[future] = job
                    state['submitted'] += 1

            fill()
            while pending:
                try:
                    done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    state['stop_reason'] = 'interrupted'
                    print('Stopping new submissions; waiting for active judgments to be logged.', flush=True)
                    continue
                for future in done:
                    item, metric, rep = pending.pop(future)
                    future.result()
                    state['completed'] += 1
                    valid = job_id(item, label, metric, rep) in journal.success
                    consecutive_failures = 0 if valid else consecutive_failures + 1
                    if consecutive_failures >= 3:
                        state['stop_reason'] = 'consecutive_failures'
                    total = len(selected) * len(models) * len(METRICS) * args.repeats
                    print(f'{len(journal.success)}/{total} valid; {label} {metric} '
                          f'{item["source_file"]} row={item["source_row"]} '
                          f'ctx={capacities[label]} workers={args.workers}', flush=True)
                    progress()
                fill()
            if state['stop_reason'] is not None:
                break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'run', 'export'])
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('protocol.json'))
    parser.add_argument('--host', default='http://127.0.0.1:11434')
    parser.add_argument('--judge', action='append', help='label=installed_model; repeat for both judges')
    parser.add_argument('--pilot', action='store_true', help='Select all schemas and longest inputs; use a separate directory')
    parser.add_argument('--only-judge', help='Run one configured judge; ensemble remains incomplete until both finish')
    parser.add_argument('--max-calls', type=int, help='Stop after this many previously unfinished metric jobs')
    parser.add_argument('--num-ctx', type=int, default=16384, help='Minimum; increased conservatively per input')
    parser.add_argument('--num-predict', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent requests to one judge; match OLLAMA_NUM_PARALLEL')
    parser.add_argument('--export-every', type=int, default=100,
                        help='Completed jobs between CSV snapshots; raw logs remain immediate')
    parser.add_argument('--export-seconds', type=float, default=60,
                        help='Maximum interval between CSV snapshots while jobs complete')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.export_every < 1 or args.export_seconds <= 0:
        parser.error('Workers and export intervals must be positive')
    if min(args.num_ctx, args.num_predict, args.repeats) < 1 or (args.max_calls is not None and args.max_calls < 1):
        parser.error('Context, output limit, repeats and max-calls must be positive')
    protocol = json.loads(args.protocol.read_text())
    judges = dict(x.split('=', 1) for x in args.judge) if args.judge else protocol['JUDGES']
    if len(judges) != 2 or len(set(judges.values())) != 2:
        parser.error('This study requires two distinct configured judges')
    if args.only_judge and args.only_judge not in judges:
        parser.error('Unknown --only-judge')
    items, files = read_archive(args.zip)
    selected = pilot_items(items) if args.pilot else items
    args.out.mkdir(parents=True, exist_ok=True)
    audit = {'csv_files': len(files), 'archive_rows': len(items), 'selected_rows': len(selected),
             'chunk_counts': sorted(set(i['n_chunks'] for i in items)),
             'expected_model_calls': len(selected) * 3 * len(judges) * args.repeats,
             'zip_sha256': digest(args.zip.read_bytes()), 'protocol_sha256': digest(protocol)}
    if args.action == 'audit':
        atomic_json(args.out / 'audit.json', audit)
        write_csv(args.out / 'input_inventory.csv', files)
        print(json.dumps(audit, indent=2))
        return 0
    # Export is offline, and validates the inputs/protocol against the saved run.
    if args.action == 'export':
        saved = json.loads((args.out / 'run_config.json').read_text())
        for k in ('zip_sha256', 'protocol_sha256'):
            if saved['config'][k] != audit[k]:
                raise ValueError('Export inputs/protocol differ from the recorded run')
        if saved['config']['item_ids'] != [i['item_id'] for i in selected]:
            raise ValueError('Export selection differs; use the original --pilot flag')
        journal = Journal(args.out / 'calls.jsonl', saved['run_id'])
        status = export_results(args.out, selected, list(saved['config']['models']), saved['config']['repeats'], journal)
        print(json.dumps(status, indent=2))
        return 0
    models, missing = resolve_models(args.host, judges)
    atomic_json(args.out / 'preflight.json', {'time': now(), **audit, 'models': models, 'missing_models': missing})
    if missing:
        print('Missing required local models: ' + ', '.join(missing), file=sys.stderr)
        return 2
    if len({m['digest'] for m in models.values()}) != len(models):
        raise ValueError('Both judge names resolve to the same model digest')
    # Use a single adequate context allocation per judge to avoid reloading the
    # model for every transition between a long grounding and shorter relevance prompt.
    capacities = {label: max(context_budget(make_prompt(protocol, item, metric), model,
                                           args.num_ctx, args.num_predict)[0]
                             for item in selected for metric in METRICS)
                  for label, model in models.items()}
    config = {'runner_sha256': digest(Path(__file__).read_bytes()), **audit,
              'item_ids': [i['item_id'] for i in selected], 'models': models,
              'ollama_version': api(args.host, 'version'), 'host': args.host,
              'minimum_num_ctx': args.num_ctx, 'effective_context_by_judge': capacities,
              'context_strategy': 'utf8_byte_upper_bound_power_of_two_fixed_per_judge',
              'num_predict': args.num_predict, 'temperature': args.temperature,
              'seed': args.seed, 'repeats': args.repeats, 'context_mode': 'all',
              'parser': 'bare_scalar_or_explicit_leading_score_v1',
              'execution': {'scheduler': 'bounded_threads_judge_blocks_v1',
                            'workers': args.workers,
                            'server_environment_declared_by_client': {
                                key: os.environ.get(key) for key in (
                                    'OLLAMA_NUM_PARALLEL', 'OLLAMA_FLASH_ATTENTION',
                                    'OLLAMA_KV_CACHE_TYPE', 'OLLAMA_MAX_LOADED_MODELS',
                                    'OLLAMA_HOST', 'OLLAMA_MODELS', 'CUDA_VISIBLE_DEVICES')}},
              'aggregation': 'equal_weight_complete_two_judges_and_all_repeats'}
    run_id = bind_run(args.out, config)
    journal = Journal(args.out / 'calls.jsonl', run_id)
    if not (args.out / 'items.jsonl').exists():
        with (args.out / 'items.jsonl').open('w', encoding='utf-8') as f:
            for item in selected:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    write_csv(args.out / 'input_inventory.csv', files)
    total = config['expected_model_calls']
    started = time.monotonic()
    initial_valid = len(journal.success)
    state = {'submitted': 0, 'completed': 0, 'stop_reason': None}
    last_export = {'time': started, 'completed': 0}

    def progress(force=False):
        snapshot = journal.snapshot()
        if (force or state['completed'] - last_export['completed'] >= args.export_every
                or time.monotonic() - last_export['time'] >= args.export_seconds):
            status = export_results(args.out, selected, list(models), args.repeats, snapshot)
            last_export.update(time=time.monotonic(), completed=state['completed'])
        else:
            status = {'expected_metric_judgments': total,
                      'valid_metric_judgments': len(snapshot.success),
                      'failed_or_pending': total - len(snapshot.success),
                      'complete': len(snapshot.success) == total, 'updated_at': now()}
            atomic_json(args.out / 'status.json', status)
        return status

    progress(force=True)
    try:
        run_parallel(args, selected, models, protocol, capacities, journal, progress, state)
    except KeyboardInterrupt:
        state['stop_reason'] = 'interrupted'
    finally:
        status = progress(force=True)
        elapsed = time.monotonic() - started
        new_valid = len(journal.success) - initial_valid
        performance = {'time': now(), 'run_id': run_id, 'workers': args.workers,
                       'wall_seconds_this_invocation': elapsed,
                       'new_valid_metric_judgments': new_valid, **state,
                       'valid_judgments_per_minute': new_valid * 60 / elapsed if elapsed else None,
                       'note': 'Includes model loading and export; excludes preflight and downloads.'}
        atomic_json(args.out / 'performance.json', performance)
        with (args.out / 'performance_history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(performance) + '\n')
        print(json.dumps(status), flush=True)
        print(json.dumps(performance), flush=True)
    if state['stop_reason'] == 'interrupted':
        return 130
    if state['stop_reason'] == 'max_calls':
        return 0
    return 0 if status['complete'] else 3

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
