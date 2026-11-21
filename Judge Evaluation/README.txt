COLLABEVAL-COMPATIBLE B200 COLLECTOR
Prepared 18 November 2026

This is the NEW runnable collector for the chunk-based pipeline supplied as
collabeval_full_pipeline.py. It is separate from the earlier whole-context B200
collector. No live pilot results are included: only input audits and offline tests.

QUICK START ON B200

Copy judge_evaluation_collabeval_b200.zip to /path/to/project, then:

  cd /path/to/project
  unzip judge_evaluation_collabeval_b200.zip
  bash judge_evaluation_collabeval_b200/run_b200.sh pilot

Pilot: two responses from each of 40 files = 80 responses, 1,440 judge calls,
480 per-judge response-metric scores. It includes 24 responses with archived
aggregate scores. Selection seed 20261118 is independent of generation settings.
Models are NOT downloaded or updated. The installed model digests and Ollama
version must match baseline_models.json, copied from the completed B200 run.

Output directory:
  /path/to/project/b200_collabeval_pilot_w4
Evaluation log:
  /path/to/project/b200_collabeval_pilot_w4.log

Progress:
  tail -f /path/to/project/b200_collabeval_pilot_w4.log

Repeat the same command to resume UNATTEMPTED calls. Completed failures are not
silently regenerated. Review them before using the explicit retry command below.
The launcher reuses an existing server without changing it; otherwise it starts
one and stops only its own server when the script exits. No systemd is required.

After inspecting the pilot, a full run uses a distinct directory:

  bash judge_evaluation_collabeval_b200/run_b200.sh full

Full: 806 responses, 14,508 judge calls, 4,836 per-judge metric scores.
Full output: /path/to/project/b200_collabeval_full_w4
The pilot is not silently merged into a full run; each has its own recorded sample.

DEPENDENCIES AND PATHS

Collector: Python 3.10+ standard library only; Linux/macOS. No conda activation,
LangChain, pandas, NLTK or pip installation is required. The launcher defaults to
python3 and the archive at:
  /path/to/project/collect_judgments/Collection_of_Models_Answer_prompt_context.zip
It also recognizes the same filename with spaces. Override paths when necessary:

  PROJECT_DIR=/your/project PYTHON_BIN=/your/python \
    INPUT_ZIP=/your/input.zip OUTPUT_DIR=/your/new_output \
    bash judge_evaluation_collabeval_b200/run_b200.sh pilot

JUDGE_WORKERS defaults to 4. When the launcher starts Ollama, it requests four
parallel requests, Flash Attention and an f16 KV cache. Existing server settings
are not inferable from client environment declarations and remain unchanged.

EXACTLY WHAT IS MATCHED

- The two per-chunk rubrics and answer-relevance rubric were extracted from the
  supplied source, as was its complete outer wrapper with Scoring request,
  Candidate output and Rubric fields. Actual inputs remain inside the rubric.
- Original question and answer whitespace is preserved; each |||-separated chunk
  is stripped exactly as in the supplied parse_retrieved_docs function.
- Four context-relevance calls and four faithfulness calls per judge per row;
  one answer-relevance call per judge. Equal weights 0.5/0.5; one repetition.
- temperature=0, num_ctx=8192, keep_alive=5m. No automatic context expansion.
- Generation seed, num_predict, thinking mode, model system/template and other
  sampling options are omitted unless explicitly overridden. This follows the
  supplied script's unspecified defaults rather than adding the old B200 seed
  or output limit. Effective model metadata and server version are saved.
- The LEGACY ensemble follows the original first-number parser, clamping,
  failure-to-zero behavior, four-decimal committee formatting, and chunk mean.

INTENTIONAL EXECUTION / AUDIT DIFFERENCES

The collector uses Ollama's generate API directly with stream=false and saves
the complete returned response (including thinking when present). It does not
install the unpinned LangChain stack. No JSON format, extra system prompt or
different rubric is imposed. Streaming/client/server numerical equivalence to
an unknown historical environment is not guaranteed.

For B200 efficiency, requests run in bounded threads within one judge at a time,
avoiding continual model swaps. The original code calls the two judges in parallel
for each chunk. The scores are independent; there is no actual debate loop in
the supplied source. Scheduling and runtime can still affect exact repeatability.

The standalone smoke example and unused downloads/imports are omitted. CSV rows
are read strictly; malformed/missing inputs stop the run rather than disappearing.
Raw input identities are kept compatible with the previous B200 collector, with
an additional chunk index in call IDs. Changed code, inputs, protocol, options,
models, selection or concurrency cannot be resumed into an existing run.

MODEL BASELINE

Ollama: 0.40.16
deepseek-r1:latest
  6995872bfe4c521a67b32da386cd21d5c6e819b6e0d62f79f64ec83be99f5763
llama3.2:latest
  a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72

These pins reproduce the checkpoints of the completed B200 run; the historical
paper's precise checkpoints remain unknown. --allow-baseline-change is available
only to create an explicitly different experiment, and the mismatch is recorded.
Do not pull mutable latest tags merely to get past a mismatch.

TWO SEPARATE SCORE VIEWS

1. VALIDATED
   per_judge_scores.csv: two judge rows per response and metric, with a score only
   if all its required calls passed parsing and execution checks.
   scores_wide.csv: complete equal-weight judge averages; failures remain missing.
   These averages are unrounded. Their differences from legacy values include
   parser/failure changes and, at most 0.00005 from committee rounding alone.

2. LEGACY / HISTORICAL-COMPATIBILITY
   legacy_scores_long.csv: both judge averages plus a 'mean' row. The mean row
   is NOT a third judge. Failure zeros are retained to show the supplied code's
   behavior and must not be presented as validated model assessments.
   legacy_scores_wide.csv: original-style rounded chunk committee averages.
   *_first_result_failed_calls identifies how many constituent calls were not
   valid on their first completed attempt. The first-result legacy view stays
   frozen when failed calls are explicitly retried.

The two views are generated from the SAME saved model replies, not separate
generations. Choose the view by the scientific question, not by closeness to
the paper. Agreement or inference should disclose which view and missing-score
policy it uses. No Bayesian analysis is run automatically.

FILES TO RETURN FOR REVIEW

- run_config.json, items.jsonl, calls.jsonl: configuration, exact inputs and raw
  request/result journal. calls.jsonl retains each judge/chunk/metric separately.
- call_scores.csv: validated and legacy score, status, accepted attempt and
  raw-result hash for each individual call.
- per_judge_scores.csv, scores_wide.csv, legacy_scores_long.csv,
  legacy_scores_wide.csv and per_judge_summary.csv.
- historical_comparison.csv and historical_comparison_summary.csv: matched-row
  comparisons with historical aggregate scores embedded in the input files.
  The pilot compares its 24 historical-scored rows, not the paper's full-sample
  means. These are descriptive comparisons, not pooled Bayesian evidence.
- status.json, progress.json, remaining_jobs.json and evaluation log.
- descriptive_agreement.csv: complete-pair signed and absolute judge gaps;
  these are descriptive measures, not kappa/ICC.

Validated aggregates require every constituent chunk score. A failed call does
not change the denominator by silently dropping a chunk or a judge. Summary
tables report available and expected counts; incomplete-group means are not
substitutes for full-data results.

FAILURES, CONTEXT AND RETRIES

The original 8192 context setting is retained. prompt_eval_count near the context
boundary is flagged; generation stopped by a token limit is invalid for the
validated view. Recorded model/context checks verify the served model digest
and allocated context. The token-boundary check is conservative and does not
prove the absence of truncation below the boundary. No tokenizer/truncation
ground truth from the historical environment is available.

Five consecutive failed calls stop new submissions; active calls are logged.
Exit code 3 means validated collection is incomplete, not that saved records
were lost. A single transport failure is recorded as a failure and a legacy zero;
no silent network retry overwrites it. Default request timeout is 1800 seconds.

After reviewing transient failures, retry each currently failed call once:

  COLLECTOR_ACTION=retry bash judge_evaluation_collabeval_b200/run_b200.sh pilot

This leaves original legacy scores unchanged and accepts the first valid result
for the validated view. The same options/checkpoints are required. Retry does
not fill never-attempted jobs; run the normal command afterwards if some remain.
Do not repeatedly retry malformed scores just to obtain preferred results.

If you intentionally change num_ctx or num_predict, use a DIFFERENT output
directory. Such a run is a recorded sensitivity experiment, not the default
source-compatible pilot. Example, only when investigating truncation:

  OUTPUT_DIR=/path/to/project/b200_collabeval_pilot_ctx16384 \
    bash judge_evaluation_collabeval_b200/run_b200.sh pilot \
      --num-ctx 16384 --num-predict 8192

No output-file edits, automatic context expansion or score-dependent retries
are used to make the numbers resemble the paper.

OFFLINE COMMANDS

Audit inputs, without contacting Ollama:

  python collect_judgments.py audit --zip /path/to/input.zip --out ./audit --pilot

Re-export an existing run, without contacting Ollama:

  python collect_judgments.py export --zip /path/to/input.zip \
    --out /path/to/existing_pilot --pilot

Tests from the unpacked package:

  python -m unittest test_collector -v

The collector and standard tests use only the standard library. One additional
reference-parity test uses NumPy to exercise the co-author's actual classes and
functions with fake model responses; it is skipped if NumPy is absent.
collector_core.py is a frozen helper from the earlier B200 collector. Run the
NEW collect_judgments.py entry point, not that helper's old CLI.

API references
https://docs.ollama.com/api/generate
https://docs.ollama.com/api/ps
