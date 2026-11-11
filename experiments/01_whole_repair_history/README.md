# Whole-Repair-History Baseline (Reviewer jcJU Q1)

## Reviewer question

> What is Pass@1 when the whole PR is treated as one record in the same pipeline?

This control measures whether ReCAP's problem-specific reconstruction and
problem–repair attribution improve repair performance over representing each
historical fail-to-pass PR as one complete repair history.

## Experimental unit

One historical PR remains one indivisible record. For every PR, the builder
collects:

- initial failing CI evidence;
- repository and workflow metadata;
- the fail-to-pass SHA transition and commit metadata available in the dataset;
- the complete cumulative fail-to-pass patch;
- all changed files; and
- the final successful CI outcome.

The complete cumulative patch is treated as one aggregate repair event. The
builder does not reconstruct or separate individual commits, and it does not
associate changes with individual CI problems.

## What is excluded

The control does not use:

- CI-problem decomposition;
- problem–commit or problem–hunk attribution;
- forward/backward problem reconciliation;
- problem dependency expansion; or
- problem-specific repair ordering.

The builder reads raw benchmark rows and log-only CI analysis. It must not use
`data/bidirect_trs`, `data/back_trs`, or `data/fwr_trs` as construction input,
because those files already contain ReCAP's reconstructed associations.

## Three compatible memory levels

Each source PR produces exactly one item per level:

- **L1:** one concrete whole-PR episode. For compatibility, it is wrapped in a
  one-element `problems` array, but the element describes the complete PR and
  is marked `record_scope: whole_pr`. It has no `problem_id`, `enabled`,
  `dependencies`, or `repair_sequence` fields.
- **L2:** one repository-level strategy in a one-element `repair_strategies`
  array.
- **L3:** one flat transferable whole-PR pattern.

Construction is sequential, matching the normal abstraction direction while
using singular whole-PR prompts:

```text
raw whole-PR evidence
  -> direct whole-PR analysis when it fits
  -> dependency-graph chunk summaries + hierarchical reduction when oversized
  -> one L1 record -> one L2 strategy -> one L3 pattern
```

Oversized cumulative patches reuse the repository's deterministic diff parser,
changed-file dependency graph, and token-aware chunker. Related callers,
callees, tests, configuration, and source files stay together when possible;
oversized components are split at file/hunk boundaries with graph context
retained. A strict coverage check adds a recovery chunk if the shared chunker
omits a file or hunk. CI logs and workflow evidence are chunked separately at
logical source boundaries. All chunk summaries are reduced into one aggregate
L1 input; chunks never become problems or retrieval records.

For CI evidence, the builder uses the existing structured log analysis when
available and records the raw-log count, size, and SHA-256 fingerprint. It
falls back to raw logs only when structured analysis is unavailable. Full
patch evidence is never truncated. Each L1 record reports whether chunking was
patch evidence is never truncated. Construction details such as graph/chunk
counts, file/hunk coverage, reduction rounds, and `evidence_truncated: false`
are stored separately in `abstraction_metadata.json`; they are not embedded in
retrieval-facing L1 memory.

The ordinary plural-producing L2/L3 prompts are not used because they could
reintroduce problem-specific records.

For the fixed 157 historical PRs, a completed build therefore contains 157 L1
records, 157 L2 strategies, and 157 L3 patterns.

## Recent Improvements (2026-11-10)

**Token Tracking & Error Handling**  
The build script now provides comprehensive real-time token usage tracking and
improved error handling. See [CHANGES.md](CHANGES.md) for detailed documentation.

Key features:
- Real-time token usage display for every API call
- Per-instance summary with phase breakdown
- Aggregate statistics with cost estimates for remaining work
- Enhanced error handling for token length limits
- Detailed phase tracking (patch chunks, reduction rounds, L1/L2/L3)

All token usage and costs are automatically saved to:
- `construction_metrics/run_metrics.json` - Full API call ledger
- `construction_metrics/cost_time_report.json` - Summary report
- `construction_metrics/detailed_usage.json` - Per-instance breakdown

## Build the memory

The default command uses `data/memory_issue_ids.json`, the pinned raw
`ci-benchmark-user/ci-repair-bench` dataset, and DeepSeek-V4-Flash for the
three singular abstractions. It does not call GitHub or require a GitHub token:

```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py \
  --model deepseek-v4-flash
```

The build provides real-time token usage feedback and estimates costs for
remaining instances. At the end, you'll see aggregate statistics including
estimated total cost for all 157 instances.

All 157 records use the same benchmark-only evidence policy: failing logs,
workflow and repository metadata, fail/pass SHAs, cumulative repair diff,
changed files, and the benchmark's successful endpoint. This avoids mixing
GitHub-enriched records with dataset-only records. The script resumes from
completed raw episodes and completed L1/L2/L3 records.

If a raw local benchmark JSONL is available, use it to pin the exact rows:

```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py \
  --dataset data/memory_set.jsonl \
  --model deepseek-v4-flash
```

Before the paid abstraction pass, raw collection can be checked on one PR:

```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py \
  --collect-only \
  --limit 1
```

`--allow-incomplete` is a debugging option only. Reviewer-grade data should
contain every required benchmark field. The manifest records incompleteness,
which issues used dependency-graph chunking, and verifies that no abstraction
evidence was truncated.

## Generated files

```text
experimental_data/whole_repair_history/
├── memory_issue_ids.json
├── raw_episodes.json
├── abstraction_metadata.json
├── build_manifest.json
├── construction_metrics/
│   ├── run_metrics.json
│   ├── cost_time_report.json
│   ├── cost_time_report.csv
│   └── detailed_usage.json
└── whole_pr_trs/
    ├── failure_memory.json
    ├── repo_memory.json
    └── cross_memory.json
```

Memory construction accounting is checkpointed after every LLM call. The JSON
and CSV reports include per-PR and aggregate API calls, input, cached-input,
cache-write, output and reasoning tokens, API time, cost, pricing source, and
failed or resumed attempts. `build_manifest.json` records whether every
generated memory record has complete construction-metrics coverage. A build
that was partially completed before accounting was introduced must be rebuilt
to obtain reviewer-grade totals for all historical PRs; missing historical
calls are never estimated silently.

The existing standard and cross-repository memories are not modified.

## Run the fixed 100-instance evaluation

Use the same 100 evaluation IDs, dataset, repair model, retrieval settings,
context budget, and CI validation protocol used for Full ReCAP:

```bash
bash ./run_miniswe_direct.sh \
  experimental_data/eval_issue_ids.json \
  L1+L2+L3 whole_history deepseek-v4-flash \
  "" experimental_data/eval_set.jsonl 1
```

Results are saved under:

```text
results/miniswe-agent/whole_history/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/
```

Compare that output against Full ReCAP on the exact same 100 IDs:

```text
results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/
```

Report paired successes and failures and use an exact paired McNemar test. Do
not compare the 100-instance whole-history result with a previously reported
200-instance Full ReCAP number.
