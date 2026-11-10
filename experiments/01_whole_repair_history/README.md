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
- the ordered commit sequence;
- the complete cumulative fail-to-pass patch;
- all changed files; and
- the final successful CI outcome.

The ordered commits are evidence nested inside the PR record. They are not
separate memory records and are not associated with individual CI problems.

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
  one-element `problems` array, but the element describes the complete PR.
- **L2:** one repository-level strategy in a one-element `repair_strategies`
  array.
- **L3:** one flat transferable whole-PR pattern.

For the fixed 157 historical PRs, a completed build therefore contains 157 L1
records, 157 L2 strategies, and 157 L3 patterns.

## Build the memory

The default command uses `data/memory_issue_ids.json`, the raw
`ci-benchmark-user/ci-repair-bench` dataset, GitHub compare data for ordered
commits, and DeepSeek-V4-Flash for the three singular abstractions:

```bash
export GITHUB_TOKEN=<github-token>
python experiments/01_whole_repair_history/build_whole_pr_memory.py \
  --model deepseek-v4-flash
```

A GitHub token is required for the complete 157-PR build because the
unauthenticated hourly API allowance is too small. The script checks the
available allowance before collecting and resumes from completed raw episodes
and completed L1/L2/L3 records.

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

`--allow-incomplete` and `--no-github` are debugging options only. Data built
with either option is not reviewer-grade because the ordered commit sequence
may be missing. The manifest records any incompleteness and any LLM evidence
truncation.

## Generated files

```text
experimental_data/whole_repair_history/
├── memory_issue_ids.json
├── raw_episodes.json
├── build_manifest.json
└── bidirect_trs/
    ├── failure_memory.json
    ├── repo_memory.json
    └── cross_memory.json
```

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
