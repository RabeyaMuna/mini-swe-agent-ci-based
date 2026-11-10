# Cross-Repository Evaluation

This experiment evaluates the same fixed 100 targets under three conditions:

1. Existing No-Experience baseline (reuse the matching 100 result rows)
2. Normal ReCAP with repository-disjoint L1/L2/L3 memory
3. ReCAP with repository-disjoint L3-only memory

The generated memory contains 110 L1 records, the corresponding 110 L2
records, and 302 L3 patterns. Its 45 normalized source repository names have
zero overlap with the 30 target repository names.

## Prepare and validate data

```bash
bash experiments/04_cross_repository_l3/setup.sh
```

This uses the fixed 100 IDs in `experimental_data/eval_issue_ids.json`, excludes
all 47 histories from their repositories, and writes the filtered memory to
`experimental_data/bidirect_trs/`.

## Experiment types and commands

All configurations use the same 100 IDs from
`experimental_data/eval_issue_ids.json` and the corresponding records in
`experimental_data/eval_set.jsonl`.

| Type | Configuration | Memory enabled | Purpose |
|---|---|---|---|
| Baseline | No-Experience | None | Shared reference for both reviewers |
| Cross-repository full | Normal ReCAP | L1 + L2 + L3 | Test ReCAP without target-repository history |
| Cross-repository ablation | L3-only ReCAP | L3 only | Isolate transferable cross-repository patterns |

### Type 1: No-Experience baseline

Reuse the matching 100 rows from the existing baseline:

```text
results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl
```

Filter this result file using the IDs in
`experimental_data/eval_issue_ids.json`. A new baseline run is unnecessary
when its model, agent settings, and execution budget match the two runs below.

### Type 2: Normal cross-repository ReCAP (L1 + L2 + L3)

Direct command:

```bash
bash ./run_miniswe_direct.sh \
  experimental_data/eval_issue_ids.json \
  L1+L2+L3 cross_repo deepseek-v4-flash \
  "" experimental_data/eval_set.jsonl 1
```

Convenience script:

```bash
bash experiments/04_cross_repository_l3/run_cross_repo_l1_l2_l3.sh
```

Memory loaded:

- L1: 110 records from `failure_memory.json`
- L2: 110 records from `repo_memory.json`
- L3: 302 patterns from `cross_memory.json`

Results:

```text
results/miniswe-agent/cross_repo/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/
```

This is the primary comparison against No-Experience for reviewer xm6z Q3.

### Type 3: Cross-repository L3-only

Direct command:

```bash
bash ./run_miniswe_direct.sh \
  experimental_data/eval_issue_ids.json \
  L3 cross_repo deepseek-v4-flash \
  "" experimental_data/eval_set.jsonl 1
```

Convenience script:

```bash
bash experiments/04_cross_repository_l3/run_cross_repo_l3_only.sh
```

Memory loaded:

- L1: disabled
- L2: disabled
- L3: 302 patterns from `cross_memory.json`

Results:

```text
results/miniswe-agent/cross_repo/l3_deepseek-v4-flash/L3/
```

This is the primary comparison against No-Experience for reviewer jcJU Q4.

Each memory-enabled run validates the repository split before starting. The
main outcome file in each result directory is `jobs_results_diff.jsonl`.

## Result comparisons

| Comparison | Question answered |
|---|---|
| Cross-repository L1/L2/L3 vs. No-Experience | Does ReCAP work without target-repository history? |
| Cross-repository L3-only vs. No-Experience | Do L3 patterns transfer across repositories? |
| Cross-repository L1/L2/L3 vs. L3-only | What is the joint incremental effect of enabling L1 and L2? |

## What This Tests

- **jcJU Q4**: Does L3 transfer across repositories?
- **xm6z Q3**: Can ReCAP work without same-repo history?

The existing unrestricted `data/bidirect_trs` results are not substitutes for
these runs because 47 of their histories come from target repositories.
