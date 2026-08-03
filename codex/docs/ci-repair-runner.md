# Codex CI Repair Runner Setup

This document explains how to run Codex as a CI-repair agent over eval issues,
with and without the shared memory plugin.

> ** LLM Configuration**: This system supports **any LLM provider** (Claude, GPT, Gemini, Ollama, Cohere, etc.). See [LLM Configuration Guide](../../docs/LLM_CONFIGURATION.md) for complete setup instructions with different models.

The runner is:

```bash
codex/scripts/run_codex_ci_repair.py
```

It does not modify Codex core behavior. It prepares a task prompt for each CI
problem, optionally injects memory from the existing `memory_plugin`, executes
Codex, and saves the resulting patch and metadata.

## Required Inputs

Default dataset:

```bash
data/eval_set.jsonl
```

Default eval issue ID list:

```bash
data/eval_issue_ids.json
```

If `--issue-ids` is omitted, the runner loads all issue IDs from
`data/eval_issue_ids.json`, then loads the full issue records from
`data/eval_set.jsonl`. If the full local dataset is missing, it fetches from
Hugging Face through the shared loader:

```text
utilities.dataset_fetcher.fetch_dataset()
```

Each issue should contain fields such as:

```text
id
repo_owner
repo_name
sha_fail
workflow_path
workflow
logs
error_type
```

If you have an eval issues JSON file instead, pass it explicitly:

```bash
--dataset path/to/eval_issues.json
```

The JSON file must be a list of issue objects.

## Existing Analysis Inputs

The runner first tries to load cached CI context from:

```bash
data/log_details.json
data/workflow_validation_cache.json
```

Memory-enabled runs reuse the shared memory plugin and memory banks from:

```bash
data/back_trs/failure_memory.json
data/back_trs/repo_memory.json
data/back_trs/cross_memory.json
```

You can override the memory root:

```bash
--memory-root data/fwr_trs
```

## Codex Setup

Install and authenticate Codex CLI first. Codex is the repair agent; this
runner only prepares the task prompt and calls Codex.

Install Codex:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Or install with npm:

```bash
npm install -g @openai/codex
```

Authenticate:

```bash
codex
```

Follow the login flow, then exit the interactive session.

From the project root:

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
codex --version
codex exec --help
```

The runner invokes Codex with:

```bash
codex exec --full-auto
```

Override it if needed:

```bash
--codex-command "codex exec --full-auto"
```

Do a local smoke test:

```bash
mkdir -p /tmp/codex-smoke
cd /tmp/codex-smoke
git init
codex exec --full-auto "Create hello.txt containing hello"
git diff
```

If `git diff` shows a new `hello.txt`, Codex execution is working.

## Environment Setup

Use the root Python environment with the shared utilities and memory plugin
dependencies installed.

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-shared.txt
python -m pip install demjson3 datasets litellm typer rich
```

The runner loads CI failure and workflow verification from cache first. If a
cache entry is missing, it generates the missing analysis with the existing
LLM-backed analyzers:

```text
utilities/ci_log_analyzer.py
utilities/ci_workflow_aware_retrieval.py
```

Configure your model/API keys as used by `utilities/model_registry.py`, then
pass a context model:

```bash
--context-model minimax2.5
```

There is no fallback log analysis. If a cache entry is missing and no context
model is available, the runner exits with a clear error.

Cache entries that only contain parser errors are treated as unusable and are
regenerated the same way as missing entries.

For cache-only runs, explicitly disable generation:

```bash
--no-generate-missing-analysis
```

## Dry Run

Use dry run to generate task prompts and result files without cloning/running
Codex:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --dry-run \
```

Generated prompt:

```bash
results/codex/baseline/43/issue_document_problem_1.md
```

## Run Without Memory

Baseline means no memory context is injected.

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline
```

Cache-only baseline, no analysis generation:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --no-generate-missing-analysis
```

Baseline with missing-analysis generation:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --context-model minimax2.5
```

For multiple issues:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43,121,145 \
  --ablations baseline
```

Run all eval issues from `data/eval_issue_ids.json`:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --ablations baseline \
  --context-model minimax2.5 \
```

## Run With Memory

Backward-generated memory is the default:

```bash
data/back_trs/
```

Run with L1 only:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations L1 \
  --memory-root data/back_trs \
  --context-model minimax2.5
```

Run with L1 and L2:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations L1+L2 \
  --memory-root data/back_trs \
  --context-model minimax2.5
```

Run with full memory:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations L1+L2+L3 \
  --memory-root data/back_trs \
  --context-model minimax2.5
```

Run all comparison levels:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43,121,145 \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
  --memory-root data/back_trs \
  --context-model minimax2.5
```

Run all eval issues with backward-generated memory:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
  --memory-root data/back_trs \
  --context-model minimax2.5 \
```

## Run With Forward-Generated Memory

Forward-generated memory lives in:

```bash
data/fwr_trs/
```

Run full memory with forward-generated memory:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations L1+L2+L3 \
  --memory-root data/fwr_trs \
  --context-model minimax2.5
```

Run baseline plus forward-memory comparison:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43,121,145 \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
  --memory-root data/fwr_trs \
  --context-model minimax2.5
```

Run all eval issues with forward-generated memory:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
  --memory-root data/fwr_trs \
  --context-model minimax2.5 \
```

## Run With Eval Issues File

If your eval issues are stored in a JSON file:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --dataset data/eval_issues.json \
  --issue-ids 43,121,145 \
  --ablations baseline,L1+L2+L3
```

If using the default `data/eval_set.jsonl`, omit `--dataset`.

## Verification Command

The workflow-aware analyzer records candidate validation commands, but some
steps are descriptive rather than directly runnable. For benchmark pass/fail,
pass a concrete command when known:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline,L1+L2+L3 \
  --verification-command "python -m pytest"
```

The command result is saved in:

```bash
results/codex/<ablation>/<issue_id>/verification.log
results/codex/<ablation>/<issue_id>/result.json
```

## Per-Problem Execution

For each issue, the runner extracts a problem list from CI analysis. It then
passes one problem at a time to Codex.

Each Codex task prompt contains:

```text
Problem To Fix
Previous Experience / Memory
CI Verification
Required Workflow
Scope Rules
Final Response requirements
```

This lets Codex focus on one repair step while preserving earlier patches in
the same checkout.

Limit how many problems Codex receives:

```bash
```

## Output Layout

Results are written to:

```bash
results/codex/<ablation>/<issue_id>/
```

Important files:

```text
checkout/                         repository checkout at sha_fail
ci_failure.json                   loaded/generated CI failure analysis
ci_failure.md                     markdown CI failure summary
ci_verification.json              workflow-aware verification data
memory_context.md                 memory prompt section, for memory runs
memory_retrieval.json             raw memory retrieval result
issue_document_problem_1.md       prompt passed to Codex
codex_transcript_problem_1.txt    Codex output
patch.diff                        final git diff
verification.log                  optional verification command output
result.json                       patch/verification metadata
```

## Recommended Benchmark Commands

Smoke test:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --dry-run \
```

Single issue, full comparison:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
```

Batch eval:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43,121,145 \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
```

Batch eval with automatic missing analysis generation:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43,121,145 \
  --ablations baseline,L1,L1+L2,L1+L2+L3 \
  --context-model minimax2.5
```
