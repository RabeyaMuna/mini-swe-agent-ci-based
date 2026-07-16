# CI-Repair-Bench: Multi-Agent Evaluation Framework

Benchmark for evaluating agent scaffolds on CI failure repair with memory-guided repair at the pull-request level.

## 🎯 Problem Statement

Unlike existing benchmarks (SWE-bench, SWE-bench Verified, SWE-bench Pro) that focus on resolving single, atomic issues, **CI-Repair-Bench targets CI failure repair at the pull-request level**, where:

- One PR may contain **multiple commits**
- Multiple **types of CI failures** (style checks, dependency issues, test failures, configuration errors)
- Multiple **verification stages** (linting, tests, builds, deployment checks)
- **Merge-related problems** (conflicts, integration issues)

This is closer to real-world software development workflows.

## 🏗️ Project Structure

```
.
├── data/              # Shared datasets and memory
│   └── trs/          # Three-layer memory system
│       ├── failure_memory.json      # L1: Similar CI failures
│       ├── repo_memory.json         # L2: Repository-specific patterns
│       ├── cross_memory.json        # L3: Cross-repository failures
│       ├── eval_set.jsonl           # Evaluation dataset
│       └── memory_set.jsonl         # Memory building dataset
│
├── results/          # All experiment results
│   ├── miniswe-agent/
│   │   ├── minimax/{baseline,L1,L1_L2,L1_L2_L3}/
│   │   ├── glm/{baseline,L1_L2_L3}/
│   │   └── kimi/{baseline,L1_L2_L3}/
│   └── openhands/
│       ├── glm/{baseline,L1_L2_L3}/
│       └── minimax/{baseline,L1_L2_L3}/
│
├── repo/             # Testbed repository clones
│   └── {owner}__{repo}/
│
├── scripts/          # Evaluation and memory tools
│   ├── decompose_ci_failure.py       # Build memory
│   ├── evaluate_ablation_preds.py    # Calculate metrics
│   ├── compare_runs.py               # Compare experiments
│   └── run_eval.py                   # Run evaluations
│
├── miniswe-agent/    # Mini-SWE-Agent implementation
│   ├── src/minisweagent/
│   ├── tests/
│   └── README.md
│
└── openhands/        # OpenHands integration (TODO)
    └── README.md
```

## 📊 Three-Layer Memory System

| Layer | Name | Description |
|-------|------|-------------|
| **L1** | Failure Memory | Similar CI failures from the same repository |
| **L2** | Repository Memory | Repository-specific patterns and conventions |
| **L3** | Cross-Repository Memory | Common failures across different projects |

Memory is retrieved **one problem at a time** to reduce hallucination.

## Supported Ablations

- `baseline` — No memory
- `L1` — Failure memory only
- `L1_L2` (or `L1+L2`) — Failure + Repository memory
- `L1_L2_L3` (or `L1+L2+L3`) — All three layers

## 🚀 Quick Setup

### Automated Setup (Recommended)

```bash
# Run automated setup script
bash setup_environments.sh
```

This creates 3 virtual environments:
- `.venv/` - Shared tools (memory, evaluation)
- `miniswe-agent/.venv/` - Mini-SWE-Agent
- `openhands/.venv/` - OpenHands

### Manual Setup

#### 1. ROOT Environment (Shared Tools)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-shared.txt
deactivate
```

#### 2. Mini-SWE-Agent
```bash
cd miniswe-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install sentence-transformers
deactivate
```

#### 3. OpenHands (Optional)
```bash
cd openhands
python3.12 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
deactivate
```

### First Experiment

```bash
# Activate mini-swe-agent environment
cd miniswe-agent
source .venv/bin/activate

# Run on first 5 issues
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test

# Evaluate
cd ..
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/test/preds.json
```

See [FINAL_SETUP_SUMMARY.md](FINAL_SETUP_SUMMARY.md) for complete guide.

## What This Workspace Contains

- `scripts/run_cibench_minimax_openrouter.sh`
  - main run entrypoint
- `src/minisweagent/run/benchmarks/cibench.py`
  - benchmark runner
- `data/trs/`
  - seeded L1/L2/L3 memory bank (high-quality, built from CI-REPAIR-BENCH)
- `repo/`
  - shared repo clone cache

## Setup Overview

There are two common ways to run this workspace:

- local machine
  - foreground runs first, then longer runs
- remote server / dev server
  - smoke test first, then `nohup` background runs

This workspace is now installable again via `pip install -e .`.

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install the workspace:

```bash
python -m pip install -e .
```

Install an embedding backend for memory retrieval:

```bash
python -m pip install sentence-transformers
```

If `sentence-transformers` is problematic on your machine, use:

```bash
python -m pip install fastembed
```

## Server Setup

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install the workspace:

```bash
python -m pip install -e .
```

Install an embedding backend for memory retrieval:

```bash
python -m pip install sentence-transformers
```

Fallback:

```bash
python -m pip install fastembed
```

## Project `.env`

Create a local `.env` in the project root:

```bash
cat > .env <<'EOF'
MINIMAX_API_KEY=<your_openrouter_key>
OPENROUTER_API_KEY=<your_openrouter_key>
MINIMAX_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MEMCI_LLM_MODEL=minimax/minimax-m2.5
PYTHON_BIN=.venv/bin/python
EOF
```

Optional:

```bash
echo 'MSWEA_REPO_CACHE_ROOT=/path/to/shared/repo-cache' >> .env
```

Notes:

- Use `minimax/minimax-m2.5`, not `MiniMax-M2.5`
- Shell exports override `.env`
- Repo cache defaults to `<project>/repo`
- `PYTHON_BIN` should usually point to the venv Python on that machine

## Required Packages

Installed by `pip install -e .`:

- `requests`
- `pyyaml`
- `jinja2`
- `pydantic`
- `litellm`
- `tenacity`
- `rich`
- `python-dotenv`
- `typer`
- `platformdirs`
- `textual`
- `prompt_toolkit`
- `datasets`
- `openai`

Additional package needed for memory-enabled runs:

- `sentence-transformers`
  - recommended
- or `fastembed`
  - fallback

Without one of those, `L1`, `L1+L2`, and `L1+L2+L3` will log:

```text
No embedding model available ... memory retrieval disabled
```

and behave like no-memory runs.

## Data Check

Confirm required files exist:

```bash
ls -lh data/trs/eval_issues.json
ls -lh data/trs/
```

Expected:

- `data/trs/eval_issues.json` — evaluation dataset
- `data/trs/failure_memory.json` — L1 per-file memories (1.3MB)
- `data/trs/repo_memory.json` — L2 repo-level patterns (900KB)
- `data/trs/cross_memory.json` — L3 universal principles (468KB)

## Quick Sanity Checks

```bash
source .venv/bin/activate
python -c "import minisweagent; print(minisweagent.__file__)"
python -m minisweagent.run.benchmarks.cibench --help
python -m py_compile src/minisweagent/run/benchmarks/cibench.py
python -m py_compile src/minisweagent/run/benchmarks/utils/ci_context.py
bash -n scripts/run_cibench_minimax_openrouter.sh
```

Check memory backend:

```bash
python -c "import sentence_transformers; print('sentence-transformers ok')"
```

or:

```bash
python -c "import fastembed; print('fastembed ok')"
```

## Smoke Test

Run a single instance:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:1
```

Run first 10:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:10
```

Memory-enabled smoke test:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1 --slice 0:1
```

In memory-enabled logs, confirm you do **not** see:

```text
No embedding model available
Cosine similarity will return 0.0
memory retrieval disabled
```

## Main Run Commands

Baseline:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline
```

L1:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1
```

L1 + L2:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2
```

L1 + L2 + L3:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3
```

The last one is also the default:

```bash
bash scripts/run_cibench_minimax_openrouter.sh
```

## Run Selected Issues Only

Example:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline --filter '^(110e09997f8a22a617e261dec9e301129bbead65|16c6139ab089)'
```

## Run On Local Machine

Recommended order:

1. smoke test `baseline --slice 0:1`
2. smoke test `L1 --slice 0:1`
3. full `baseline`
4. full memory runs

Examples:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:1
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1 --slice 0:1
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline
```

## Run On Server / `nohup`

Use separate logs per run:

```bash
nohup bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline > baseline.log 2>&1 &
nohup bash scripts/run_cibench_minimax_openrouter.sh --ablation L1 > l1.log 2>&1 &
nohup bash scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2 > l1_l2.log 2>&1 &
nohup bash scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3 > l1_l2_l3.log 2>&1 &
```

Monitor:

```bash
tail -n 50 baseline.log
tail -n 50 l1.log
tail -n 50 l1_l2.log
tail -n 50 l1_l2_l3.log
```

Safer server workflow:

1. run one smoke test first
2. confirm imports, checkout, and model access work
3. confirm memory backend is installed
4. then launch the full `nohup` jobs

## Stop Background Jobs

Kill all benchmark wrapper jobs:

```bash
pkill -f 'scripts/run_cibench_minimax_openrouter.sh'
pkill -f 'minisweagent.run.benchmarks.cibench'
```

## Outputs

Each run writes to:

- `results/baseline/`
- `results/l1/`
- `results/l1_l2/`
- `results/l1_l2_l3/`

Important files:

- `preds.json`
  - predicted patch per instance
- `cibench.log`
  - runner log
- `<sha_fail>/<sha_fail>.traj.json`
  - full agent trajectory

Examples:

```bash
tail -n 100 results/baseline/cibench.log
cat results/baseline/preds.json
```

Inspect one issue:

```bash
ls results/baseline/<instance_id>/
cat results/baseline/<sha_fail>/<sha_fail>.traj.json
```

## Notes About Validation

- installation and validation commands are provided as hints
- the agent may use them, extend them, or ignore them
- local validation is optional
- patch generation does not require successful local reproduction
- the agent may run extra setup commands first if it decides local execution is worthwhile

## Notes About Repo Cloning

- remote repos are cached under `repo/`
- each issue gets its own isolated working copy under `results/.../testbed`
- the runner fetches the exact historical `sha_fail` if needed
- the worktree is force-checked out, hard-reset, and cleaned before the agent runs

## Troubleshooting

### `ModuleNotFoundError: No module named 'minisweagent'`

Fix:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Then verify:

```bash
python -c "import minisweagent; print(minisweagent.__file__)"
```

### `No embedding model available ... memory retrieval disabled`

Fix:

```bash
source .venv/bin/activate
python -m pip install sentence-transformers
```

Fallback:

```bash
python -m pip install fastembed
```

### `fatal: unable to read tree` or missing historical commit

The runner now fetches the exact `sha_fail`, force-checks it out, hard-resets, and cleans the worktree.
If a run was interrupted, remove that instance directory and rerun:

```bash
rm -rf results/baseline/<instance_id>
```

### Existing partial run outputs

Delete one broken instance run:

```bash
rm -rf results/baseline/<instance_id>
```

Delete prior predictions for a rerun:

```bash
rm -f results/baseline/preds.json
```

## Common Cleanup

Delete one broken instance run:

```bash
rm -rf results/baseline/<instance_id>
```

Delete prior predictions for a rerun:

```bash
rm -f results/baseline/preds.json
```
