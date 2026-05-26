# mini-swe-agent CI Benchmark Workspace

This workspace is a trimmed CI-repair benchmark setup built around `mini-swe-agent`.
It is intended to run CI-failure repair experiments on the prepared eval split in:

- `data/eval_dataset.jsonl`
- `data/memory_seed.jsonl`
- `results/shared_memory/`

It supports four ablations:

- `baseline`
- `L1`
- `L1+L2`
- `L1+L2+L3`

## What This Workspace Contains

- `scripts/run_cibench_minimax_openrouter.sh`
  - main run entrypoint
- `src/minisweagent/run/benchmarks/cibench.py`
  - benchmark runner
- `results/shared_memory/`
  - seeded L1/L2/L3 memory bank
- `repo/`
  - shared repo clone cache

## Environment Setup

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install runtime dependencies:

```bash
pip install requests pyyaml jinja2 pydantic tenacity rich python-dotenv typer platformdirs textual prompt_toolkit datasets openai litellm
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

## Data Check

Confirm required files exist:

```bash
ls -lh data/eval_dataset.jsonl data/memory_seed.jsonl
ls -lh results/shared_memory/
```

Expected:

- `data/eval_dataset.jsonl`
- `data/memory_seed.jsonl`
- `results/shared_memory/failure_memory.json`
- `results/shared_memory/repo_memory.json`
- `results/shared_memory/cross_memory.json`

## Quick Sanity Checks

```bash
source .venv/bin/activate
python -m py_compile src/minisweagent/run/benchmarks/cibench.py
python -m py_compile src/minisweagent/run/benchmarks/utils/ci_context.py
bash -n scripts/run_cibench_minimax_openrouter.sh
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

## Server / `nohup` Runs

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
- `<instance_id>/<instance_id>.traj.json`
  - full agent trajectory

Examples:

```bash
tail -n 100 results/baseline/cibench.log
cat results/baseline/preds.json
```

Inspect one issue:

```bash
ls results/baseline/<instance_id>/
cat results/baseline/<instance_id>/<instance_id>.traj.json
```

## Notes About Validation

- installation and validation commands are provided as hints
- the agent may use them, extend them, or ignore them
- local validation is optional
- patch generation does not require successful local reproduction

## Notes About Repo Cloning

- remote repos are cached under `repo/`
- each issue gets its own isolated working copy under `results/.../testbed`
- the runner fetches the exact historical `sha_fail` if needed
- the worktree is force-checked out, hard-reset, and cleaned before the agent runs

## Common Cleanup

Delete one broken instance run:

```bash
rm -rf results/baseline/<instance_id>
```

Delete prior predictions for a rerun:

```bash
rm -f results/baseline/preds.json
```
