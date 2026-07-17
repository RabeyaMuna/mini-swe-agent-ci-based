# CI-Repair-Bench Evaluation Setup

This repository runs CI repair evaluations for two agents:

- `mini-swe-agent`
- `openhands`

Each agent can be evaluated with two models:

- `openrouter/minimax/minimax-m2.5`
- `openrouter/z-ai/glm-5.2`

Each model is run at four ablation levels:

- `baseline`: no memory
- `L1`: failure memory
- `L1+L2`: failure memory + repository memory
- `L1+L2+L3`: full memory

The evaluation dataset is:

```bash
data/trs/eval_set.jsonl
```

This file contains the full 89 eval issue records. Do not use `data/trs/eval_issue_ids.json` for Mini-SWE-Agent runs because it contains only IDs, not the full issue data.

## Setup

Run everything from the project root:

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
```

### 1. Shared Root Environment

The root environment is used for shared scripts, memory utilities, and result evaluation.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-shared.txt
python -m pip install demjson3 datasets litellm typer rich

deactivate
```

Verify:

```bash
source .venv/bin/activate
python -c "import numpy, pandas, sentence_transformers, demjson3; print('root env ok')"
deactivate
```

### 2. Mini-SWE-Agent Environment

Mini-SWE-Agent has its own isolated environment under `miniswe-agent/.venv`.

```bash
cd miniswe-agent

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install demjson3 sentence-transformers fastembed chromadb datasets

deactivate
cd ..
```

Verify:

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
export PYTHONPATH="$PWD/miniswe-agent/src:$PWD"
miniswe-agent/.venv/bin/python -c "import minisweagent, demjson3; print('mini-swe-agent env ok')"
miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench --help
```

### 3. OpenHands Environment

OpenHands has its own isolated environment under `openhands/.venv`.

Use Python 3.12 or newer if available. If your machine only has `python3`, use that.

```bash
cd openhands

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install poetry
poetry install
python -m pip install demjson3 datasets litellm

deactivate
cd ..
```

If `python3.12` is not installed:

```bash
cd openhands

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install poetry
poetry install
python -m pip install demjson3 datasets litellm

deactivate
cd ..
```

Verify:

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
openhands/.venv/bin/python openhands/ci_bench_runner.py --help
```

### 4. API Keys

Configure API keys in the root `.env` file:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MINIMAX_API_KEY=your_openrouter_key
MINIMAX_BASE_URL=https://openrouter.ai/api/v1
EOF
```

## Cache Behavior

The runner loads precomputed CI analysis from:

```bash
data/log_details.json
data/workflow_validation_cache.json
```

If these cache files exist, the runner uses them instead of regenerating CI log analysis and workflow validation. To force regeneration for missing cache entries only:

```bash
export CIBENCH_REGENERATE_MISSING_CACHE=1
```

## Common Environment Variables

Set these once in your shell before running direct commands:

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
export PYTHONPATH="$PWD/miniswe-agent/src:$PWD"
export MINI_PY="$PWD/miniswe-agent/.venv/bin/python"
export OH_PY="$PWD/openhands/.venv/bin/python"

test -x "$MINI_PY" && echo "MINI_PY ok: $MINI_PY"
test -x "$OH_PY" && echo "OH_PY ok: $OH_PY"
```

For a quick smoke test, append `--slice 0:5` to Mini-SWE-Agent commands and OpenHands commands.

## Model Names

Use either the full provider model name or a short alias.

| Alias | Resolved model |
|---|---|
| `minimax2.5` | `openrouter/minimax/minimax-m2.5` |
| `minimax-m2.5` | `openrouter/minimax/minimax-m2.5` |
| `glm5.2` | `openrouter/z-ai/glm-5.2` |
| `glm-5.2` | `openrouter/z-ai/glm-5.2` |

The same model value is used for:

- memory building
- cached CI log/workflow generation when regeneration is enabled
- memory retrieval LLM steps
- Mini-SWE-Agent repair runs
- OpenHands shared memory adapter runs

## Build Or Rebuild Memory With A Selected Model

MiniMax 2.5:

```bash
source .venv/bin/activate
python scripts/decompose_ci_failure.py \
  --dataset data/trs/memory_set.jsonl \
  --output-dir data/trs \
  --model minimax2.5
deactivate
```

GLM-5.2:

```bash
source .venv/bin/activate
python scripts/decompose_ci_failure.py \
  --dataset data/trs/memory_set.jsonl \
  --output-dir data/trs \
  --model glm5.2
deactivate
```

Full memory/eval split workflow with a selected model:

```bash
MODEL=minimax2.5 bash scripts/run_memory_split_workflow.sh
MODEL=glm5.2 bash scripts/run_memory_split_workflow.sh
```

## Mini-SWE-Agent Commands

### Mini-SWE-Agent + MiniMax 2.5

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model minimax2.5 \
  --output results/miniswe-agent/minimax-m2.5/baseline \
  --no-memory-enabled
```

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model minimax2.5 \
  --output results/miniswe-agent/minimax-m2.5/L1 \
  --memory-enabled \
  --memory-root data/trs \
  --memory-ablation L1
```

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model minimax2.5 \
  --output results/miniswe-agent/minimax-m2.5/L1_L2 \
  --memory-enabled \
  --memory-root data/trs \
  --memory-ablation L1+L2
```

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model minimax2.5 \
  --output results/miniswe-agent/minimax-m2.5/L1_L2_L3 \
  --memory-enabled \
  --memory-root data/trs \
  --memory-ablation L1+L2+L3
```

### Mini-SWE-Agent + GLM-5.2

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model glm5.2 \
  --output results/miniswe-agent/glm-5.2/baseline \
  --no-memory-enabled
```

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model glm5.2 \
  --output results/miniswe-agent/glm-5.2/L1 \
  --memory-enabled \
  --memory-root data/trs \
  --memory-ablation L1
```

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model glm5.2 \
  --output results/miniswe-agent/glm-5.2/L1_L2 \
  --memory-enabled \
  --memory-root data/trs \
  --memory-ablation L1+L2
```

```bash
$MINI_PY -m minisweagent.run.benchmarks.cibench \
  --dataset data/trs/eval_set.jsonl \
  --model glm5.2 \
  --output results/miniswe-agent/glm-5.2/L1_L2_L3 \
  --memory-enabled \
  --memory-root data/trs \
  --memory-ablation L1+L2+L3
```

## OpenHands Commands

### OpenHands + MiniMax 2.5

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode baseline \
  --model minimax2.5 \
  --output results/openhands/minimax-m2.5/baseline
```

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode memory \
  --memory-layers L1 \
  --model minimax2.5 \
  --output results/openhands/minimax-m2.5/L1
```

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode memory \
  --memory-layers L1 L2 \
  --model minimax2.5 \
  --output results/openhands/minimax-m2.5/L1_L2
```

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode memory \
  --memory-layers L1 L2 L3 \
  --model minimax2.5 \
  --output results/openhands/minimax-m2.5/L1_L2_L3
```

### OpenHands + GLM-5.2

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode baseline \
  --model glm5.2 \
  --output results/openhands/glm-5.2/baseline
```

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode memory \
  --memory-layers L1 \
  --model glm5.2 \
  --output results/openhands/glm-5.2/L1
```

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode memory \
  --memory-layers L1 L2 \
  --model glm5.2 \
  --output results/openhands/glm-5.2/L1_L2
```

```bash
$OH_PY openhands/ci_bench_runner.py \
  --eval-issues data/trs/eval_set.jsonl \
  --mode memory \
  --memory-layers L1 L2 L3 \
  --model glm5.2 \
  --output results/openhands/glm-5.2/L1_L2_L3
```

## Evaluate Results

Evaluate one result file:

```bash
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py \
  results/miniswe-agent/minimax-m2.5/baseline/preds.json
```

Compare multiple result files:

```bash
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py \
  results/miniswe-agent/minimax-m2.5/baseline/preds.json \
  results/miniswe-agent/minimax-m2.5/L1/preds.json \
  results/miniswe-agent/minimax-m2.5/L1_L2/preds.json \
  results/miniswe-agent/minimax-m2.5/L1_L2_L3/preds.json
```

## Output Layout

Results are written under:

```bash
results/{agent}/{model}/{ablation}/preds.json
```

Examples:

```bash
results/miniswe-agent/minimax-m2.5/baseline/preds.json
results/miniswe-agent/glm-5.2/L1_L2_L3/preds.json
results/openhands/minimax-m2.5/L1/preds.json
results/openhands/glm-5.2/L1_L2/preds.json
```
