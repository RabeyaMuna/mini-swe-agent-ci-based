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

## 📁 Data Organization

The evaluation dataset is now organized as follows:

```
data/
├── eval_set.jsonl              ← Eval issues (shared across workflows)
├── memory_set.jsonl            ← Memory issues (shared across workflows)
├── eval_issue_ids.json
├── memory_issue_ids.json
├── split_metadata.json
├── workflow_validation_cache.json
│
├── back_trs/                   ← Backward decomposition outputs
│   ├── decomposed_issues.json
│   ├── log_details.json
│   ├── failure_memory.json     (L1)
│   ├── repo_memory.json        (L2)
│   └── cross_memory.json       (L3)
│
└── fwr_trs/                    ← Forward (commit-based) outputs
    ├── commit_decomposed_issue.json
    ├── failure_memory.json     (L1)
    ├── repo_memory.json        (L2)
    └── cross_memory.json       (L3)
```

**Key points:**
- `data/eval_set.jsonl` contains the full eval issue records (NOT just IDs)
- `data/memory_set.jsonl` contains the memory issue records
- `data/back_trs/` contains backward decomposition outputs (default)
- `data/fwr_trs/` contains forward (commit-based) decomposition outputs
- Split files are shared across ALL decomposition approaches

**See [DATA_ORGANIZATION.md](DATA_ORGANIZATION.md) for complete details.**

---

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
# MiniMax M2.5 through OpenRouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM through native Z.ai-compatible LiteLLM provider
GLM_API_KEY=your_glm_provider_key

# Optional HuggingFace token, only needed for private/gated dataset access.
# HUGGINGFACE_TOKEN=your_huggingface_token
EOF
```

MiniMax and GLM can use different providers or credentials. The scripts select credentials by model:

- `MODEL=minimax2.5` uses `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL`.
- `MODEL=glm5.2` uses `GLM_API_KEY` with LiteLLM's native `zai` provider.
- Optional overrides are still supported: set `GLM_MODEL_NAME` if your GLM provider requires a different LiteLLM model string, and set `GLM_BASE_URL` only if your provider does not use the default Z.ai API base.
- Do not set `MEMCI_LLM_MODEL` in `.env` for this workflow. Pass the model explicitly with `MODEL=minimax2.5` or `MODEL=glm5.2`.
- Do not set duplicate MiniMax keys. With the config above, `MINIMAX_API_KEY` and `MINIMAX_BASE_URL` are not needed.

---

## Cache Behavior

The runner loads precomputed CI analysis from:

```bash
data/back_trs/log_details.json         # Backward decomposition cache
data/workflow_validation_cache.json     # Workflow validation cache
```

If these cache files exist, the runner uses them instead of regenerating CI log analysis and workflow validation. To force regeneration for missing cache entries only:

```bash
export CIBENCH_REGENERATE_MISSING_CACHE=1
```

---

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

---

## Model Names

The system supports two models with automatic configuration:

| Model | Full Name | API Provider | Auto-Config |
|---|---|---|---|
| `minimax2.5` | `openrouter/minimax/minimax-m2.5` | `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | 4x increase |
| `glm5.2` | `zai/glm-5.2`, override with `GLM_MODEL_NAME` only if needed | `GLM_API_KEY` | 10x increase |

**Supported aliases:**
- MiniMax: `minimax2.5`, `minimax-m2.5`, `minimax/minimax-m2.5`, `MiniMax-M2.5`
- GLM: `glm5.2`, `glm-5.2`, `z-ai/glm-5.2`, `GLM-5.2`

**Model-Aware Processing:**

When you specify a model, the entire system automatically uses:
- Correct API provider and credentials
- Correct token limits (100k vs 200k input chunks)
- Correct chunk sizes (15 vs 20 L1s per chunk)
- Correct candidate limits (120/160 vs 300/400 L2 candidates)

The same model value is used throughout:
- Memory building (decomposition, L1/L2/L3 generation)
- Cached CI log/workflow analysis
- Memory retrieval and selection
- Agent repair runs (Mini-SWE-Agent and OpenHands)

**See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed model specifications.**

---

## 🚀 Workflow: Split → Decompose → Build Memory → Evaluate

### **NEW Workflow (Temporal Leakage Prevention)**

The correct workflow is now:

1. **Split FIRST** (chronological) → creates `data/memory_set.jsonl` and `data/eval_set.jsonl`
2. **Decompose ONLY memory** → creates `data/back_trs/decomposed_issues.json`
3. **Build L1/L2/L3 memory** → creates memory files in `data/back_trs/`
4. **Evaluate** → uses eval set + memory

This prevents temporal data leakage by ensuring only past issues are in memory.

---

## Step 1: Split Dataset (Chronological)

Use the root environment.

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
```

### One Command (Recommended)

```bash
# All repos
python scripts/split_before_decomposition.py

# Specific repos (recommended)
python scripts/split_before_decomposition.py --repos agno,flower,camel

# Custom memory ratio (default 0.3 = 30%)
python scripts/split_before_decomposition.py --repos agno --memory-ratio 0.2
```

**Output:**
```
data/memory_set.jsonl        ← Earliest 30% (for decomposition)
data/eval_set.jsonl          ← Latest 70% (for evaluation)
data/memory_issue_ids.json
data/eval_issue_ids.json
data/split_metadata.json     ← Temporal safety confirmation
```

---

## Step 2: Decompose Memory (Backward or Forward)

### Option A: Backward Decomposition (Default, Recommended)

```bash
# MiniMax-M2.5
MODEL=minimax2.5 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model minimax2.5

# GLM-5.2
MODEL=glm5.2 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model glm5.2
```

**Output:**
```
data/back_trs/decomposed_issues.json
data/back_trs/log_details.json
```

### Option B: Forward (Commit-Based) Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 python commit_decomposition/run_commit_decomposition.py \
  --dataset data/memory_set.jsonl \
  --output data/fwr_trs/commit_decomposed_issue.json \
  --model minimax2.5

# GLM-5.2
MODEL=glm5.2 python commit_decomposition/run_commit_decomposition.py \
  --dataset data/memory_set.jsonl \
  --output data/fwr_trs/commit_decomposed_issue.json \
  --model glm5.2
```

**Output:**
```
data/fwr_trs/commit_decomposed_issue.json
```

---

## Step 3: Build L1/L2/L3 Memory

### For Backward Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/back_trs/decomposed_issues.json \
  --output-dir data/back_trs \
  --model minimax2.5

# GLM-5.2
MODEL=glm5.2 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/back_trs/decomposed_issues.json \
  --output-dir data/back_trs \
  --model glm5.2
```

**Output:**
```
data/back_trs/failure_memory.json  (L1)
data/back_trs/repo_memory.json     (L2)
data/back_trs/cross_memory.json    (L3)
```

### For Forward Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/fwr_trs/commit_decomposed_issue.json \
  --output-dir data/fwr_trs \
  --model minimax2.5

# GLM-5.2
MODEL=glm5.2 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/fwr_trs/commit_decomposed_issue.json \
  --output-dir data/fwr_trs \
  --model glm5.2
```

**Output:**
```
data/fwr_trs/failure_memory.json   (L1)
data/fwr_trs/repo_memory.json      (L2)
data/fwr_trs/cross_memory.json     (L3)
```

---

## Step 4: Evaluate

### Mini-SWE-Agent

#### Baseline (No Memory)

```bash
export PYTHONPATH="$PWD/miniswe-agent/src:$PWD"

# MiniMax-M2.5
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5

# GLM-5.2
MODEL=glm5.2 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model glm5.2
```

#### With Memory (L1+L2+L3) - Backward Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3

# GLM-5.2
MODEL=glm5.2 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model glm5.2 \
  --memory-enabled \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3
```

#### With Memory (L1+L2+L3) - Forward Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3

# GLM-5.2
MODEL=glm5.2 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model glm5.2 \
  --memory-enabled \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3
```

#### Ablation Studies

```bash
# L1 only
--memory-ablation L1

# L1+L2
--memory-ablation L1+L2

# L1+L2+L3
--memory-ablation L1+L2+L3
```

---

### OpenHands

#### Baseline (No Memory)

```bash
# MiniMax-M2.5
MODEL=minimax2.5 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --model minimax2.5 \
  --mode baseline

# GLM-5.2
MODEL=glm5.2 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --model glm5.2 \
  --mode baseline
```

#### With Memory (L1+L2+L3) - Backward Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --decomposed-issues data/back_trs/decomposed_issues.json \
  --model minimax2.5 \
  --mode memory \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3

# GLM-5.2
MODEL=glm5.2 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --decomposed-issues data/back_trs/decomposed_issues.json \
  --model glm5.2 \
  --mode memory \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3
```

#### With Memory (L1+L2+L3) - Forward Decomposition

```bash
# MiniMax-M2.5
MODEL=minimax2.5 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --decomposed-issues data/fwr_trs/commit_decomposed_issue.json \
  --model minimax2.5 \
  --mode memory \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3

# GLM-5.2
MODEL=glm5.2 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --decomposed-issues data/fwr_trs/commit_decomposed_issue.json \
  --model glm5.2 \
  --mode memory \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3
```

---

## 🔬 Comparing Decomposition Approaches

You can compare backward vs. forward decomposition:

```bash
# 1. Split once (shared)
python scripts/split_before_decomposition.py --repos agno,flower,camel

# 2. Run BOTH decompositions
MODEL=minimax2.5 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model minimax2.5

MODEL=minimax2.5 python commit_decomposition/run_commit_decomposition.py \
  --dataset data/memory_set.jsonl \
  --output data/fwr_trs/commit_decomposed_issue.json \
  --model minimax2.5

# 3. Build memory for BOTH
MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/back_trs/decomposed_issues.json \
  --output-dir data/back_trs \
  --model minimax2.5

MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/fwr_trs/commit_decomposed_issue.json \
  --output-dir data/fwr_trs \
  --model minimax2.5

# 4. Evaluate BOTH
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3 \
  --output results/backward

MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3 \
  --output results/forward
```

---

## 🔍 Verification

Check your setup:

```bash
# Verify split files
ls -lh data/*.jsonl data/*.json

# Verify backward decomposition outputs
ls -lh data/back_trs/

# Verify forward decomposition outputs (if created)
ls -lh data/fwr_trs/

# Check temporal safety
cat data/split_metadata.json | jq '.temporal_leakage_prevented'
# Should output: true

# Check split statistics
cat data/split_metadata.json | jq '{total: .total_issues, memory: .memory_size, eval: .eval_size}'
```

---

## 📊 Results

Results are saved to:

```
results/
├── miniswe-agent/
│   ├── minimax2.5/
│   │   ├── baseline/
│   │   ├── L1/
│   │   ├── L1+L2/
│   │   └── L1+L2+L3/
│   └── glm5.2/
│       ├── baseline/
│       ├── L1/
│       ├── L1+L2/
│       └── L1+L2+L3/
└── openhands/
    ├── minimax2.5/
    │   ├── baseline/
    │   ├── L1/
    │   ├── L1+L2/
    │   └── L1+L2+L3/
    └── glm5.2/
        ├── baseline/
        ├── L1/
        ├── L1+L2/
        └── L1+L2+L3/
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Wrong eval file format**: Use `data/eval_set.jsonl` (full records), NOT `data/eval_issue_ids.json` (IDs only)

2. **Old workflow**: Don't use `scripts/run_memory_split_workflow.sh` - it has temporal leakage. Use the new split-first workflow above.

3. **Missing memory files**: Make sure you ran Step 3 (Build L1/L2/L3) before evaluation

4. **Wrong memory root**:
   - For backward decomposition: `--memory-root data/back_trs`
   - For forward decomposition: `--memory-root data/fwr_trs`

5. **Temporal leakage**: Verify `data/split_metadata.json` shows `"temporal_leakage_prevented": true`

---

## 📖 Additional Documentation

- [DATA_ORGANIZATION.md](DATA_ORGANIZATION.md) - Complete data structure guide
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage guide
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [commit_decomposition/README_V2.md](commit_decomposition/README_V2.md) - Commit decomposition details

---

## ✅ Quick Checklist

Before running evaluation, ensure:

- [ ] Split created: `data/memory_set.jsonl` and `data/eval_set.jsonl` exist
- [ ] Temporal safety: `data/split_metadata.json` shows `temporal_leakage_prevented: true`
- [ ] Decomposition complete: Either `data/back_trs/decomposed_issues.json` or `data/fwr_trs/commit_decomposed_issue.json` exists
- [ ] Memory built: `data/back_trs/*_memory.json` or `data/fwr_trs/*_memory.json` files exist
- [ ] API keys configured in `.env`
- [ ] Correct memory root specified in evaluation command

---

**For questions or issues, see the documentation files or check the issue tracker.**
