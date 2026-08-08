# CI-Repair-Bench Evaluation Setup

This repository runs CI repair evaluations for two agents:

- `mini-swe-agent`

## LLM Provider Support

**The system supports ANY LLM provider** through [LiteLLM](https://docs.litellm.ai/):

- **OpenAI**: GPT-4o, GPT-4 Turbo, GPT-3.5
- **100+ providers**: See [LLM Configuration Guide](docs/LLM_CONFIGURATION.md)

Default evaluation models:

- `openrouter/minimax/minimax-m2.5`
- `openrouter/z-ai/glm-5.2`

Each model is run at four ablation levels:

- `baseline`: no memory
- `L1`: failure memory
- `L1+L2`: failure memory + repository memory
- `L1+L2+L3`: full memory

## Data Organization

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
./run_codex_direct.sh
```

To target a different model, ablation, or direction:

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

test -x "$MINI_PY" && echo "MINI_PY ok: $MINI_PY"
```

For a quick smoke test, append `--slice 0:5` to Mini-SWE-Agent commands and Codex commands.

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
- Agent repair runs (Mini-SWE-Agent and Codex)

**See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed model specifications.**

---

## Workflow: Split -> Decompose -> Build Memory -> Evaluate

### **NEW Workflow (Temporal Leakage Prevention)**

The correct workflow is now:

1. **Split FIRST** (chronological) -> creates `data/memory_set.jsonl` and `data/eval_set.jsonl`
2. **Decompose ONLY memory** -> creates `data/back_trs/decomposed_issues.json`
3. **Build L1/L2/L3 memory** -> creates memory files in `data/back_trs/`
4. **Evaluate** -> uses eval set + memory

This prevents temporal data leakage by ensuring only past issues are in memory.

---

## Step 1: Split Dataset (Chronological)

Use the root environment.

```bash
./run_codex_direct.sh "" baseline   backward gpt-5-mini
./run_codex_direct.sh "" L1         backward gpt-5-mini
./run_codex_direct.sh "" L2         backward gpt-5-mini
./run_codex_direct.sh "" L3         backward gpt-5-mini
./run_codex_direct.sh "" L1+L2      backward gpt-5-mini
./run_codex_direct.sh "" L1+L2+L3   backward gpt-5-mini
```

Forward (uses `data/fwr_trs`):

```bash
# All repos
MODEL=glm5.2 scripts/workflows/run_memory_decompositions.sh

# Specific repos (recommended)
MODEL=glm5.2 scripts/workflows/run_memory_decompositions.sh agno,flower,camel,crewAI

# Custom memory ratio (default 0.3 = 30%)
MEMORY_RATIO=0.2 MODEL=glm5.2 scripts/workflows/run_memory_decompositions.sh agno
```

**Output:**
```
data/memory_set.jsonl        ← Earliest 30% (for decomposition)
data/eval_set.jsonl          ← Latest 70% (for evaluation)
data/memory_issue_ids.json
data/eval_issue_ids.json
data/split_metadata.json     ← Temporal safety confirmation
data/back_trs/               ← Backward decomposition output
data/fwr_trs/                ← Forward decomposition output
```

---

## Step 2: Decompose Memory (Backward or Forward)

### Option A: Backward Decomposition (Default, Recommended)

```bash
# MiniMax-M2.5
MODEL=minimax2.5 python backward_decomposition/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model minimax2.5

# GLM-5.2
MODEL=glm5.2 python backward_decomposition/decompose_ci_failure.py \
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
# All ablations + both directions on all issues (default model gpt‑5‑mini)
./run_miniswe_direct.sh                    # default workers=1
./run_miniswe_direct.sh "" all both gpt-5-mini '' '' 4   # 4 workers

# MiniMax, full memory, backward on all issues
./run_miniswe_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5

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

### Codex CLI

Codex is a third evaluation agent that uses the Codex CLI tool.

> > Multi-model: This repo supports gpt-5-mini, gpt-5.4-mini-2026-03-17, and minimax2.5.

#### 1. Install Codex CLI

```bash
# Install Codex CLI globally
pip install codex-cli

# Verify installation
codex --version
```

For more installation options, see: https://github.com/anthropics/codex

#### 2. Baseline (No Memory)

Run all validation issues with unlimited problems per issue:

```bash
# MiniMax 2.5 (via OpenRouter)
export OPENROUTER_API_KEY="your-key-here"

python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model minimax2.5

# GLM 5.2 (via Z-AI)
export GLM_API_KEY="your-key-here"

python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model glm5.2
```

**Using other LLM providers:**

```bash

#### 3. With Memory (L1+L2+L3) - Backward Decomposition

```bash
# MiniMax 2.5
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model minimax2.5 \
  --memory-root data/back_trs

# GLM 5.2
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model glm5.2 \
  --memory-root data/back_trs
```

#### 4. With Memory (L1+L2+L3) - Forward Decomposition

```bash
# MiniMax 2.5
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model minimax2.5 \
  --memory-root data/fwr_trs

# GLM 5.2
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model glm5.2 \
  --memory-root data/fwr_trs
```

#### 5. Separate Codex Evaluation Runs

Run baseline, backward-memory, and forward-memory experiments separately. Store
each run directly under `results/codex/` with this naming convention:

```text
results/codex/<ablation>_<model>
results/codex/<ablation>_<model>_<decomposition>
```

Use the decomposition suffix only when memory is enabled.

Set the model once:

```bash
MODEL=minimax2.5
MODEL_NAME=minimax2.5
```

the `MODEL`, `MODEL_NAME`, and API key. Use `MODEL_NAME` as a filesystem-safe
label, for example `openrouter_minimax_m2_5` instead of
`openrouter/minimax/minimax-m2.5`.

##### Baseline: No Memory

Do not pass `--memory-root` for baseline. Results are saved under the model's
baseline directory:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model "${MODEL}" \
  --model-name "${MODEL_NAME}" \
  --output-root "results/codex/baseline_${MODEL_NAME}"
```

Combined predictions:

```text
results/codex/baseline_<model>/predictions.json
```

##### Backward Decomposition Memory

Use `data/back_trs` for backward-decomposed memory. Run the memory ablations one
at a time so each result root includes the ablation name:

```bash
for LEVEL in L1 L1+L2 L1+L2+L3; do
  SAFE_LEVEL=$(printf "%s" "${LEVEL}" | tr "[:upper:]+" "[:lower:]_")

  python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation BASELINE \
    --model gpt-5-mini \
    --direction backward \
    --workers 4

python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation L1+L2+L3 \
    --model gpt-5.4-mini-2026-03-17 \
    --direction backward \
    --workers 4
```

### Run Both Agents (Codex + Mini‑SWE)

Run both agents with one command:

```bash
./run_both_agents.sh "" all both gpt-5-mini           # Codex:1, Mini‑SWE:1
# Or with MiniMax across both agents
./run_both_agents.sh "" all both minimax/minimax-m2.5
```

## Selecting Issues

The runner accepts explicit IDs, a repo filter, or defaults to `data/eval_issue_ids.json`.

- Explicit IDs (comma‑separated):
  ```bash
  ./run_codex_direct.sh 129,130 baseline backward gpt-5-mini
  ```

- All IDs from `data/eval_issue_ids.json` (leave first arg empty):
  ```bash
  ./run_codex_direct.sh "" all both gpt-5-mini
  ```

#### 6. Single Issue Testing

Test a single issue before running the full evaluation:

```bash
# Test with issue 43
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --context-model minimax2.5 \
  --dry-run

# Run it (remove --dry-run)
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --context-model minimax2.5 \
  ```

#### Key Parameters

- **`--issue-ids-file`**: Path to JSON file containing issue IDs (default: `data/eval_issue_ids.json`)
- **`--issue-ids`**: Comma-separated issue IDs for testing specific issues
- **`--ablations`**: Comma-separated list: `baseline`, `L1`, `L1+L2`, `L1+L2+L3`
- **`--context-model`**: LLM model to use (supports 100+ providers via LiteLLM)
- **`--model-name`**: Model label saved in `predictions.json` (defaults to `--context-model`)
- **`--memory-root`**: Path to memory data (`data/back_trs` or `data/fwr_trs`)
- **`--output-root`**: Results directory; include model/decomposition names to keep runs separate
- **`--generate-missing-analysis`**: Auto-generate CI failure analysis for uncached issues; enabled by default
- **`--no-generate-missing-analysis`**: Disable generation and require cached analysis only
- **`--dry-run`**: Generate prompts without executing Codex

#### Output Structure

```
results/codex/
├── baseline_minimax2.5/
│   ├── predictions.json
│   └── <issue_id>/
│       ├── issue_document_problem_1.md
│       ├── codex_transcript_problem_1.txt
│       ├── patch.diff                   ← Unified issue fix after all problem prompts
│       └── result.json
├── l1_minimax2.5_backward/
│   ├── predictions.json
│   └── <issue_id>/
├── l1_l2_minimax2.5_backward/
│   ├── predictions.json
│   └── <issue_id>/
├── l1_l2_l3_minimax2.5_backward/
│   ├── predictions.json
│   └── <issue_id>/
├── l1_minimax2.5_forward/
│   ├── predictions.json
│   └── <issue_id>/
├── l1_l2_minimax2.5_forward/
│   ├── predictions.json
│   └── <issue_id>/
└── l1_l2_l3_minimax2.5_forward/
    ├── predictions.json
    └── <issue_id>/
```

#### Documentation

- **[Complete LLM Configuration Guide](docs/LLM_CONFIGURATION.md)** - All providers, API keys, advanced config
- **[Quick Start Guide](docs/QUICK_START_LLM.md)** - Common examples and troubleshooting
- **[CI Repair Runner](codex/docs/ci-repair-runner.md)** - Full runner documentation
- **[Example Scripts](examples/run_with_different_llms.sh)** - Executable examples

---

## Comparing Decomposition Approaches

You can compare backward vs. forward decomposition:

```bash
# 1. Split once, then run BOTH decompositions
MODEL=minimax2.5 scripts/workflows/run_memory_decompositions.sh agno,flower,camel

# 2. Build memory for BOTH
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

## Verification

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

## Results

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

## Troubleshooting

### Common Issues

1. **Wrong eval file format**: Use `data/eval_set.jsonl` (full records), NOT `data/eval_issue_ids.json` (IDs only)

2. **Old workflow**: Don't use `scripts/run_memory_split_workflow.sh` - it has temporal leakage. Use the new split-first workflow above.

3. **Missing memory files**: Make sure you ran Step 3 (Build L1/L2/L3) before evaluation

4. **Wrong memory root**:
   - For backward decomposition: `--memory-root data/back_trs`
   - For forward decomposition: `--memory-root data/fwr_trs`

5. **Temporal leakage**: Verify `data/split_metadata.json` shows `"temporal_leakage_prevented": true`

- GPT‑5‑mini, forward memory only across all issues:
  ```bash
  ./run_codex_direct.sh "" L1+L2+L3 forward gpt-5-mini
  ```

## Outputs

Each run writes a structured folder per ablation and model with:
- `checkout/` – repository at failing commit
- `ci_failure.json`, `ci_failure.md`
- `ci_verification.json`
- `memory_context.md` (for memory runs)
- `issue_document_problem_*.md` (prompts)
- `codex_transcript_problem_*.txt`
- `patch.diff`
- `result.json`

## Notes

- Local Codex state and keys live in `~/.codex`; repo‑local `codex/.codex-config/**` is ignored by git.
- Ensure `.env` has the right key for the model you run: `OPENAI_API_KEY` for GPT‑5, `OPENROUTER_API_KEY` for MiniMax.
- To smoke‑test prompt generation without calling Codex: add `--dry-run` to the Python command in `run_codex_direct.sh` (or run the script once with no network to see pre‑flight fail early).
