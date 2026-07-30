# Codex CI-Repair Quick Reference

## Installation

```bash
pip install codex-cli
codex --version
```

## Basic Commands

### Run All Validation Issues (Recommended)

```bash
# Set API key
export OPENROUTER_API_KEY="your-key-here"
MODEL=minimax2.5
MODEL_NAME=minimax2.5

# Baseline, no memory
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model "${MODEL}" \
  --model-name "${MODEL_NAME}" \
  --output-root "results/codex/baseline_${MODEL_NAME}"
```

### Test Single Issue

```bash
# Dry run (generate prompts only)
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --context-model minimax2.5 \
  --model-name minimax2.5 \
  --output-root results/codex/baseline_minimax2.5 \
  --dry-run

# Actual run
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --context-model minimax2.5 \
  --model-name minimax2.5 \
  --output-root results/codex/baseline_minimax2.5
```

### Run Multiple Issues

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43,75,121 \
  --ablations baseline \
  --context-model minimax2.5 \
  --model-name minimax2.5 \
  --output-root results/codex/baseline_minimax2.5
```

---

## Memory Ablations

### Baseline (No Memory)

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model minimax2.5 \
  --model-name minimax2.5 \
  --output-root results/codex/baseline_minimax2.5
```

### L1 Only (Failure Memory)

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1 \
  --context-model minimax2.5 \
  --memory-root data/back_trs \
  --model-name minimax2.5 \
  --output-root results/codex/l1_minimax2.5_backward
```

### L1+L2 (Failure + Repository Memory)

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2 \
  --context-model minimax2.5 \
  --memory-root data/back_trs \
  --model-name minimax2.5 \
  --output-root results/codex/l1_l2_minimax2.5_backward
```

### L1+L2+L3 (Full Memory)

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model minimax2.5 \
  --memory-root data/back_trs \
  --model-name minimax2.5 \
  --output-root results/codex/l1_l2_l3_minimax2.5_backward
```

### Backward Decomposition Memory, All Levels

```bash
for LEVEL in L1 L1+L2 L1+L2+L3; do
  SAFE_LEVEL=$(printf "%s" "${LEVEL}" | tr "[:upper:]+" "[:lower:]_")
  python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids-file data/eval_issue_ids.json \
    --ablations "${LEVEL}" \
    --context-model minimax2.5 \
    --model-name minimax2.5 \
    --memory-root data/back_trs \
    --output-root "results/codex/${SAFE_LEVEL}_minimax2.5_backward"
done
```

### Forward Decomposition Memory, All Levels

```bash
for LEVEL in L1 L1+L2 L1+L2+L3; do
  SAFE_LEVEL=$(printf "%s" "${LEVEL}" | tr "[:upper:]+" "[:lower:]_")
  python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids-file data/eval_issue_ids.json \
    --ablations "${LEVEL}" \
    --context-model minimax2.5 \
    --model-name minimax2.5 \
    --memory-root data/fwr_trs \
    --output-root "results/codex/${SAFE_LEVEL}_minimax2.5_forward"
done
```

---

## Different LLM Models

### MiniMax 2.5 (Default)

```bash
export OPENROUTER_API_KEY="your-key"
--context-model minimax2.5
```

### GLM 5.2

```bash
export GLM_API_KEY="your-key"
--context-model glm5.2
```

### Claude Sonnet 4.5

```bash
export ANTHROPIC_API_KEY="your-key"
--context-model claude-sonnet-4-5
```

### GPT-4o

```bash
export OPENAI_API_KEY="your-key"
--context-model gpt-4o
```

### Gemini 1.5 Pro

```bash
export GEMINI_API_KEY="your-key"
--context-model gemini/gemini-1.5-pro
```

### Ollama (Local, Free)

```bash
ollama pull llama3.1
--context-model ollama/llama3.1
```

**See [LLM Configuration Guide](../docs/LLM_CONFIGURATION.md) for 100+ more models**

---

## Key Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--issue-ids-file` | Path to issue IDs JSON file | `data/eval_issue_ids.json` | `data/eval_issue_ids.json` |
| `--issue-ids` | Comma-separated issue IDs | - | `43,75,121` |
| `--ablations` | Memory ablation levels | `baseline,L1,L1+L2,L1+L2+L3` | `baseline` or `L1+L2+L3` |
| `--context-model` | LLM model identifier | env: `CODEX_CONTEXT_MODEL` | `minimax2.5` |
| `--model-name` | Model label saved in predictions | `--context-model` or `codex` | `minimax2.5` |
| `--memory-root` | Memory data directory | `data/back_trs` | `data/fwr_trs` |
| `--generate-missing-analysis` | Auto-generate CI failure analysis | `True` | Usually omit; already enabled |
| `--no-generate-missing-analysis` | Require cached analysis only | `False` | (flag) |
| `--dry-run` | Generate prompts without running Codex | `False` | (flag) |
| `--output-root` | Results output directory | `results/codex` | `results/codex/l1_l2_l3_minimax2.5_backward` |
| `--timeout` | Codex execution timeout (seconds) | `3600` | `7200` |

---

## Output Files

For each issue, Codex generates:

```
results/codex/<ablation>_<model>[_<decomposition>]/<ablation>/<issue-id>/
├── ci_failure.json              # CI failure analysis
├── ci_failure.md                # Human-readable failure description
├── workflow_verification.json   # Workflow validation steps
├── issue_document_problem_1.md  # Codex prompt for problem 1
├── issue_document_problem_2.md  # Codex prompt for problem 2
├── issue_document_problem_N.md  # Codex prompt for problem N
├── codex_transcript_problem_1.txt  # Codex execution log
├── patch.diff                   # Generated fix (git diff)
└── result.json                  # Execution metadata
```

The runner also writes consolidated files:

```
results/codex/<ablation>_<model>[_<decomposition>]/predictions.json
results/codex/<ablation>_<model>[_<decomposition>]/<ablation>/predictions.json
```

`patch.diff` is captured once per issue after all problem prompts finish, so it
contains the unified diff for every change made while solving that issue.

---

## Common Workflows

### 1. Full Evaluation (Separate Result Roots)

```bash
export OPENROUTER_API_KEY="your-key"
MODEL=minimax2.5
MODEL_NAME=minimax2.5

# Baseline, no memory
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model "${MODEL}" \
  --model-name "${MODEL_NAME}" \
  --output-root "results/codex/baseline_${MODEL_NAME}"

# Backward memory
for LEVEL in L1 L1+L2 L1+L2+L3; do
  SAFE_LEVEL=$(printf "%s" "${LEVEL}" | tr "[:upper:]+" "[:lower:]_")
  python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids-file data/eval_issue_ids.json \
    --ablations "${LEVEL}" \
    --context-model "${MODEL}" \
    --model-name "${MODEL_NAME}" \
    --memory-root data/back_trs \
    --output-root "results/codex/${SAFE_LEVEL}_${MODEL_NAME}_backward"
done
```

### 2. Quick Test (Single Issue, Dry Run)

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --ablations baseline \
  --context-model minimax2.5 \
  --model-name minimax2.5 \
  --output-root results/codex/baseline_minimax2.5 \
  --dry-run

# Check generated prompt
cat results/codex/baseline_minimax2.5/baseline/43/issue_document_problem_1.md
```

### 3. Compare Forward vs Backward Decomposition

```bash
# Backward decomposition
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model minimax2.5 \
  --memory-root data/back_trs \
  --model-name minimax2.5 \
  --output-root results/codex/l1_l2_l3_minimax2.5_backward

# Forward decomposition
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations L1+L2+L3 \
  --context-model minimax2.5 \
  --memory-root data/fwr_trs \
  --model-name minimax2.5 \
  --output-root results/codex/l1_l2_l3_minimax2.5_forward
```

### 4. Compare Different Models

```bash
# MiniMax 2.5
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model minimax2.5 \
  --output-root results/minimax

# Claude Sonnet
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file data/eval_issue_ids.json \
  --ablations baseline \
  --context-model claude-sonnet-4-5 \
  --output-root results/claude

# Compare results
diff results/minimax/baseline/43/patch.diff \
     results/claude/baseline/43/patch.diff
```

---

## Troubleshooting

### Error: "No CI failure cache entry was found"

**Solution:** Add `--context-model`; missing analysis generation is enabled by default:

```bash
python3 codex/scripts/run_codex_ci_repair.py \
  --issue-ids 43 \
  --context-model minimax2.5
```

### Error: "Authentication failed"

**Solution:** Set the appropriate API key:

```bash
export OPENROUTER_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

### Check API key is set

```bash
echo $OPENROUTER_API_KEY
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

### Codex not found

**Solution:** Install Codex CLI:

```bash
pip install codex-cli
codex --version
```

---

## See Also

- **[Main README](../README.md)** - Full project setup and workflow
- **[LLM Configuration Guide](../docs/LLM_CONFIGURATION.md)** - All supported LLM providers
- **[Quick Start Guide](../docs/QUICK_START_LLM.md)** - Common examples
- **[CI Repair Runner Docs](docs/ci-repair-runner.md)** - Detailed documentation
