# Decomposition Commands Reference

Quick reference for running both backward and forward decomposition approaches.

## 1. Backward Decomposition (CI → Problem)

**Location**: `backward_decomposition/`
**Output**: `data/back_trs/`
**Entry point**: `scripts/decompose_backward.py`

### Basic Commands

```bash
# Single issue
python scripts/decompose_backward.py \
  --issue-id <issue_id> \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/back_trs

# Batch (all issues)
python scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/back_trs

# Limited batch (first N issues)
python scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --limit 10 \
  --output-dir data/back_trs
```

### Auto-Split Mode (Recommended)

Automatically splits data into memory (30%) and eval (70%) sets per repo:

```bash
python scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --auto-split \
  --memory-ratio 0.3 \
  --output-dir data/back_trs
```

### Available Options

- `--issue-id <id>`: Process specific issue
- `--batch`: Process all issues
- `--use-huggingface`: Load from HuggingFace dataset
- `--model <model>`: LLM model (minimax2.5, glm5.2, etc.)
- `--limit <N>`: Process first N issues
- `--output-dir <path>`: Output directory (default: data/back_trs)
- `--skip-memory`: Skip L1/L2/L3 building (only save decomposed_issues.json)
- `--auto-split`: Enable automatic similarity-based train/test split
- `--memory-ratio <ratio>`: Memory set ratio for auto-split (default: 0.3)

---

## 2. Forward Decomposition (Commit → Problem)

**Location**: `commit_decomposition/`
**Output**: `data/fwr_trs/`
**Entry point**: `scripts/decompose_commits.py`

### Basic Commands

```bash
# Batch (all issues)
python scripts/decompose_commits.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/fwr_trs

# Limited batch (first N issues)
python scripts/decompose_commits.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --limit 10 \
  --output-dir data/fwr_trs
```

### Available Options

- `--batch`: Process all issues
- `--use-huggingface`: Load from HuggingFace dataset
- `--model <model>`: LLM model (minimax2.5, etc.)
- `--limit <N>`: Process first N issues
- `--output-dir <path>`: Output directory (default: data/fwr_trs)

---

## 3. Typical Workflows

### Quick Test (5 issues)

```bash
# Test backward approach
python scripts/decompose_backward.py \
  --batch --limit 5 \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/back_trs

# Test forward approach
python scripts/decompose_commits.py \
  --batch --limit 5 \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/fwr_trs
```

### Full Production Run

```bash
# Backward with auto-split
python scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --auto-split \
  --memory-ratio 0.3 \
  --output-dir data/back_trs

# Forward (all issues)
python scripts/decompose_commits.py \
  --batch \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/fwr_trs
```

### Debug Single Issue

```bash
# Backward
python scripts/decompose_backward.py \
  --issue-id agno-agi__agno-36 \
  --use-huggingface \
  --model minimax2.5 \
  --output-dir data/back_trs
```

---

## 4. Output Files

### Backward Output (`data/back_trs/`)

```
data/back_trs/
├── decomposed_issues.json     # All decomposed issues
├── failure_memory.json        # L1: Failure sequences
├── repo_memory.json           # L2: Repair strategies
└── cross_memory.json          # L3: Universal patterns
```

### Forward Output (`data/fwr_trs/`)

```
data/fwr_trs/
├── decomposed_issues.json     # All decomposed issues
├── failure_memory.json        # L1: Failure sequences
├── repo_memory.json           # L2: Repair strategies
└── cross_memory.json          # L3: Universal patterns
```

### Shared Cache (`data/`)

```
data/
├── log_details.json                  # CI log analysis cache
└── workflow_validation_cache.json    # Workflow validation cache
```

---

## 5. Checking Results

```bash
# List output files
ls -lh data/back_trs/
ls -lh data/fwr_trs/

# View issue summaries
cat data/back_trs/decomposed_issues.json | jq '.[] | {issue_id, total_problems}'
cat data/fwr_trs/decomposed_issues.json | jq '.[] | {issue_id, total_problems}'

# Count L1/L2/L3 entries
cat data/back_trs/failure_memory.json | jq '. | length'
cat data/back_trs/repo_memory.json | jq '. | length'
cat data/back_trs/cross_memory.json | jq '. | length'
```

---

## 6. Model Options

Supported models:
- `minimax2.5` (recommended)
- `glm5.2`
- Other HuggingFace models (check availability)

API options:
- `--use-huggingface`: HuggingFace API (default)
- `--use-openai`: OpenAI API
- `--use-anthropic`: Anthropic API

---

## 7. Architecture Overview

```
backward_decomposition/          ← Backward: CI → Problem
├── decompose_ci_failure.py

commit_decomposition/            ← Forward: Commit → Problem
├── commit_based_decomposer.py
├── commit_decomposition.py
└── ... (other commit modules)

build_memory/                    ← Shared L1/L2/L3 pipeline
├── build_l1.py
├── build_l2.py
└── build_l3.py

utilities/                       ← Shared utilities
├── ci_log_analyzer.py
├── ci_workflow_aware_retrieval.py
└── ci_cache.py

scripts/                         ← Entry points
├── decompose_backward.py        ← Backward wrapper
└── decompose_commits.py         ← Forward wrapper

data/
├── back_trs/                    ← Backward results
└── fwr_trs/                     ← Forward results
```

---

## 8. Troubleshooting

### Import errors

If you get `ModuleNotFoundError`:
- Make sure you're in the project root directory
- Check that `commit_decomposition/__init__.py` exists
- Check that `backward_decomposition/__init__.py` exists

### Cache issues

If results seem stale:
```bash
# Clear cache
rm data/log_details.json
rm data/workflow_validation_cache.json
```

### Memory issues

If decomposition completes but no L1/L2/L3 files:
- Check that `--skip-memory` is NOT set
- Verify LLM API is working
- Check logs for errors during memory building

---

For more details, see [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)
