# Codex Setup & Run Guide

## What is Codex?

Codex is Anthropic's official CLI agent (from the Harbor framework) that can fix code issues autonomously. You want to use Codex to fix CI failures with different models (minimax, GLM, etc.).

---

## Setup (One Time)

### 1. Install Codex CLI

```bash
# Install via uv (recommended)
uv tool install codex-cli

# Or via pip
pip install codex-cli

# Verify installation
codex --version
```

### 2. Configure Model Access

Codex uses OpenRouter/Anthropic API by default. To use custom models (minimax, GLM), you need to configure API keys:

```bash
# Add to .env file
cat >> .env << 'EOF'
# OpenRouter (for minimax, GLM via OpenRouter)
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE

# Or direct API keys
MINIMAX_API_KEY=your_minimax_key_here
GLM_API_KEY=your_glm_key_here

# Anthropic (if using Claude models)
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
EOF
```

### 3. Install Project Dependencies

```bash
# Install Python dependencies
pip install -r requirements-shared.txt
pip install python-dotenv

# Add utilities to PYTHONPATH (needed by run_codex_ci_repair.py)
export PYTHONPATH=$PWD:$PYTHONPATH
```

---

## Running Codex

### Method 1: Using run_codex_ci_repair.py (Recommended)

This Python script handles everything: dataset loading, memory retrieval, and Codex execution.

```bash
# Basic usage with minimax
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations baseline \
    --codex-command "codex exec --full-auto --model minimax"

# With memory (L1+L2+L3)
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax"

# Using GLM 5.2
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model glm5.2 \
    --codex-command "codex exec --full-auto --model glm5.2"

# Multiple ablations
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax"
```

### Method 2: Direct Codex CLI (Manual)

For single-issue testing:

```bash
# Clone the failing repo
git clone https://github.com/adap/flower /tmp/test-flower
cd /tmp/test-flower
git checkout df0344c9  # failing commit

# Run Codex with minimax
codex exec --full-auto --model minimax \
    "Fix the CI failure: [paste error here]"

# Run with GLM 5.2
codex exec --full-auto --model glm5.2 \
    "Fix the CI failure: [paste error here]"
```

---

## Available Models

Codex supports models via OpenRouter or direct API:

### Via OpenRouter (Recommended)
Set `OPENROUTER_API_KEY` in .env, then use:
- `openrouter/minimax/minimax-m2.5` - Minimax
- `openrouter/zhipuai/glm-4-plus` - GLM 4+
- `openrouter/zhipuai/glm-5-2` - GLM 5.2
- `openrouter/deepseek/deepseek-chat` - DeepSeek

### Direct API
Set model-specific API keys in .env:
- `minimax` - Requires `MINIMAX_API_KEY`
- `glm5.2` - Requires `GLM_API_KEY`
- `gpt-4o` - Requires `OPENAI_API_KEY`

### Claude Models (Default)
- `sonnet` - Claude Sonnet (requires `ANTHROPIC_API_KEY`)
- `opus` - Claude Opus
- `haiku` - Claude Haiku

---

## Script Options

### `codex/scripts/run_codex_ci_repair.py` Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--issue-ids` | Comma-separated issue IDs | `125,126,127` |
| `--issue-ids-file` | File with issue IDs | `data/eval_issue_ids.json` |
| `--ablations` | Memory levels to test | `baseline`, `L1`, `L1+L2`, `L1+L2+L3` |
| `--memory-root` | Memory data directory | `data/back_trs` (backward) or `data/fwr_trs` (forward) |
| `--context-model` | Model for memory/analysis | `minimax`, `glm5.2` |
| `--codex-command` | Codex CLI command | `"codex exec --full-auto --model minimax"` |
| `--output` | Results directory | `results/codex/` (default) |
| `--workers` | Parallel workers | `4` |

### `codex exec` Options

| Option | Description |
|--------|-------------|
| `--full-auto` | Run without human approval (auto-execute) |
| `--model` | Model to use |
| `-C <dir>` | Working directory |
| `--search` | Enable web search |
| `-s <mode>` | Sandbox mode: `read-only`, `workspace-write`, `danger-full-access` |

---

## Examples

### Example 1: Single Issue, Baseline

```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125 \
    --ablations baseline \
    --codex-command "codex exec --full-auto --model minimax"
```

**Results:** `results/codex/baseline_minimax/125/`

### Example 2: Multiple Issues, Full Memory

```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127,128 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model glm5.2 \
    --codex-command "codex exec --full-auto --model glm5.2" \
    --workers 2
```

**Results:** `results/codex/l1_l2_l3_glm5_2/125/`, `126/`, etc.

### Example 3: All Ablations Comparison

```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126 \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax"
```

**Results:**
- `results/codex/baseline_minimax/125/`
- `results/codex/l1_minimax/125/`
- `results/codex/l1_l2_minimax/125/`
- `results/codex/l1_l2_l3_minimax/125/`

### Example 4: From Issue IDs File

```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids-file data/eval_issue_ids.json \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax" \
    --workers 4
```

---

## Results Structure

```
results/codex/
├── baseline_minimax/
│   ├── 125/
│   │   ├── patch.diff          # Generated fix
│   │   ├── ci_failure.json     # Analyzed CI failure
│   │   ├── ci_failure.md       # Human-readable CI failure
│   │   ├── codex_output.txt    # Codex execution log
│   │   └── report.json         # Full metadata
│   └── 126/
├── l1_l2_l3_minimax/
│   ├── 125/
│   └── 126/
└── l1_l2_l3_glm5_2/
    ├── 125/
    └── 126/
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'utilities'"
```bash
# Run with PYTHONPATH set
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py ...
```

### "codex: command not found"
```bash
# Install Codex CLI
uv tool install codex-cli

# Or via pip
pip install codex-cli
```

### "API key not found"
```bash
# Check .env file
cat .env | grep API_KEY

# Or set environment variable
export OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY
```

### Model not recognized
```bash
# For OpenRouter models, use full path:
codex exec --full-auto --model openrouter/minimax/minimax-m2.5

# Or configure model alias in utilities/model_registry.py
```

### Memory retrieval fails (PyTorch CUDA error)
```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --force-reinstall sentence-transformers
```

---

## Quick Reference

### Codex with Minimax
```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax"
```

### Codex with GLM 5.2
```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model glm5.2 \
    --codex-command "codex exec --full-auto --model glm5.2"
```

### Baseline (No Memory)
```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations baseline \
    --codex-command "codex exec --full-auto --model minimax"
```

---

## That's It!

The `codex/scripts/run_codex_ci_repair.py` script is your main entry point for running Codex with any model over CI repair benchmarks.

**Key Points:**
- ✅ Works with ANY model (minimax, GLM, DeepSeek, Claude, GPT-4)
- ✅ Supports memory augmentation (L1/L2/L3)
- ✅ Loads issues from HuggingFace automatically
- ✅ Saves results in organized directories
- ✅ Runs multiple issues in parallel

Just set your API keys and run!
