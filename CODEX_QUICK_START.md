# Codex Quick Start

## Setup (One Time)

```bash
# 1. Install Codex CLI
uv tool install codex-cli

# 2. Install Python dependencies
pip install -r requirements-shared.txt

# 3. Add API key to .env
echo "OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY" >> .env
```

## Run Codex

### Simple Wrapper (Easiest)

```bash
./run_codex.sh <model> <ablation> <issue-ids>
```

**Examples:**
```bash
# Minimax, baseline
./run_codex.sh minimax baseline 125,126,127

# GLM 5.2, full memory
./run_codex.sh glm5.2 L1+L2+L3 125

# DeepSeek, L1+L2 memory
./run_codex.sh deepseek-chat L1+L2 125,126
```

### Direct Python Script (More Control)

```bash
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax"
```

## Available Options

| Model | Ablation | Description |
|-------|----------|-------------|
| `minimax` | `baseline` | No memory |
| `glm5.2` | `L1` | File-level memory |
| `glm4` | `L1+L2` | File + repo memory |
| `deepseek-chat` | `L1+L2+L3` | Full memory |
| `gpt-4o` | | |

## Results

Results are saved to:
```
results/codex/{ablation}_{model}/{issue_id}/
```

Example:
```
results/codex/l1_l2_l3_minimax/125/patch.diff
results/codex/baseline_glm5_2/126/patch.diff
```

## Full Documentation

See **CODEX_SETUP_GUIDE.md** for complete documentation, troubleshooting, and advanced usage.

---

## Quick Examples

```bash
# Run single issue with minimax
./run_codex.sh minimax baseline 125

# Run multiple issues with GLM 5.2
./run_codex.sh glm5.2 L1+L2+L3 125,126,127

# Run from issue IDs file
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids-file data/eval_issue_ids.json \
    --ablations L1+L2+L3 \
    --context-model minimax \
    --codex-command "codex exec --full-auto --model minimax"
```

That's it! 🚀
