# CI Failure Repair with Memory Plugin

Automated CI failure repair using Codex CLI with STaIR memory system.

## Quick Start

### One-Time Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv-codex

# 2. Activate environment
source .venv-codex/bin/activate

# 3. Install ALL Python dependencies
pip install -r requirements-codex.txt

# 4. Install Codex CLI
npm install -g @openai/codex

# 5. Add your API key to .env
echo "OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE" > .env

# 6. Setup Codex authentication
./update_codex_auth.sh
```

### Run Commands

```bash
# Activate environment
source .venv-codex/bin/activate

# Run single issue
./run_codex_direct.sh 129 baseline backward gpt-5-mini

# Run ALL 60 evaluation issues
./run_codex_direct.sh "" baseline backward gpt-5-mini
```

> **📋 See [COMPLETE_SETUP_SUMMARY.md](COMPLETE_SETUP_SUMMARY.md) for detailed installation and troubleshooting**

## Common Commands Cheat Sheet

| Task | Command |
|------|---------|
| **Single issue** | `./run_codex_direct.sh 129 baseline backward gpt-5-mini` |
| **ALL issues (baseline)** | `./run_codex_direct.sh "" baseline backward gpt-5-mini` |
| **ALL issues (with memory)** | `./run_codex_direct.sh "" L1+L2+L3 backward gpt-5-mini` |
| **ALL issues (forward)** | `./run_codex_direct.sh "" baseline forward gpt-5-mini` |
| **ALL ablations (single issue)** | `for a in baseline L1 L1+L2 L1+L2+L3; do ./run_codex_direct.sh 129 $a backward gpt-5-mini; done` |
| **ALL ablations (ALL issues)** | `for a in baseline L1 L1+L2 L1+L2+L3; do ./run_codex_direct.sh "" $a backward gpt-5-mini; done` |
| **Complete benchmark** | `for a in baseline L1 L1+L2 L1+L2+L3; do for d in backward forward; do ./run_codex_direct.sh "" $a $d gpt-5-mini; done; done` |

> 💡 **Empty string `""` = run ALL 60 evaluation issues**

## Supported Models

### OpenAI Models ✅

| Model | Command | Best For |
|-------|---------|----------|
| **GPT-5 Mini** | `gpt-5-mini` | Latest, fastest, most efficient |
| **GPT-4o** | `gpt-4o` | Fast GPT-4 |
| **GPT-4** | `gpt-4` | Most capable GPT-4 |
| **GPT-4 Turbo** | `gpt-4-turbo` | Fast, capable |
| **O1** | `o1` | Complex reasoning |
| **O1 Mini** | `o1-mini` | Fast reasoning |
| **O3 Mini** | `o3-mini` | Latest reasoning |

### Anthropic Models ✅

| Model | Command | Best For |
|-------|---------|----------|
| **Claude Opus 4** | `anthropic/claude-opus-4` | Most capable |
| **Claude Sonnet 4** | `anthropic/claude-sonnet-4` | Balanced |
| **Claude Sonnet 3.5** | `anthropic/claude-sonnet-3.5` | Fast, capable |

## Complete Command Reference

### Basic Usage

```bash
./run_codex_direct.sh <issue-ids> [ablation] [direction] [model]
```

**Parameters:**
- `issue-ids` - Issue ID(s) from dataset
  - Single: `129`
  - Multiple: `129,150,151`
  - **ALL issues**: `""` (empty string) - runs all 60 evaluation issues
- `ablation` - Memory mode (default: `baseline`)
  - `baseline` - No memory
  - `L1` - Failure-level memory only
  - `L1+L2` - Failure + repo-level memory
  - `L1+L2+L3` - All memory levels (cross-repo)
- `direction` - Decomposition direction (default: `backward`)
  - `backward` - Backward decomposition (failure to root cause)
  - `forward` - Forward decomposition (symptoms to fix)
- `model` - Model to use (default: `gpt-4o`)
  - OpenAI: `gpt-5-mini`, `gpt-4o`, `gpt-4`, `o1`, `o3-mini`
  - Anthropic: `anthropic/claude-opus-4`, `anthropic/claude-sonnet-4`

### Examples

#### OpenAI Models

```bash
# GPT-5 Mini (recommended - latest, fastest, most efficient)
./run_codex_direct.sh 129 baseline backward gpt-5-mini

# GPT-4o (fast GPT-4)
./run_codex_direct.sh 129 baseline backward gpt-4o

# GPT-4 (most capable)
./run_codex_direct.sh 129 baseline backward gpt-4

# O1 (complex reasoning)
./run_codex_direct.sh 129 baseline backward o1

# O3 Mini (latest reasoning model)
./run_codex_direct.sh 129 baseline backward o3-mini
```

#### Anthropic Models

```bash
# Claude Opus 4 (most capable)
./run_codex_direct.sh 129 baseline backward anthropic/claude-opus-4

# Claude Sonnet 4 (balanced)
./run_codex_direct.sh 129 baseline backward anthropic/claude-sonnet-4

# Claude Sonnet 3.5 (fast)
./run_codex_direct.sh 129 baseline backward anthropic/claude-sonnet-3.5
```

#### With Memory Plugin

```bash
# L1 memory (failure-level only)
./run_codex_direct.sh 129 L1 backward gpt-5-mini

# L1+L2 memory (failure + repo-level)
./run_codex_direct.sh 129 L1+L2 backward gpt-5-mini

# L1+L2+L3 memory (all levels)
./run_codex_direct.sh 129 L1+L2+L3 backward gpt-5-mini
```

#### Forward vs Backward Decomposition

```bash
# Backward decomposition (default - from failure to root cause)
./run_codex_direct.sh 129 baseline backward gpt-5-mini

# Forward decomposition (from symptoms to fix)
./run_codex_direct.sh 129 baseline forward gpt-5-mini
```

#### Multiple Issues & ALL Evaluation Issues

```bash
# Run on multiple specific issues
./run_codex_direct.sh 129,150,151 baseline backward gpt-5-mini

# Run ALL 60 evaluation issues (empty string = all issues)
./run_codex_direct.sh "" baseline backward gpt-5-mini

# Run ALL issues with memory
./run_codex_direct.sh "" L1+L2+L3 backward gpt-5-mini

# Run ALL issues with forward decomposition
./run_codex_direct.sh "" baseline forward gpt-5-mini
```

## Complete Setup Guide

### Prerequisites

- **Python 3.13+**: https://www.python.org/downloads/
- **Node.js & npm**: https://nodejs.org/
- **Git**: https://git-scm.com/

### Installation Steps

```bash
# 1. Create Python virtual environment
python3 -m venv .venv-codex

# 2. Activate environment
source .venv-codex/bin/activate

# 3. Install ALL Python dependencies
pip install -r requirements-codex.txt
```

**Installed packages:**
- Core: numpy, pandas, jsonlines, datasets
- LLM: langchain-openai, litellm, openai
- Memory: sentence-transformers, torch, scikit-learn, scipy
- Utilities: tqdm, pyyaml, python-dotenv, requests, demjson3

```bash
# 4. Install Codex CLI
npm install -g @openai/codex

# Verify installation
codex --version  # Should show: codex-cli 0.146.1
```

### API Keys Setup

Create `.env` file in project root:

```bash
# OpenAI API key (REQUIRED)
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE

# Anthropic API key (optional - for Claude models)
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
```

Then sync to Codex configuration:

```bash
./update_codex_auth.sh
```

## Dataset

### Evaluation Set

- **Total issues:** 60 CI failures from CI-bench
- **Source:** `data/eval_set.jsonl`
- **Issue IDs file:** `data/eval_issue_ids.json`

### Check Available Issues

```bash
# Count total issues
cat data/eval_issue_ids.json | python3 -c "import sys, json; print(f'Total: {len(json.load(sys.stdin))}')"

# List all issue IDs
cat data/eval_issue_ids.json | python3 -c "import sys, json; print('Issues:', json.load(sys.stdin))"
```

**Sample issue IDs:** 190, 185, 129, 81, 206, 87, 178, 188, 164, 172, ...

### Running Specific vs ALL Issues

```bash
# Single issue
./run_codex_direct.sh 129 baseline backward gpt-5-mini

# Multiple specific issues
./run_codex_direct.sh 129,185,190 baseline backward gpt-5-mini

# ALL 60 issues (empty string)
./run_codex_direct.sh "" baseline backward gpt-5-mini
```

## Results

Results are saved to:
```
results/codex/<ablation>/
  <issue-id>/
    checkout/                          # Cloned repository
    issue_document_problem_1.md        # Problem description
    codex_transcript_problem_1.txt     # Agent execution log
    patch.diff                         # Generated fix
    result.json                        # Metadata
  predictions.json                     # All predictions
```

View results:
```bash
# Find all patches
find results/codex/ -name "patch.diff"

# View specific patch
cat results/codex/baseline/129/patch.diff

# Check results summary
cat results/codex/baseline/predictions.json
```

## Complete Workflow Examples

### 1. Single Issue (Quick Test)

```bash
source .venv-codex/bin/activate
./run_codex_direct.sh 129 baseline backward gpt-5-mini
```

**Expected output:** Patch generated in `results/codex/baseline_gpt-5-mini/129/patch.diff`

### 2. ALL Evaluation Issues - Baseline Mode

```bash
# Run all 60 issues with baseline (no memory)
./run_codex_direct.sh "" baseline backward gpt-5-mini
```

**Expected output:** 60 directories in `results/codex/baseline_gpt-5-mini/*/`

### 3. ALL Evaluation Issues - With Memory

```bash
# Run all 60 issues with full memory (L1+L2+L3)
./run_codex_direct.sh "" L1+L2+L3 backward gpt-5-mini
```

**Expected output:** 60 directories in `results/codex/l1_l2_l3_gpt-5-mini/*/`

### 4. Ablation Study - Single Issue

```bash
# Compare all memory levels on one issue
for ablation in baseline L1 L1+L2 L1+L2+L3; do
  ./run_codex_direct.sh 129 $ablation backward gpt-5-mini
done
```

**Results locations:**
- `results/codex/baseline_gpt-5-mini/129/`
- `results/codex/l1_gpt-5-mini/129/`
- `results/codex/l1_l2_gpt-5-mini/129/`
- `results/codex/l1_l2_l3_gpt-5-mini/129/`

### 5. Ablation Study - ALL Issues

```bash
# Run ALL 60 issues with ALL ablations (baseline, L1, L1+L2, L1+L2+L3)
for ablation in baseline L1 L1+L2 L1+L2+L3; do
  ./run_codex_direct.sh "" $ablation backward gpt-5-mini
done
```

**Total runs:** 60 issues × 4 ablations = 240 runs

### 6. Forward vs Backward - ALL Issues

```bash
# Compare decomposition directions on ALL issues
for direction in backward forward; do
  ./run_codex_direct.sh "" baseline $direction gpt-5-mini
done
```

**Results:**
- Backward: `results/codex/baseline_gpt-5-mini/*/`
- Forward: Uses `data/fwr_trs/` memory instead of `data/back_trs/`

### 7. Complete Benchmark (ALL Issues × ALL Ablations × Both Directions)

```bash
# Full experiment: 60 issues × 4 ablations × 2 directions = 480 runs
for ablation in baseline L1 L1+L2 L1+L2+L3; do
  for direction in backward forward; do
    echo "Running: $ablation / $direction"
    ./run_codex_direct.sh "" $ablation $direction gpt-5-mini
  done
done
```

### 8. Model Comparison - ALL Issues

```bash
# Compare different models on all 60 issues
for model in gpt-5-mini gpt-4o gpt-4; do
  echo "Running with model: $model"
  ./run_codex_direct.sh "" baseline backward $model
done
```

**Results:**
- GPT-5 Mini: `results/codex/baseline_gpt-5-mini/*/`
- GPT-4o: `results/codex/baseline_gpt-4o/*/`
- GPT-4: `results/codex/baseline_gpt-4/*/`

## Troubleshooting

### Missing Dependency: sentence-transformers

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Fix:** Install all dependencies:
```bash
source .venv-codex/bin/activate
pip install -r requirements-codex.txt
```

Or install just this package:
```bash
pip install sentence-transformers>=2.7.0
```

**Note:** Required for memory modes (L1, L1+L2, L1+L2+L3). Baseline mode works without it.

### API Key Not Set

```
✗ OPENAI_API_KEY not set in .env
```

**Fix:**
```bash
echo "OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE" > .env
./update_codex_auth.sh
```

### Codex CLI Not Found

```
command not found: codex
```

**Fix:**
```bash
npm install -g @openai/codex
```

### Issue Not Found

```
Issue 125 not found in dataset
```

**Fix:** Use an issue from `data/eval_issue_ids.json`:
```bash
cat data/eval_issue_ids.json
```

### Empty Results (0 Problems)

If memory mode returns 0 problems, the system now automatically falls back to baseline mode and uses decomposed CI problems.

**Check logs for:**
```
[WARNING] Memory retrieval failed: ...
[WARNING] Falling back to baseline mode (no memory)
```

**To verify dependencies are installed:**
```bash
python3 -c "import sentence_transformers; print('✓ Dependencies OK')"
```

## Advanced Usage

### Custom Codex Command

Edit `codex/scripts/run_codex_ci_repair.py` line 1234:

```python
CODEX_REPAIR_COMMAND=codex exec --custom-flag

--codex-command "$CODEX_REPAIR_COMMAND"
```

### Add New Models

OpenAI-compatible models can be added by setting environment variables:

```bash
# Example: Using a custom API
export OPENAI_BASE_URL=https://your-api.com/v1
export OPENAI_API_KEY=your-key

./run_codex_direct.sh 129 baseline backward custom-model
```

## Monitoring Progress

### Watch Running Jobs

```bash
# Count completed issues in real-time
watch -n 5 'find results/codex/baseline_gpt-5-mini/ -name "patch.diff" | wc -l'

# List completed issues
find results/codex/baseline_gpt-5-mini/ -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort -n

# Check how many have patches
find results/codex/baseline_gpt-5-mini/ -name "result.json" -exec grep -l "patch_generated.*true" {} \; | wc -l
```

### View Results

```bash
# View specific patch
cat results/codex/baseline_gpt-5-mini/129/patch.diff

# View agent transcript
cat results/codex/baseline_gpt-5-mini/129/codex_transcript_problem_1.txt

# Check all predictions
cat results/codex/baseline_gpt-5-mini/predictions.json
```

## Performance Tips

1. **Use GPT-5 Mini** - Latest, fastest, and most cost-effective
2. **Start with baseline** - Test without memory first
3. **Test single issue first** - Verify setup before running ALL issues
4. **Monitor results** - Check `patch.diff` after each run
5. **Run in background** - For ALL issues, consider running in tmux/screen session

## Citation

If you use this code, please cite:

```bibtex
@article{stair2024,
  title={STaIR: Structured Memory for AI Repair},
  year={2024}
}
```

## Files

- `run_codex_direct.sh` - Main runner script
- `codex/scripts/run_codex_ci_repair.py` - Core repair logic
- `memory_plugin/` - STaIR memory system
- `data/eval_set.jsonl` - CI failure dataset
- `data/back_trs/` - Backward decomposition memory
- `data/fwr_trs/` - Forward decomposition memory

## License

[Your License]
