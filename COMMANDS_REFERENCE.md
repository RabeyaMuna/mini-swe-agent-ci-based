# Commands Reference - Complete Guide

## Quick Start

### Single Issue Test:

```bash
python3 scripts/run_eval.py \
    --issue-ids 125 \
    --ablation BASELINE \
    --model glm5.2 \
    --direction backward

# Results: results/minisweagent/baseline_glm5_2_backward/
```

### All Repos, All Modes:

```bash
MODEL=glm5.2 DIRECTION=backward ./run_all_repos_all_modes.sh

# Results: results/minisweagent/baseline_glm5_2_backward/
#          results/minisweagent/l1_glm5_2_backward/
#          results/minisweagent/l1_l2_glm5_2_backward/
#          results/minisweagent/l1_l2_l3_glm5_2_backward/
```

---

## Parameters

### Required:
- `--repos` OR `--issue-ids` OR `--issue-ids-file`

### Optional:
- `--ablation` - Memory level (default: `L1+L2+L3`)
  - `BASELINE` - No memory
  - `L1` - File-level memory
  - `L1+L2` - + Cross-workflow
  - `L1+L2+L3` - + Universal patterns

- `--model` - Model name (default: `minimax`)
  - `glm5.2`
  - `minimax`
  - `gpt-4`
  - Any other model

- `--direction` - Decomposition direction (default: `backward`)
  - `backward` - Backward decomposition
  - `forward` - Forward decomposition

- `--workers` - Parallel workers (default: `1`)

- `--exclude-memory` - Exclude memory seed issues

- `--dry-run` - Print command without executing

---

## Examples

### 1. Test Single Issue:

```bash
# BASELINE with GLM, backward
python3 scripts/run_eval.py \
    --issue-ids 125 \
    --ablation BASELINE \
    --model glm5.2 \
    --direction backward

# L1+L2+L3 with Minimax, forward
python3 scripts/run_eval.py \
    --issue-ids 125 \
    --ablation L1+L2+L3 \
    --model minimax \
    --direction forward
```

### 2. Multiple Issues:

```bash
python3 scripts/run_eval.py \
    --issue-ids 125,126,127 \
    --ablation L1+L2+L3 \
    --model glm5.2 \
    --direction backward \
    --workers 4
```

### 3. Specific Repos:

```bash
# All flower issues
python3 scripts/run_eval.py \
    --repos flower \
    --ablation L1+L2+L3 \
    --model glm5.2 \
    --direction backward \
    --exclude-memory

# Multiple repos
python3 scripts/run_eval.py \
    --repos crewai,flower,camel \
    --ablation BASELINE \
    --model minimax \
    --direction forward \
    --workers 8
```

### 4. From File:

```bash
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --model glm5.2 \
    --direction backward \
    --workers 4
```

---

## Scripts

### 1. All Repos, All Modes - Parallel:

```bash
# Default (minimax, backward)
./run_all_repos_all_modes.sh

# GLM, backward
MODEL=glm5.2 ./run_all_repos_all_modes.sh

# GLM, forward
MODEL=glm5.2 DIRECTION=forward ./run_all_repos_all_modes.sh

# Minimax, forward
MODEL=minimax DIRECTION=forward ./run_all_repos_all_modes.sh
```

### 2. Single Repo, All Modes:

```bash
# Syntax: ./run_single_repo_all_modes.sh <repo> <workers> <model> <direction>

# flower, defaults (4 workers, minimax, backward)
./run_single_repo_all_modes.sh flower

# flower, 8 workers, glm5.2, backward
./run_single_repo_all_modes.sh flower 8 glm5.2

# flower, 8 workers, glm5.2, forward
./run_single_repo_all_modes.sh flower 8 glm5.2 forward

# camel, 4 workers, minimax, forward
./run_single_repo_all_modes.sh camel 4 minimax forward
```

### 3. Sequential (Safer):

```bash
./run_sequential_all.sh
```

---

## Complete Evaluation Matrix

### Run All Combinations:

```bash
# 1. GLM, backward
MODEL=glm5.2 DIRECTION=backward ./run_all_repos_all_modes.sh

# 2. GLM, forward
MODEL=glm5.2 DIRECTION=forward ./run_all_repos_all_modes.sh

# 3. Minimax, backward
MODEL=minimax DIRECTION=backward ./run_all_repos_all_modes.sh

# 4. Minimax, forward
MODEL=minimax DIRECTION=forward ./run_all_repos_all_modes.sh
```

This creates 16 result directories:
- 4 ablations × 2 models × 2 directions = 16

---

## Results Structure

### Directory Format:

```
results/minisweagent/{ablation}_{model}_{direction}/
```

### Examples:

| Command | Results Directory |
|---------|------------------|
| `--ablation BASELINE --model glm5.2 --direction backward` | `baseline_glm5_2_backward/` |
| `--ablation L1 --model minimax --direction forward` | `l1_minimax_forward/` |
| `--ablation L1+L2+L3 --model gpt-4 --direction backward` | `l1_l2_l3_gpt_4_backward/` |

---

## Check Results

### Quick Summary:

```bash
./summarize_results.sh
```

### Detailed Check:

```bash
./check_results.sh
```

### Manual Check:

```bash
# List all results
ls -la results/minisweagent/

# Check specific run
cat results/minisweagent/baseline_glm5_2_backward/preds.json | jq '.'

# Count instances
find results/minisweagent -name "*.traj.json" | wc -l

# Check memory retrieval
cat results/minisweagent/l1_l2_l3_glm5_2_backward/*/df0344c9*.traj.json | \
    jq '.memory_retrieval.problems | length'
```

---

## Comparison Commands

### Compare Models:

```bash
# GLM vs Minimax (both backward)
diff \
    results/minisweagent/baseline_glm5_2_backward/preds.json \
    results/minisweagent/baseline_minimax_backward/preds.json
```

### Compare Directions:

```bash
# Backward vs Forward (same model)
diff \
    results/minisweagent/l1_l2_l3_glm5_2_backward/preds.json \
    results/minisweagent/l1_l2_l3_glm5_2_forward/preds.json
```

### Compare Ablations:

```bash
# Baseline vs L1+L2+L3
diff \
    results/minisweagent/baseline_glm5_2_backward/preds.json \
    results/minisweagent/l1_l2_l3_glm5_2_backward/preds.json
```

---

## Tips

### Dry Run First:

```bash
python3 scripts/run_eval.py \
    --repos crewai,flower,camel \
    --ablation L1+L2+L3 \
    --model glm5.2 \
    --direction backward \
    --dry-run
```

### Monitor Progress:

```bash
# Watch results directory
watch -n 5 'ls -lah results/minisweagent/*/ | head -30'

# Count completed
watch -n 10 'find results/minisweagent -name "*.traj.json" | wc -l'

# Check logs
tail -f logs/*.log
```

### Test Small First:

```bash
# Test single issue first
python3 scripts/run_eval.py --issue-ids 125 --ablation BASELINE --model glm5.2

# Then scale up
./run_all_repos_all_modes.sh
```

---

## Troubleshooting

### "Out of memory":

Reduce workers:
```bash
python3 scripts/run_eval.py --repos flower --ablation L1+L2+L3 --workers 2
```

Or use sequential:
```bash
./run_sequential_all.sh
```

### Check if running:

```bash
ps aux | grep "run_eval.py\|cibench"

# Kill if needed
pkill -f "run_eval.py"
```

### Clean old results:

```bash
# Remove old flat structure
rm -rf results/BASELINE results/L1 results/L1_L2 results/L1_L2_L3

# Keep only new structure
ls results/minisweagent/
```

---

## Summary

**Basic Template:**
```bash
python3 scripts/run_eval.py \
    --repos <repos> \
    --ablation <ablation> \
    --model <model> \
    --direction <direction> \
    --workers <N>
```

**Quick Scripts:**
```bash
MODEL=<model> DIRECTION=<direction> ./run_all_repos_all_modes.sh
./run_single_repo_all_modes.sh <repo> <workers> <model> <direction>
```

**Results:**
```
results/minisweagent/{ablation}_{model}_{direction}/
```

🚀 **Ready to run!**
