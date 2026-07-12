# Command Reference

## Quick Commands

### Check Everything
```bash
# Validate setup (pre-flight checks)
bash scripts/validate_setup.sh

# Check current status
bash scripts/check_outputs.sh

# Check specific directory
bash scripts/check_outputs.sh experiments/run1
```

### Run Complete Workflow
```bash
# All repos (default) - ~20-40 hours
bash scripts/run_complete_workflow.sh

# Single repo - ~1-2 hours (RECOMMENDED for testing)
bash scripts/run_complete_workflow.sh flower

# Multiple repos - ~3-5 hours
bash scripts/run_complete_workflow.sh flower,agno,camel
```

## Individual Steps (Advanced)

### Step 1: Load Issues
```bash
# Load from HuggingFace
python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('ci-benchmark-user/ci-repair-bench', verification_mode='no_checks')
issues = [dict(item) for item in ds['train']]
with open('data/trs/filtered_issues.jsonl', 'w') as f:
    for issue in issues:
        f.write(json.dumps(issue) + '\n')
print(f'Saved {len(issues)} issues')
"
```

### Step 2: Decompose All Issues
```bash
# Decompose all (first pass - no L1/L2/L3)
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output-dir data/trs \
    --skip-memory

# Decompose single issue (for testing)
python3 scripts/decompose_ci_failure.py \
    --issue-id 206 \
    --skip-memory
```

### Step 3: Similarity & Split
```bash
# Default (30% memory, 70% eval)
python3 scripts/prepare_memory_train_test_split.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output-dir data/trs \
    --memory-ratio 0.3

# Filter by repo
python3 scripts/prepare_memory_train_test_split.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output-dir data/trs \
    --repos flower,agno

# Different split (40% memory, 60% eval)
python3 scripts/prepare_memory_train_test_split.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output-dir data/trs \
    --memory-ratio 0.4
```

### Step 4: Build L1/L2/L3 Memory
```bash
# Build memory from memory set
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs

# Build with different model
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs \
    --model openrouter/anthropic/claude-3-sonnet

# Limit number of issues (for testing)
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs \
    --limit 5
```

## Fix Missing L1/L2/L3 (If Step 4 Failed)

Your current situation:
```bash
# Check what's missing
bash scripts/check_outputs.sh

# Shows:
# ✓ Step 1-3 complete
# ✗ L1/L2/L3 missing

# Fix: Run Step 4
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs
```

## Inspect Results

### Check File Sizes
```bash
ls -lh data/trs/*.json data/trs/*.jsonl
```

### Count Entries
```bash
# Memory set issues
wc -l data/trs/memory_set.jsonl

# Eval set issues
wc -l data/trs/eval_set.jsonl

# L1 problems
cat data/trs/failure_memory.json | jq 'length'

# L2 sequences
cat data/trs/repo_memory.json | jq 'length'

# L3 patterns
cat data/trs/cross_memory.json | jq 'length'
```

### View Content
```bash
# Split statistics
cat data/trs/split_metadata.json | jq

# Similarity analysis
cat data/trs/similarity_analysis.json | jq

# L1: File problems
cat data/trs/failure_memory.json | jq '.[] | {file, problem}' | head

# L2: Repair sequences
cat data/trs/repo_memory.json | jq '.[] | {issue: .issue_id, steps: (.problems | length)}'

# L3: Patterns
cat data/trs/cross_memory.json | jq '.[] | {pattern: .pattern_id, type: .failure_type}'

# L3: Show one full pattern
cat data/trs/cross_memory.json | jq '.[0]'
```

### Search Content
```bash
# Find all mypy-related problems
cat data/trs/failure_memory.json | jq '.[] | select(.validation_cmd | contains("mypy"))'

# Find all formatting issues
cat data/trs/failure_memory.json | jq '.[] | select(.failure_type == "formatting")'

# Find issues for specific file
cat data/trs/failure_memory.json | jq '.[] | select(.file | contains("client.py"))'

# Find issues in specific repo
cat data/trs/failure_memory.json | jq '.[] | select(.repo == "flower")'
```

## Debugging

### Check Decomposition Status
```bash
# Count decomposed issues
cat data/trs/decomposed_issues.json | jq 'length'

# List decomposed issue IDs
cat data/trs/decomposed_issues.json | jq '.[] | .original_issue_id'

# Check specific issue
cat data/trs/decomposed_issues.json | jq '.[] | select(.original_issue_id == "206")'

# Check for errors
cat data/trs/decomposed_issues.json | jq '.[] | select(.error != null)'
```

### Check Memory Set
```bash
# Count memory issues
wc -l data/trs/memory_set.jsonl

# List memory issue IDs
cat data/trs/memory_issue_ids.json | jq

# View first memory issue
head -1 data/trs/memory_set.jsonl | jq
```

### Check Logs
```bash
# If you saved output to a log file
tail -100 workflow.log

# Check for errors
grep -i error workflow.log
grep -i fail workflow.log
```

### Re-run Failed Step
```bash
# Re-decompose specific issue
python3 scripts/decompose_ci_failure.py --issue-id 206

# Re-run Step 4 (memory building)
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs

# Force re-split (ignores cache)
rm data/trs/split_metadata.json
python3 scripts/prepare_memory_train_test_split.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output-dir data/trs
```

## Configuration

### Set LLM Model
```bash
# Default (MiniMax M2.5)
export MEMCI_LLM_MODEL="openrouter/minimax/minimax-m2.5"

# Claude Sonnet (more expensive, better quality)
export MEMCI_LLM_MODEL="openrouter/anthropic/claude-3-sonnet"

# GPT-4
export MEMCI_LLM_MODEL="openrouter/openai/gpt-4"

# Make persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export MEMCI_LLM_MODEL="openrouter/minimax/minimax-m2.5"' >> ~/.bashrc
```

### Set API Keys
```bash
# OpenRouter
export OPENROUTER_API_KEY="your-key-here"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# MiniMax direct
export MINIMAX_API_KEY="your-key-here"
export MINIMAX_BASE_URL="your-base-url"

# Make persistent
echo 'export OPENROUTER_API_KEY="your-key"' >> ~/.bashrc
```

### Change Memory Ratio
```bash
# Edit workflow script
vim scripts/run_complete_workflow.sh

# Change this line:
MEMORY_RATIO=0.3  # 30% memory, 70% eval

# Options:
# 0.2 → 20% memory, 80% eval (less memory, more eval data)
# 0.3 → 30% memory, 70% eval (default, balanced)
# 0.4 → 40% memory, 60% eval (more memory, less eval data)
```

## Performance Tips

### Speed Up Decomposition
```bash
# Use faster model
export MEMCI_LLM_MODEL="openrouter/minimax/minimax-m2.5"  # Fast

# Process fewer issues (testing)
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/filtered_issues.jsonl \
    --limit 10 \
    --skip-memory
```

### Reduce Memory Usage
```bash
# Process one issue at a time
for issue_id in $(cat data/trs/memory_issue_ids.json | jq -r '.[]'); do
    python3 scripts/decompose_ci_failure.py --issue-id $issue_id
    sleep 5  # Rate limiting
done
```

### Resume After Failure
```bash
# The system auto-resumes from cache
# Just re-run the same command

# Example: Step 2 failed at issue #50
# Re-run (will skip already processed issues)
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output-dir data/trs \
    --skip-memory
```

## Clean Up

### Remove Caches (Force Re-compute)
```bash
# Remove all caches
rm data/trs/decomposed_issues.json
rm data/trs/log_details.json
rm data/trs/workflow_validation_cache.json

# Remove split (force re-split)
rm data/trs/memory_set.jsonl
rm data/trs/eval_set.jsonl
rm data/trs/split_metadata.json

# Remove memory files (force rebuild)
rm data/trs/failure_memory.json
rm data/trs/repo_memory.json
rm data/trs/cross_memory.json
```

### Start Fresh
```bash
# Remove everything
rm -rf data/trs/*

# Re-run workflow
bash scripts/run_complete_workflow.sh flower
```

## Common Workflows

### Test with Single Repo
```bash
# 1. Validate
bash scripts/validate_setup.sh

# 2. Run workflow
bash scripts/run_complete_workflow.sh flower

# 3. Check outputs
bash scripts/check_outputs.sh

# 4. Inspect results
cat data/trs/failure_memory.json | jq '.[] | {file, problem}' | head
```

### Fix Missing L1/L2/L3
```bash
# Your current situation
# Steps 1-3 done, Step 4 missing

# Just run Step 4
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs

# Verify
bash scripts/check_outputs.sh
```

### Add More Repos
```bash
# 1. Load new issues
python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('ci-benchmark-user/ci-repair-bench')
issues = [dict(item) for item in ds['train'] if item.get('repo_name') in ['newrepo1', 'newrepo2']]
with open('data/trs/new_issues.jsonl', 'w') as f:
    for issue in issues:
        f.write(json.dumps(issue) + '\n')
"

# 2. Decompose new issues
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/new_issues.jsonl \
    --output-dir data/trs \
    --skip-memory

# 3. Combine with existing
cat data/trs/filtered_issues.jsonl data/trs/new_issues.jsonl > data/trs/all_issues.jsonl

# 4. Re-split
python3 scripts/prepare_memory_train_test_split.py \
    --dataset data/trs/all_issues.jsonl \
    --output-dir data/trs

# 5. Rebuild memory
python3 scripts/decompose_ci_failure.py \
    --dataset data/trs/memory_set.jsonl \
    --output-dir data/trs
```

### Compare Different Splits
```bash
# Try different memory ratios
for ratio in 0.2 0.3 0.4; do
    python3 scripts/prepare_memory_train_test_split.py \
        --dataset data/trs/filtered_issues.jsonl \
        --output-dir data/trs/split_${ratio} \
        --memory-ratio $ratio

    python3 scripts/decompose_ci_failure.py \
        --dataset data/trs/split_${ratio}/memory_set.jsonl \
        --output-dir data/trs/split_${ratio}
done

# Compare results
for ratio in 0.2 0.3 0.4; do
    echo "Ratio: $ratio"
    cat data/trs/split_${ratio}/split_metadata.json | jq
    echo ""
done
```

## Troubleshooting Commands

### "No module named 'X'"
```bash
pip3 install datasets litellm sentence-transformers scikit-learn typer rich numpy
```

### "API key not found"
```bash
# Check if set
echo $OPENROUTER_API_KEY

# Set it
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# Verify
python3 -c "import os; print('Key:', os.getenv('OPENROUTER_API_KEY', 'NOT SET'))"
```

### "File not found"
```bash
# Check what exists
ls -la data/trs/

# Create directory
mkdir -p data/trs

# Check specific file
if [ -f "data/trs/filtered_issues.jsonl" ]; then
    echo "File exists"
else
    echo "File missing - need to run Step 1"
fi
```

### "Empty output"
```bash
# Check file size
ls -lh data/trs/*.json

# Check content
cat data/trs/failure_memory.json | jq 'length'

# If 0 or empty, check logs
tail -100 workflow.log | grep -i error
```

## Getting Help

```bash
# Script help
python3 scripts/decompose_ci_failure.py --help
python3 scripts/prepare_memory_train_test_split.py --help

# Validate setup
bash scripts/validate_setup.sh

# Check outputs
bash scripts/check_outputs.sh

# View logs (if saved)
less workflow.log
```
