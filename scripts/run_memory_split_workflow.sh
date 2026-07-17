#!/bin/bash
# run_memory_split_workflow.sh
# Complete workflow for memory/eval split with decomposition
#
# Usage:
#   bash scripts/run_memory_split_workflow.sh agno,flower,camel  # Specific repos
#   bash scripts/run_memory_split_workflow.sh                    # All repos

set -e  # Exit on error

REPOS=${1:-""}  # Empty = all repos
OUTPUT_DIR="data/trs"
HF_DATASET="ci-benchmark-user/ci-repair-bench"
MEMORY_RATIO=0.3
MODEL="${MODEL:-${MEMCI_LLM_MODEL:-minimax2.5}}"

echo "======================================"
echo "Memory/Eval Split Workflow"
echo "======================================"
if [ -z "$REPOS" ]; then
    echo "Repos: ALL (no filter specified)"
else
    echo "Repos: $REPOS"
fi
echo "Output: $OUTPUT_DIR"
echo "Memory ratio: ${MEMORY_RATIO} (30%)"
echo "Model: $MODEL"
echo ""

# Step 1: Load and filter issues from HuggingFace
echo "Step 1: Loading issues from HuggingFace..."
python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Load from HuggingFace
print('Loading dataset from HuggingFace: $HF_DATASET')
ds = load_dataset('$HF_DATASET')
all_issues = [dict(item) for item in ds['train']]
print(f'Total issues loaded: {len(all_issues)}')

# Filter by repos (if specified)
repos_arg = '$REPOS'
if repos_arg:
    repos_filter = [r.strip().lower() for r in repos_arg.split(',')]
    print(f'Filtering by repos: {repos_filter}')

    filtered_issues = []
    for issue in all_issues:
        repo_name = issue.get('repo_name', '').lower()
        repo_owner = issue.get('repo_owner', '').lower()
        repo_key = f'{repo_owner}/{repo_name}'

        if any(filter_repo in repo_name or filter_repo in repo_key for filter_repo in repos_filter):
            filtered_issues.append(issue)

    print(f'Filtered issues: {len(filtered_issues)}')
else:
    print('No repo filter - using all issues')
    filtered_issues = all_issues

# Save filtered issues
Path('$OUTPUT_DIR').mkdir(parents=True, exist_ok=True)
output_file = Path('$OUTPUT_DIR') / 'filtered_issues.jsonl'
with open(output_file, 'w') as f:
    for issue in filtered_issues:
        f.write(json.dumps(issue) + '\n')

print(f'Saved to: {output_file}')
"

echo ""

# Step 2: Run decomposition on filtered issues
echo "Step 2: Running decomposition on filtered issues..."
echo "This will analyze each issue and extract atomic problems..."
echo ""

python scripts/decompose_ci_failure.py \
    --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --skip-memory

echo ""

# Step 3: Compute similarity and split
echo "Step 3: Computing similarity and splitting into memory/eval sets..."
echo ""

python scripts/prepare_memory_train_test_split.py \
    --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --memory-ratio $MEMORY_RATIO \
    --repos "$REPOS"

echo ""

# Step 4: Build L1/L2/L3 memory from memory issues
echo "Step 4: Building L1/L2/L3 memory from selected memory issues..."
echo "This runs the full L1/L2/L3 pipeline on the 30% memory issues..."
echo ""

python scripts/decompose_ci_failure.py \
    --dataset "$OUTPUT_DIR/memory_set.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL"

echo ""
echo "L1/L2/L3 memory generation complete!"

echo ""
echo "======================================"
echo "Workflow Complete!"
echo "======================================"
echo "Output files:"
echo ""
echo "Split datasets:"
echo "  - $OUTPUT_DIR/memory_set.jsonl (30% - most similar/representative)"
echo "  - $OUTPUT_DIR/memory_issue_ids.json (30% - IDs only)"
echo "  - $OUTPUT_DIR/eval_set.jsonl (70% - for evaluation)"
echo "  - $OUTPUT_DIR/eval_issue_ids.json (70% - IDs only)"
echo ""
echo "Memory files (L1/L2/L3):"
echo "  - $OUTPUT_DIR/failure_memory.json (L1 - file-level problems)"
echo "  - $OUTPUT_DIR/repo_memory.json (L2 - repair sequences)"
echo "  - $OUTPUT_DIR/cross_memory.json (L3 - cross-cutting patterns)"
echo ""
echo "Analysis:"
echo "  - $OUTPUT_DIR/decomposed_issues.json (all decomposed issues)"
echo "  - $OUTPUT_DIR/similarity_analysis.json (similarity stats)"
echo "  - $OUTPUT_DIR/split_metadata.json (split metadata)"
echo ""
echo "Next step:"
echo "  Run evaluation on eval_set.jsonl using the L1/L2/L3 memory files"
