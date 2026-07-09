#!/bin/bash
# run_memory_split_workflow.sh
# Complete workflow for memory/eval split with decomposition
#
# Usage:
#   bash scripts/run_memory_split_workflow.sh agno,flower,camel

set -e  # Exit on error

REPOS=${1:-"agno,flower,camel"}
OUTPUT_DIR="data/trs"
HF_DATASET="ci-benchmark-user/ci-repair-bench"
MEMORY_RATIO=0.3

echo "======================================"
echo "Memory/Eval Split Workflow"
echo "======================================"
echo "Repos: $REPOS"
echo "Output: $OUTPUT_DIR"
echo "Memory ratio: ${MEMORY_RATIO} (30%)"
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

# Filter by repos
repos_filter = [r.strip().lower() for r in '$REPOS'.split(',')]
print(f'Filtering by repos: {repos_filter}')

filtered_issues = []
for issue in all_issues:
    repo_name = issue.get('repo_name', '').lower()
    repo_owner = issue.get('repo_owner', '').lower()
    repo_key = f'{repo_owner}/{repo_name}'

    if any(filter_repo in repo_name or filter_repo in repo_key for filter_repo in repos_filter):
        filtered_issues.append(issue)

print(f'Filtered issues: {len(filtered_issues)}')

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
    --output "$OUTPUT_DIR/decomposed_issues.json" \
    --batch

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
echo "======================================"
echo "Workflow Complete!"
echo "======================================"
echo "Output files:"
echo "  - $OUTPUT_DIR/memory_set.jsonl (30% - most similar/representative)"
echo "  - $OUTPUT_DIR/eval_set.jsonl (70% - for evaluation)"
echo "  - $OUTPUT_DIR/similarity_analysis.json (analysis details)"
echo ""
echo "Next steps:"
echo "  1. Run agent on memory_set.jsonl to generate repair trajectories"
echo "  2. Convert trajectories to L2 memory format"
echo "  3. Evaluate agent on eval_set.jsonl using L2 memory"
