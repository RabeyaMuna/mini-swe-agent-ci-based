#!/bin/bash
# run_complete_workflow.sh
# Complete end-to-end workflow for memory-based CI repair system
#
# Usage:
#   bash scripts/run_complete_workflow.sh [repo1,repo2,...]  # Specific repos
#   bash scripts/run_complete_workflow.sh                    # All repos

set -e  # Exit on error

# Configuration
REPOS=${1:-""}  # Empty = all repos
OUTPUT_DIR="data/trs"
HF_DATASET="ci-benchmark-user/ci-repair-bench"
MEMORY_RATIO=0.3
MODEL="${MEMCI_LLM_MODEL:-openrouter/minimax/minimax-m2.5}"

echo "================================================================================"
echo "CI Memory System - Complete Workflow"
echo "================================================================================"
if [ -z "$REPOS" ]; then
    echo "Processing: ALL repos (no filter specified)"
else
    echo "Processing repos: $REPOS"
fi
echo "Output directory: $OUTPUT_DIR"
echo "Memory ratio: ${MEMORY_RATIO} (30% memory, 70% eval)"
echo "LLM Model: $MODEL"
echo ""
echo "This workflow will:"
echo "  1. Load issues from HuggingFace"
echo "  2. Decompose ALL issues (first pass)"
echo "  3. Compute similarity and split into memory/eval sets"
echo "  4. Build L1/L2/L3 memory from memory set"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# STEP 1: Load and Filter Issues from HuggingFace
# =============================================================================
echo "================================================================================"
echo "STEP 1: Loading issues from HuggingFace"
echo "================================================================================"

python3 -c "
from datasets import load_dataset
import json
from pathlib import Path

# Load from HuggingFace
print('Loading dataset from HuggingFace: $HF_DATASET')
ds = load_dataset('$HF_DATASET', verification_mode='no_checks')
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

print(f'Saved {len(filtered_issues)} issues to: {output_file}')
print('')
print('Breakdown by repo:')
from collections import Counter
repos = Counter()
for issue in filtered_issues:
    repo_owner = issue.get('repo_owner', 'unknown')
    repo_name = issue.get('repo_name', 'unknown')
    repos[f'{repo_owner}/{repo_name}'] += 1

for repo, count in repos.most_common():
    print(f'  {repo}: {count} issues')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load issues from HuggingFace"
    exit 1
fi

echo ""
echo "✓ Step 1 complete: Issues loaded and filtered"
echo ""

# =============================================================================
# STEP 2: Decompose ALL Issues (First Pass - No Memory Building)
# =============================================================================
echo "================================================================================"
echo "STEP 2: Decomposing ALL issues (first pass)"
echo "================================================================================"
echo "This will analyze each issue and extract atomic problems..."
echo "NOTE: We use --skip-memory flag to skip L1/L2/L3 generation at this stage"
echo "      (L1/L2/L3 will be built only for memory set in Step 4)"
echo ""

python3 scripts/decompose_ci_failure.py \
    --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --skip-memory

if [ $? -ne 0 ]; then
    echo "ERROR: Decomposition failed"
    exit 1
fi

echo ""
echo "✓ Step 2 complete: All issues decomposed"
echo "  Output: $OUTPUT_DIR/decomposed_issues.json"
echo ""

# =============================================================================
# STEP 3: Compute Similarity and Split (30% Memory, 70% Eval)
# =============================================================================
echo "================================================================================"
echo "STEP 3: Computing similarity and splitting into memory/eval sets"
echo "================================================================================"
echo "This will:"
echo "  - Compute embeddings using decomposed problems"
echo "  - Calculate cosine similarity within each repo"
echo "  - Rank by similarity (highest = most representative)"
echo "  - Select top 30% for memory set"
echo "  - Remaining 70% for evaluation set"
echo ""

python3 scripts/prepare_memory_train_test_split.py \
    --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --memory-ratio $MEMORY_RATIO \
    $([ -n "$REPOS" ] && echo "--repos $REPOS")

if [ $? -ne 0 ]; then
    echo "ERROR: Similarity analysis and split failed"
    exit 1
fi

echo ""
echo "✓ Step 3 complete: Issues split into memory/eval sets"
echo "  Memory set: $OUTPUT_DIR/memory_set.jsonl (30%)"
echo "  Eval set: $OUTPUT_DIR/eval_set.jsonl (70%)"
echo ""

# =============================================================================
# STEP 4: Build L1/L2/L3 Memory from Memory Set
# =============================================================================
echo "================================================================================"
echo "STEP 4: Building L1/L2/L3 memory from memory set"
echo "================================================================================"
echo "This will:"
echo "  - Reuse decomposition from Step 2"
echo "  - Build L1 (file-level problems)"
echo "  - Build L2 (repair sequences)"
echo "  - Build L3 (universal patterns)"
echo ""
echo "NOTE: This step runs the full L1/L2/L3 pipeline on the memory set only"
echo "      The decomposition is already done, so it will reuse that data"
echo ""

python3 scripts/decompose_ci_failure.py \
    --dataset "$OUTPUT_DIR/memory_set.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL"

if [ $? -ne 0 ]; then
    echo "ERROR: L1/L2/L3 memory generation failed"
    exit 1
fi

echo ""
echo "✓ Step 4 complete: L1/L2/L3 memory generated"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "================================================================================"
echo "WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Output files in: $OUTPUT_DIR/"
echo ""
echo "1. DATASETS:"
echo "   - filtered_issues.jsonl       : All issues (filtered by repo if specified)"
echo "   - memory_set.jsonl           : Memory set (30% - most representative)"
echo "   - memory_issue_ids.json      : Memory issue IDs only"
echo "   - eval_set.jsonl             : Evaluation set (70%)"
echo "   - eval_issue_ids.json        : Eval issue IDs only"
echo ""
echo "2. DECOMPOSITION:"
echo "   - decomposed_issues.json     : All issues decomposed into atomic problems"
echo ""
echo "3. SIMILARITY ANALYSIS:"
echo "   - similarity_analysis.json   : Similarity statistics per repo"
echo "   - split_metadata.json        : Split configuration and stats"
echo ""
echo "4. MEMORY FILES (L1/L2/L3):"
echo "   - failure_memory.json        : L1 - File-level problems"
echo "   - repo_memory.json           : L2 - Repair sequences per issue"
echo "   - cross_memory.json          : L3 - Universal patterns"
echo ""
echo "5. CACHES (for performance):"
echo "   - log_details.json           : CI log analysis cache"
echo "   - workflow_validation_cache.json : Workflow validation sequences"
echo ""
echo "================================================================================"
echo "NEXT STEPS:"
echo "================================================================================"
echo ""
echo "The system is now ready! You can:"
echo ""
echo "1. INSPECT THE MEMORY:"
echo "   cat $OUTPUT_DIR/failure_memory.json | jq '.[] | select(.repo == \"flower\")' | head"
echo "   cat $OUTPUT_DIR/repo_memory.json | jq '.[] | .problems'"
echo "   cat $OUTPUT_DIR/cross_memory.json | jq '.[] | .pattern_id'"
echo ""
echo "2. RUN EVALUATION on eval set using the memory:"
echo "   python scripts/run_eval.py \\"
echo "       --dataset $OUTPUT_DIR/eval_set.jsonl \\"
echo "       --memory-dir $OUTPUT_DIR \\"
echo "       --output results/eval_with_memory.jsonl"
echo ""
echo "3. COMPARE PERFORMANCE with/without memory:"
echo "   python scripts/compare_runs.py \\"
echo "       --with-memory results/eval_with_memory.jsonl \\"
echo "       --without-memory results/eval_baseline.jsonl"
echo ""
echo "================================================================================"
