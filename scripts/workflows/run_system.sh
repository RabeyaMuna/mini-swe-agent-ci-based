#!/bin/bash
# run_system.sh - Intelligent workflow that handles existing data
#
# Usage:
#   bash scripts/workflows/run_system.sh [repo1,repo2,...]  # Specific repos
#   bash scripts/workflows/run_system.sh                    # All repos

set -e

# Configuration
REPOS=${1:-""}
OUTPUT_DIR="data/trs"
CACHE_DIR="data"
HF_DATASET="ci-benchmark-user/ci-repair-bench"
MEMORY_RATIO=0.3
MODEL="${MODEL:-}"

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL is required. Use MODEL=minimax2.5 or MODEL=glm5.2."
    exit 1
fi

echo "================================================================================"
echo "CI Memory System - Intelligent Workflow"
echo "================================================================================"
echo "Configuration:"
echo "  Repos: ${REPOS:-ALL}"
echo "  Output: $OUTPUT_DIR"
echo "  Cache: $CACHE_DIR"
echo "  Memory ratio: $MEMORY_RATIO"
echo "  Model: $MODEL"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# =============================================================================
# INTELLIGENT STEP DETECTION
# =============================================================================

NEEDS_STEP_1=false
NEEDS_STEP_2=false
NEEDS_STEP_3=false
NEEDS_STEP_4=false

echo "Checking existing data..."
echo ""

# Check Step 1: filtered_issues.jsonl
if [ ! -f "$OUTPUT_DIR/filtered_issues.jsonl" ]; then
    echo "[WARN] Step 1 needed: filtered_issues.jsonl not found"
    NEEDS_STEP_1=true
else
    echo "OK Step 1 done: filtered_issues.jsonl exists"
    # Count issues
    ISSUE_COUNT=$(wc -l < "$OUTPUT_DIR/filtered_issues.jsonl" | tr -d ' ')
    echo "  -> Contains $ISSUE_COUNT issues"
fi

# Check Step 2: decomposed_issues.json
DECOMP_COUNT=0
FILTERED_COUNT=0
if [ ! -f "$OUTPUT_DIR/decomposed_issues.json" ]; then
    echo "[WARN] Step 2 needed: decomposed_issues.json not found"
    NEEDS_STEP_2=true
else
    if command -v jq &> /dev/null; then
        DECOMP_COUNT=$(cat "$OUTPUT_DIR/decomposed_issues.json" | jq 'length' 2>/dev/null || echo "0")
    fi

    # Check if all filtered issues are decomposed
    if [ -f "$OUTPUT_DIR/filtered_issues.jsonl" ]; then
        FILTERED_COUNT=$(wc -l < "$OUTPUT_DIR/filtered_issues.jsonl" | tr -d ' ')
    fi

    if [ "$DECOMP_COUNT" -lt "$FILTERED_COUNT" ]; then
        echo "[WARN] Step 2 needed: Only $DECOMP_COUNT/$FILTERED_COUNT issues decomposed"
        NEEDS_STEP_2=true
    else
        echo "OK Step 2 done: decomposed_issues.json exists"
        echo "  -> Contains $DECOMP_COUNT decomposed issues (all done)"
    fi
fi

# Check Step 3: memory/eval split
# IMPORTANT: Re-run if Step 2 is needed (new decompositions)
if [ "$NEEDS_STEP_2" = true ]; then
    echo "[WARN] Step 3 needed: Will regenerate split after new decompositions"
    NEEDS_STEP_3=true
elif [ ! -f "$OUTPUT_DIR/memory_set.jsonl" ] || [ ! -f "$OUTPUT_DIR/eval_set.jsonl" ]; then
    echo "[WARN] Step 3 needed: memory_set.jsonl or eval_set.jsonl not found"
    NEEDS_STEP_3=true
else
    echo "OK Step 3 done: memory_set.jsonl and eval_set.jsonl exist"
    if command -v jq &> /dev/null && [ -f "$OUTPUT_DIR/split_metadata.json" ]; then
        cat "$OUTPUT_DIR/split_metadata.json" | jq -r '
            "  -> Memory: \(.memory_set_size) issues (\(.memory_ratio * 100 | floor)%)",
            "  -> Eval: \(.eval_set_size) issues"
        ' 2>/dev/null || true
    fi
fi

# Check Step 4: L1/L2/L3 memory
# IMPORTANT: Re-run if Step 3 is needed (new memory split)
if [ "$NEEDS_STEP_3" = true ]; then
    echo "[WARN] Step 4 needed: Will regenerate L1/L2/L3 after new memory split"
    NEEDS_STEP_4=true
elif [ ! -f "$OUTPUT_DIR/failure_memory.json" ] || [ ! -f "$OUTPUT_DIR/repo_memory.json" ] || [ ! -f "$OUTPUT_DIR/cross_memory.json" ]; then
    echo "[WARN] Step 4 needed: L1/L2/L3 memory files not found"
    NEEDS_STEP_4=true
else
    echo "OK Step 4 done: L1/L2/L3 memory files exist"
    if command -v jq &> /dev/null; then
        L1_COUNT=$(cat "$OUTPUT_DIR/failure_memory.json" | jq 'length' 2>/dev/null || echo "?")
        L2_COUNT=$(cat "$OUTPUT_DIR/repo_memory.json" | jq 'length' 2>/dev/null || echo "?")
        L3_COUNT=$(cat "$OUTPUT_DIR/cross_memory.json" | jq 'length' 2>/dev/null || echo "?")
        echo "  -> L1: $L1_COUNT problems"
        echo "  -> L2: $L2_COUNT sequences"
        echo "  -> L3: $L3_COUNT patterns"
    fi
fi

echo ""
echo "================================================================================"

# Check if everything is done
if [ "$NEEDS_STEP_1" = false ] && [ "$NEEDS_STEP_2" = false ] && [ "$NEEDS_STEP_3" = false ] && [ "$NEEDS_STEP_4" = false ]; then
    echo "OK ALL STEPS COMPLETE - System is ready!"
    echo ""
    echo "All memory files exist. Nothing to do."
    echo ""
    echo "To rebuild from scratch, delete the files and re-run:"
    echo "  rm -rf $OUTPUT_DIR/*"
    echo "  bash scripts/workflows/run_system.sh"
    echo ""
    exit 0
fi

echo "Steps needed:"
[ "$NEEDS_STEP_1" = true ] && echo "  - Step 1: Load issues"
[ "$NEEDS_STEP_2" = true ] && echo "  - Step 2: Decompose issues"
[ "$NEEDS_STEP_3" = true ] && echo "  - Step 3: Similarity & split"
[ "$NEEDS_STEP_4" = true ] && echo "  - Step 4: Build L1/L2/L3 memory"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# =============================================================================
# STEP 1: Load Issues (if needed)
# =============================================================================

if [ "$NEEDS_STEP_1" = true ]; then
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

# Save
Path('$OUTPUT_DIR').mkdir(parents=True, exist_ok=True)
output_file = Path('$OUTPUT_DIR') / 'filtered_issues.jsonl'
with open(output_file, 'w') as f:
    for issue in filtered_issues:
        f.write(json.dumps(issue) + '\n')

print(f'Saved {len(filtered_issues)} issues to: {output_file}')
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "OK Step 1 complete"
    else
        echo ""
        echo "FAIL Step 1 failed"
        exit 1
    fi
    echo ""
fi

# =============================================================================
# STEP 2: Decompose Issues (if needed)
# =============================================================================

if [ "$NEEDS_STEP_2" = true ]; then
    echo "================================================================================"
    echo "STEP 2: Decomposing issues"
    echo "================================================================================"

    # Check which issues are already decomposed
    if [ -f "$OUTPUT_DIR/decomposed_issues.json" ] && command -v python3 &> /dev/null; then
        echo "Checking which issues need decomposition..."

        MISSING_IDS=$(python3 -c "
import json
from pathlib import Path

# Load filtered issues
with open('$OUTPUT_DIR/filtered_issues.jsonl') as f:
    filtered_ids = {
        str(record.get('id') or record.get('instance_id') or record.get('issue_id') or '')
        for line in f
        if line.strip()
        for record in [json.loads(line)]
    }
    filtered_ids.discard('')

# Load decomposed issues
decomp_file = Path('$OUTPUT_DIR/decomposed_issues.json')
if decomp_file.exists():
    with open(decomp_file) as f:
        decomposed = json.load(f)
        decomposed_ids = {
            str(d.get('original_issue_id') or d.get('issue_id') or d.get('id') or d.get('instance_id') or '')
            for d in decomposed
        }
        decomposed_ids.discard('')
else:
    decomposed_ids = set()

# Find missing
missing = filtered_ids - decomposed_ids
if missing:
    print(','.join(sorted(missing, key=int)))
" 2>/dev/null)

        if [ -n "$MISSING_IDS" ]; then
            MISSING_COUNT=$(echo "$MISSING_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
            echo "  -> Found $MISSING_COUNT missing issues: $MISSING_IDS"
            echo "  -> Decomposing missing issues only..."
            echo ""

            # Decompose only missing issues
            for issue_id in $(echo "$MISSING_IDS" | tr ',' ' '); do
                echo "  Decomposing issue $issue_id..."
                python3 scripts/decompose_ci_failure.py \
                    --issue-id "$issue_id" \
                    --output-dir "$OUTPUT_DIR" \
                    --model "$MODEL" \
                    --skip-memory || echo "  Warning: Issue $issue_id failed"
            done
        else
            echo "  -> All issues already decomposed"
        fi
    else
        echo "Fresh decomposition of all issues..."
        echo "Uses --skip-memory (L1/L2/L3 will be built in Step 4)"
        echo ""

        python3 scripts/decompose_ci_failure.py \
            --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
            --output-dir "$OUTPUT_DIR" \
            --model "$MODEL" \
            --skip-memory
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo "OK Step 2 complete"
    else
        echo ""
        echo "FAIL Step 2 failed"
        exit 1
    fi
    echo ""
fi

# =============================================================================
# STEP 3: Similarity & Split (if needed)
# =============================================================================

if [ "$NEEDS_STEP_3" = true ]; then
    echo "================================================================================"
    echo "STEP 3: Computing similarity and splitting"
    echo "================================================================================"
    echo "This splits issues into 30% memory / 70% eval based on similarity."
    echo ""

    python3 scripts/prepare_memory_train_test_split.py \
        --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        --memory-ratio $MEMORY_RATIO \
        $([ -n "$REPOS" ] && echo "--repos $REPOS")

    if [ $? -eq 0 ]; then
        echo ""
        echo "OK Step 3 complete"
    else
        echo ""
        echo "FAIL Step 3 failed"
        exit 1
    fi
    echo ""
fi

# =============================================================================
# STEP 4: Build L1/L2/L3 Memory (if needed)
# =============================================================================

if [ "$NEEDS_STEP_4" = true ]; then
    echo "================================================================================"
    echo "STEP 4: Building L1/L2/L3 memory"
    echo "================================================================================"
    echo "This builds memory files from the memory set."
    echo "Reuses decomposition from Step 2 (cached)."
    echo ""

    python3 scripts/decompose_ci_failure.py \
        --dataset "$OUTPUT_DIR/memory_set.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL"

    if [ $? -eq 0 ]; then
        echo ""
        echo "OK Step 4 complete"
    else
        echo ""
        echo "FAIL Step 4 failed"
        exit 1
    fi
    echo ""
fi

# =============================================================================
# FINAL STATUS
# =============================================================================

echo "================================================================================"
echo "WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Files:"
echo "  OK filtered_issues.jsonl       - All issues"
echo "  OK decomposed_issues.json      - Decomposed problems"
echo "  OK memory_set.jsonl           - 30% memory"
echo "  OK eval_set.jsonl             - 70% eval"
echo "  OK failure_memory.json        - L1 file-level"
echo "  OK repo_memory.json           - L2 sequences"
echo "  OK cross_memory.json          - L3 patterns"
echo ""
echo "Cache directory: $CACHE_DIR/"
echo "  OK workflow_validation_cache.json"
echo "  OK log_details.json"
echo ""
echo "================================================================================"
echo "System is ready! Run evaluation:"
echo "  python scripts/run_eval.py --dataset $OUTPUT_DIR/eval_set.jsonl --memory-dir $OUTPUT_DIR"
echo "================================================================================"
