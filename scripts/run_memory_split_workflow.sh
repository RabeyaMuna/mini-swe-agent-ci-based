#!/bin/bash
# run_memory_split_workflow.sh
# Complete workflow for memory/eval split with decomposition
#
# Usage:
#   bash scripts/run_memory_split_workflow.sh agno,flower,camel  # Specific repos
#   bash scripts/run_memory_split_workflow.sh                    # All repos

set -euo pipefail

REPOS=${1:-""}  # Empty = all repos
OUTPUT_DIR="${OUTPUT_DIR:-data/trs}"
HF_DATASET="ci-benchmark-user/ci-repair-bench"
MEMORY_RATIO="${MEMORY_RATIO:-0.3}"
MODEL="${MODEL:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL is required. Use MODEL=minimax2.5 or MODEL=glm5.2."
    exit 1
fi

echo "======================================"
echo "Memory/Eval Split Workflow"
echo "Stages: HuggingFace issues -> missing decompositions -> cosine split -> L1/L2/L3 memory"
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
FETCH_ARGS=(
    scripts/fetch_dataset.py
    --split train
    --out "$OUTPUT_DIR/filtered_issues.jsonl"
)
if [ -n "$REPOS" ]; then
    FETCH_ARGS+=(--repos "$REPOS")
fi
"$PYTHON_BIN" "${FETCH_ARGS[@]}"

echo ""

# Step 2: Run decomposition on filtered issues
echo "Step 2: Running decomposition on filtered issues..."
echo "This will analyze each issue and extract atomic problems..."
echo "Existing rows in $OUTPUT_DIR/decomposed_issues.json are reused; only missing issues are generated."
echo "Memory is intentionally skipped here; it is built after cosine split to avoid eval leakage."
echo ""

"$PYTHON_BIN" scripts/decompose_ci_failure.py \
    --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --skip-memory

echo ""

# Step 3: Compute similarity and split
echo "Step 3: Computing similarity and splitting into memory/eval sets..."
echo "This step intentionally runs every time so memory/eval IDs reflect the current decompositions."
echo ""

SPLIT_ARGS=(
    scripts/prepare_memory_train_test_split.py
    --dataset "$OUTPUT_DIR/filtered_issues.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --memory-ratio "$MEMORY_RATIO"
)
if [ -n "$REPOS" ]; then
    SPLIT_ARGS+=(--repos "$REPOS")
fi
"$PYTHON_BIN" "${SPLIT_ARGS[@]}"

echo ""

# Step 4: Build L1/L2/L3 memory from memory issues
echo "Step 4: Building L1/L2/L3 memory from selected memory issues..."
echo "This runs the full L1/L2/L3 pipeline on the 30% memory issues..."
echo "Existing decompositions are reused, so only missing memory issue decompositions are generated."
echo "The selected model controls decomposition chunk sizes and L1/L2/L3 limits."
echo ""

"$PYTHON_BIN" scripts/decompose_ci_failure.py \
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
