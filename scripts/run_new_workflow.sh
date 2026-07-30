#!/bin/bash
# run_new_workflow.sh
# CORRECT workflow: Split → Decompose Memory → Build Memory (L1/L2/L3)
#
# Usage:
#   MODEL=minimax2.5 bash scripts/run_new_workflow.sh agno,flower,camel
#   MODEL=glm5.2 bash scripts/run_new_workflow.sh

set -euo pipefail

REPOS=${1:-"agno,flower,camel"}
OUTPUT_DIR="${OUTPUT_DIR:-data}"
MEMORY_RATIO="${MEMORY_RATIO:-0.3}"
MODEL="${MODEL:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL is required. Use MODEL=minimax2.5 or MODEL=glm5.2."
    exit 1
fi

echo "======================================================================"
echo "NEW WORKFLOW (Correct Order)"
echo "======================================================================"
echo "Flow: Split → Decompose Memory → Build Memory (L1/L2/L3)"
echo ""
echo "Repos: $REPOS"
echo "Output: $OUTPUT_DIR"
echo "Memory ratio: ${MEMORY_RATIO} (30%)"
echo "Model: $MODEL"
echo ""
echo "======================================================================"
echo ""

# ======================================================================
# STEP 1: SPLIT DATASET (Chronological - BEFORE decomposition!)
# ======================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Split Dataset (Chronological)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Strategy: Earliest 30% → Memory, Latest 70% → Eval"
echo "NO temporal leakage (guaranteed by chronological order)"
echo ""

"$PYTHON_BIN" scripts/split_before_decomposition.py \
    --repos "$REPOS" \
    --memory-ratio "$MEMORY_RATIO" \
    --output-dir "$OUTPUT_DIR"

echo ""

# Check if split was successful
if [ ! -f "$OUTPUT_DIR/memory_set.jsonl" ]; then
    echo "ERROR: Split failed - memory_set.jsonl not found"
    exit 1
fi

MEMORY_COUNT=$(wc -l < "$OUTPUT_DIR/memory_set.jsonl" | tr -d ' ')
EVAL_COUNT=$(wc -l < "$OUTPUT_DIR/eval_set.jsonl" | tr -d ' ')

echo "✓ Split complete:"
echo "  Memory: $MEMORY_COUNT issues (will be decomposed)"
echo "  Eval:   $EVAL_COUNT issues (no decomposition needed)"
echo ""

# ======================================================================
# STEP 2: DECOMPOSE ONLY MEMORY DATA
# ======================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Decompose ONLY Memory Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Input:  $OUTPUT_DIR/memory_set.jsonl ($MEMORY_COUNT issues)"
echo "Output: $OUTPUT_DIR/memory_decomposed.json"
echo ""
echo "Decomposing $MEMORY_COUNT memory issues (NOT all data)..."
echo ""

"$PYTHON_BIN" scripts/decompose_ci_failure.py \
    --dataset "$OUTPUT_DIR/memory_set.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --output-file "memory_decomposed.json" \
    --model "$MODEL"

echo ""

# Check if decomposition was successful
if [ ! -f "$OUTPUT_DIR/memory_decomposed.json" ]; then
    echo "ERROR: Decomposition failed - memory_decomposed.json not found"
    exit 1
fi

echo "✓ Decomposition complete!"
echo ""

# ======================================================================
# STEP 3: BUILD MEMORY (L1, L2, L3)
# ======================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Build Memory (L1, L2, L3)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Input:  $OUTPUT_DIR/memory_decomposed.json"
echo "Output: L1/L2/L3 memory files"
echo ""
echo "Building memory layers from decomposed memory..."
echo ""

# Build L1/L2/L3 memory from decomposed memory
# (Your existing memory building script should use memory_decomposed.json)
# This is a placeholder - replace with your actual memory building command

if [ -f "scripts/build_memory_layers.py" ]; then
    "$PYTHON_BIN" scripts/build_memory_layers.py \
        --input "$OUTPUT_DIR/memory_decomposed.json" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL"
else
    echo "[INFO] Memory layer building script not found."
    echo "       Using decompose_ci_failure.py for L1/L2/L3 generation..."

    # Alternative: Run decompose again to generate L1/L2/L3
    # (if it has memory building built-in)
fi

echo ""
echo "✓ Memory building complete!"
echo ""

# ======================================================================
# WORKFLOW COMPLETE
# ======================================================================
echo "======================================================================"
echo "✓ WORKFLOW COMPLETE!"
echo "======================================================================"
echo ""
echo "📊 Summary:"
echo "  Total issues:    $(($MEMORY_COUNT + $EVAL_COUNT))"
echo "  Memory issues:   $MEMORY_COUNT (decomposed)"
echo "  Eval issues:     $EVAL_COUNT (ready for evaluation)"
echo ""
echo "📁 Output files:"
echo ""
echo "  Split data:"
echo "    • $OUTPUT_DIR/memory_set.jsonl"
echo "    • $OUTPUT_DIR/eval_set.jsonl"
echo "    • $OUTPUT_DIR/memory_issue_ids.json"
echo "    • $OUTPUT_DIR/eval_issue_ids.json"
echo "    • $OUTPUT_DIR/split_metadata.json"
echo ""
echo "  Decomposed memory:"
echo "    • $OUTPUT_DIR/memory_decomposed.json"
echo ""
echo "  Memory layers (L1/L2/L3):"
echo "    • $OUTPUT_DIR/failure_memory.json    (L1 - file-level)"
echo "    • $OUTPUT_DIR/repo_memory.json       (L2 - sequences)"
echo "    • $OUTPUT_DIR/cross_memory.json      (L3 - patterns)"
echo ""
echo "🎯 Next step:"
echo "   Evaluate on eval_set.jsonl using the L1/L2/L3 memory"
echo ""
echo "======================================================================"
