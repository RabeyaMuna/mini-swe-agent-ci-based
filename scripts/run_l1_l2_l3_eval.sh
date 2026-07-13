#!/bin/bash
# Run L1+L2+L3 evaluation with proper configuration
# This script avoids ChromaDB mutex locks by forcing JSON backend

set -e  # Exit on error

echo "========================================"
echo "L1+L2+L3 Evaluation Runner"
echo "========================================"
echo ""

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -9 -f "cibench" 2>/dev/null || true
pkill -9 -f "run_eval.py" 2>/dev/null || true
sleep 2

# Force JSON backend (no ChromaDB)
export MINI_SWE_AGENT_MEMORY_BACKEND=json

# Check memory files exist
echo "📁 Checking memory files..."
if [ ! -f "data/trs/failure_memory.json" ]; then
    echo "❌ ERROR: failure_memory.json not found!"
    echo "   Run: bash scripts/run_memory_split_workflow.sh"
    exit 1
fi

if [ ! -f "data/trs/repo_memory.json" ]; then
    echo "⚠️  WARNING: repo_memory.json not found!"
fi

if [ ! -f "data/trs/cross_memory.json" ]; then
    echo "⚠️  WARNING: cross_memory.json not found!"
fi

# Check eval issues
if [ ! -f "data/trs/eval_issue_ids.json" ]; then
    echo "❌ ERROR: eval_issue_ids.json not found!"
    exit 1
fi

ISSUE_COUNT=$(cat data/trs/eval_issue_ids.json | python3 -c "import json, sys; print(len(json.load(sys.stdin)))")
echo "✓ Found $ISSUE_COUNT test issues"
echo ""

# Parse arguments
WORKERS=${1:-1}
ISSUE_IDS_FILE=${2:-data/trs/eval_issue_ids.json}

echo "Configuration:"
echo "  Backend: JSON (no ChromaDB)"
echo "  Workers: $WORKERS"
echo "  Issues: $ISSUE_IDS_FILE"
echo "  Ablation: L1+L2+L3"
echo ""

# Estimate time
MINUTES=$((ISSUE_COUNT * 3 / WORKERS))
echo "⏱️  Estimated time: ~$MINUTES minutes"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "========================================"
echo "RUNNING EVALUATION"
echo "========================================"
echo ""

# Run evaluation
python3 scripts/run_eval.py \
    --issue-ids-file "$ISSUE_IDS_FILE" \
    --ablation L1+L2+L3 \
    --workers "$WORKERS"

echo ""
echo "========================================"
echo "EVALUATION COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: results/L1_L2_L3/"
echo ""
echo "Next steps:"
echo "  1. Check results: ls -lh results/L1_L2_L3/"
echo "  2. View predictions: cat results/L1_L2_L3/preds.json"
echo "  3. Evaluate metrics: python3 scripts/evaluate_preds.py results/L1_L2_L3/preds.json"
