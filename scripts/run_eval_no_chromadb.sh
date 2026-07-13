#!/bin/bash
# Run evaluation WITHOUT ChromaDB (avoids mutex locks)
# Uninstalls ChromaDB temporarily to force JSON backend

set -e

echo "========================================"
echo "Evaluation Runner (NO ChromaDB)"
echo "========================================"
echo ""

# Parse arguments
ABLATION=${1:-"L1+L2+L3"}
WORKERS=${2:-1}
ISSUES=${3:-"data/trs/eval_issue_ids.json"}

echo "Configuration:"
echo "  Ablation: $ABLATION"
echo "  Workers: $WORKERS"
echo "  Issues: $ISSUES"
echo ""

# Clean up processes
echo "🧹 Cleaning up..."
pkill -9 -f "cibench" 2>/dev/null || true
pkill -9 -f "run_eval" 2>/dev/null || true
sleep 2

# Temporarily uninstall ChromaDB to force JSON backend
echo "📦 Uninstalling ChromaDB (will reinstall after)..."
pip uninstall -y chromadb 2>/dev/null || true

# Force JSON backend via environment
export MINI_SWE_AGENT_MEMORY_BACKEND=json
export MEMORY_BACKEND=json

echo ""
echo "✓ ChromaDB disabled"
echo "✓ Forced JSON backend"
echo ""

# Run evaluation
echo "========================================"
echo "RUNNING EVALUATION"
echo "========================================"
echo ""

python3 scripts/run_eval.py \
    --issue-ids-file "$ISSUES" \
    --ablation "$ABLATION" \
    --workers "$WORKERS"

EXIT_CODE=$?

# Reinstall ChromaDB
echo ""
echo "📦 Reinstalling ChromaDB..."
pip install chromadb 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ EVALUATION COMPLETE"
    echo "========================================"
    echo ""
    echo "Results: results/${ABLATION/+/_}/"
else
    echo ""
    echo "========================================"
    echo "✗ EVALUATION FAILED (exit code: $EXIT_CODE)"
    echo "========================================"
fi

exit $EXIT_CODE
