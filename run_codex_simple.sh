#!/bin/bash
# Simple wrapper for running Codex with any model
#
# Usage:
#   ./run_codex_simple.sh <model> <ablation> <direction> <issue-ids>
#
# Examples:
#   ./run_codex_simple.sh glm5.2 baseline backward 125,126,127
#   ./run_codex_simple.sh minimax L1+L2+L3 forward 125

set -e

MODEL=${1:-glm5.2}
ABLATION=${2:-baseline}
DIRECTION=${3:-backward}
ISSUE_IDS=${4:-125}

echo "=========================================="
echo "Running Codex"
echo "=========================================="
echo "Model: $MODEL"
echo "Ablation: $ABLATION"
echo "Direction: $DIRECTION"
echo "Issues: $ISSUE_IDS"
echo ""

# Load .env
if [ -f .env ]; then
    source .env
fi

# Check proxy
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Starting LiteLLM proxy..."
    ./start_litellm_proxy.sh &
    sleep 3
fi

echo "✓ LiteLLM proxy running"
echo ""

# Set memory root based on direction
if [ "$DIRECTION" = "backward" ]; then
    MEMORY_ROOT="data/back_trs"
else
    MEMORY_ROOT="data/fwr_trs"
fi

# Build command
CMD="PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py"
CMD="$CMD --issue-ids $ISSUE_IDS"
CMD="$CMD --ablations $ABLATION"
CMD="$CMD --codex-command 'codex exec --full-auto --model $MODEL'"

# Add memory if not baseline
if [ "$ABLATION" != "baseline" ] && [ "$ABLATION" != "BASELINE" ]; then
    CMD="$CMD --memory-root $MEMORY_ROOT"
fi

echo "Running:"
echo "$CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "=========================================="
echo "✓ Complete!"
echo "=========================================="
echo ""
echo "Results in: results/codex/${ABLATION}_${MODEL}/"
