#!/bin/bash
# Universal CI Repair Script
# Usage: ./run_repair.sh <mode> <model> <issue-id>
#   mode: baseline, L1, L1+L2, L1+L2+L3
#   model: glm5.2, glm4, deepseek-chat, minimax, etc.
#   issue-id: Issue number to fix

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Parse arguments
MODE=${1:-baseline}
MODEL=${2:-glm5.2}
ISSUE_ID=${3:-125}

echo "======================================"
echo "CI Repair Configuration"
echo "======================================"
echo "Mode:     $MODE"
echo "Model:    $MODEL"
echo "Issue ID: $ISSUE_ID"
echo "======================================"

# Check if proxy is needed
if [ -z "$OPENAI_API_BASE" ] || [ "$OPENAI_API_BASE" != "http://localhost:8000/v1" ]; then
    echo ""
    echo "⚠️  WARNING: Codex proxy not configured!"
    echo "Make sure LiteLLM proxy is running:"
    echo "  Terminal 1: ./start_litellm_proxy.sh"
    echo ""
    echo "And configure Codex:"
    echo "  export OPENAI_API_BASE=http://localhost:8000/v1"
    echo "  export OPENAI_API_KEY=sk-dummy"
    echo ""
    echo "(Or add them to .env file)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build command
CMD="PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py"
CMD="$CMD --issue-ids $ISSUE_ID"
CMD="$CMD --ablations $MODE"
CMD="$CMD --context-model $MODEL"
CMD="$CMD --codex-command \"codex exec --full-auto --model $MODEL\""

# Add memory root for non-baseline modes
if [ "$MODE" != "baseline" ]; then
    CMD="$CMD --memory-root data/back_trs"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD
