#!/bin/bash
# Run Codex directly with minimax via OpenRouter (NO PROXY NEEDED!)
#
# Usage: ./run_codex_direct.sh <issue-ids> [ablation] [direction]

set -e

ISSUE_IDS=${1:-125}
ABLATION=${2:-baseline}
DIRECTION=${3:-backward}

echo "=========================================="
echo "CODEX DIRECT (No Proxy)"
echo "=========================================="
echo "Issues:    $ISSUE_IDS"
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo ""

# Load .env
[ -f .env ] && source .env

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "✗ OPENROUTER_API_KEY not set in .env"
    exit 1
fi

# Set OpenAI environment variables to point to OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY

echo "✓ Using OpenRouter directly (no proxy)"
echo "  Base URL: $OPENAI_BASE_URL"
echo ""

# Memory args
MEMORY_ARGS=""
if [ "$ABLATION" != "baseline" ]; then
    MEMORY_ROOT="data/back_trs"
    [ "$DIRECTION" = "forward" ] && MEMORY_ROOT="data/fwr_trs"
    MEMORY_ARGS="--memory-root $MEMORY_ROOT --context-model minimax"
fi

# Activate environment
source .venv-codex/bin/activate

# Run Codex - model name must match OpenRouter format
echo "Running Codex with minimax..."
echo ""

PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids "$ISSUE_IDS" \
    --ablations "$ABLATION" \
    $MEMORY_ARGS \
    --codex-command "codex exec --sandbox workspace-write --model openrouter/minimax/minimax-01"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS!"
    echo "Results: results/codex/$ABLATION/"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
