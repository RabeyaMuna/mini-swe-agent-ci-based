#!/bin/bash
# Run Harbor with Codex agent and minimax
#
# Usage: ./run_harbor_codex.sh <issue-ids> [ablation] [direction]

set -e

ISSUE_IDS=${1:-125}
ABLATION=${2:-baseline}
DIRECTION=${3:-backward}

echo "=========================================="
echo "HARBOR + CODEX + MINIMAX"
echo "=========================================="
echo "Issues:    $ISSUE_IDS"
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo ""

# Load environment
[ -f .env ] && source .env

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "✗ OPENROUTER_API_KEY not set"
    exit 1
fi

# Set OpenAI environment for OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY

echo "✓ Using OpenRouter"
echo "  Base URL: $OPENAI_BASE_URL"
echo ""

# Convert dataset
echo "Converting issues to Harbor format..."
python3 convert_harbor_dataset.py "$ISSUE_IDS" "$ABLATION" "$DIRECTION"
echo ""

# Run Harbor
echo "Running Harbor with Codex agent..."
echo ""

harbor run \
    --path harbor_dataset \
    --agent codex \
    --model openrouter/minimax/minimax-01 \
    --jobs-dir "results/harbor/${ABLATION}_minimax_${DIRECTION}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS!"
    echo "Results: results/harbor/${ABLATION}_minimax_${DIRECTION}/"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
