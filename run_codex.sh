#!/bin/bash
# Simple wrapper to run Codex with any model
#
# Usage:
#   ./run_codex.sh <model> <ablation> <direction> <issue-ids>
#
# Examples:
#   ./run_codex.sh minimax baseline backward 125,126,127
#   ./run_codex.sh glm5.2 L1+L2+L3 backward 125
#   ./run_codex.sh minimax L1+L2+L3 forward 125,126

set -e

# Parse arguments
MODEL=${1:-minimax}
ABLATION=${2:-baseline}
DIRECTION=${3:-backward}
ISSUE_IDS=${4:-125}

echo "=========================================="
echo "Running Codex"
echo "=========================================="
echo "Model:     $MODEL"
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo "Issues:    $ISSUE_IDS"
echo ""

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✓ Loaded .env"
else
    echo "⚠ Warning: .env file not found"
fi

# Check Codex installed
if ! command -v codex &> /dev/null; then
    echo "✗ Codex CLI not found!"
    echo "Install with: uv tool install codex-cli"
    exit 1
fi

echo "✓ Codex CLI installed: $(codex --version)"

# Configure OpenRouter for Codex
if [ -n "$OPENROUTER_API_KEY" ]; then
    export OPENAI_API_BASE=https://openrouter.ai/api/v1
    export OPENAI_API_KEY=$OPENROUTER_API_KEY
    echo "✓ Configured OpenRouter API endpoint"
fi

echo ""

# Set memory root based on ablation and direction
MEMORY_ARGS=""
if [ "$ABLATION" != "baseline" ] && [ "$ABLATION" != "BASELINE" ]; then
    if [ "$DIRECTION" = "forward" ]; then
        MEMORY_ROOT="data/fwr_trs"
    else
        MEMORY_ROOT="data/back_trs"
    fi
    MEMORY_ARGS="--memory-root $MEMORY_ROOT --context-model $MODEL"
    echo "Using memory: $MEMORY_ROOT"
fi

# Map model names to OpenRouter format
case "$MODEL" in
    minimax)
        OPENROUTER_MODEL="openrouter/minimax/minimax-01"
        ;;
    glm5.2|glm-5.2)
        OPENROUTER_MODEL="openrouter/zhipuai/glm-4-plus"
        ;;
    glm4)
        OPENROUTER_MODEL="openrouter/zhipuai/glm-4-0520"
        ;;
    deepseek-chat|deepseek)
        OPENROUTER_MODEL="openrouter/deepseek/deepseek-chat"
        ;;
    gpt-4o)
        OPENROUTER_MODEL="openrouter/openai/gpt-4o"
        ;;
    openrouter/*)
        # Already in OpenRouter format
        OPENROUTER_MODEL="$MODEL"
        ;;
    *)
        # Default: assume it's a direct model name
        OPENROUTER_MODEL="$MODEL"
        ;;
esac

echo "Using OpenRouter model: $OPENROUTER_MODEL"
echo ""

# Build command
CMD="PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py"
CMD="$CMD --issue-ids $ISSUE_IDS"
CMD="$CMD --ablations $ABLATION"
CMD="$CMD $MEMORY_ARGS"
CMD="$CMD --codex-command 'codex exec --full-auto --model $OPENROUTER_MODEL'"

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
echo "Results in: results/codex/${ABLATION}_${MODEL}_${DIRECTION}/"
