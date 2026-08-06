#!/bin/bash
# Run Codex with any OpenAI or Anthropic model
#
# Usage: ./run_codex_direct.sh <issue-ids> [ablation] [direction] [model]
#   <issue-ids>: Issue ID(s) to run, or empty string "" for ALL issues
#   Examples:
#     ./run_codex_direct.sh 129 baseline backward gpt-4o      # Single issue
#     ./run_codex_direct.sh "" baseline backward gpt-4o       # ALL issues

set -e

# Handle empty string for "all issues"
if [ $# -eq 0 ]; then
    ISSUE_IDS="129"  # No args: default to 129
elif [ -z "$1" ]; then
    ISSUE_IDS=""     # Empty string: run ALL issues
else
    ISSUE_IDS="$1"   # Specific issue(s)
fi

ABLATION=${2:-baseline}
DIRECTION=${3:-backward}
MODEL=${4:-gpt-4o}  # Default: gpt-4o (has full metadata, no warnings)

echo "=========================================="
echo "CODEX CLI - Multi-Model Support"
echo "=========================================="
if [ -z "$ISSUE_IDS" ]; then
    echo "Issues:    ALL (from eval_set.jsonl or eval_issue_ids.json)"
else
    echo "Issues:    $ISSUE_IDS"
fi
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo "Model:     $MODEL"
echo ""

# Load .env BUT save specific keys first
[ -f .env ] && source .env
SAVED_OPENAI_KEY="$OPENAI_API_KEY"
SAVED_ANTHROPIC_KEY="$ANTHROPIC_API_KEY"

# Clear ALL API-related environment variables to prevent conflicts
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
unset OPENROUTER_API_KEY
unset OPENAI_BASE_URL

# Configure API based on model
case "$MODEL" in
    gpt-4o|gpt-4|gpt-4-turbo|o1|o1-mini|o3-mini|gpt-5-mini)
        # OpenAI models - use OpenAI API directly
        if [ -z "$SAVED_OPENAI_KEY" ]; then
            echo "✗ OPENAI_API_KEY not set in .env"
            exit 1
        fi
        # Set ONLY OpenAI key, everything else stays unset
        export OPENAI_API_KEY="$SAVED_OPENAI_KEY"
        CODEX_MODEL="$MODEL"
        CONTEXT_MODEL="$MODEL"
        echo "✓ Using OpenAI API"
        echo "  Model: $CODEX_MODEL"
        echo "  API Key: ${OPENAI_API_KEY:0:10}..."
        ;;

    anthropic/*)
        # Anthropic models - Codex supports them natively
        if [ -z "$SAVED_ANTHROPIC_KEY" ]; then
            echo "✗ ANTHROPIC_API_KEY not set in .env"
            exit 1
        fi
        # Set ONLY Anthropic key
        export ANTHROPIC_API_KEY="$SAVED_ANTHROPIC_KEY"
        CODEX_MODEL="$MODEL"
        CONTEXT_MODEL="claude-opus-4"
        echo "✓ Using Anthropic API"
        echo "  Model: $CODEX_MODEL"
        ;;

    *)
        echo "✗ Unknown model: $MODEL"
        echo ""
        echo "Supported models:"
        echo "  OpenAI:    gpt-4o, gpt-4, gpt-4-turbo, o1, o1-mini, o3-mini, gpt-5-mini"
        echo "  Anthropic: anthropic/claude-opus-4, anthropic/claude-sonnet-4, anthropic/claude-sonnet-3.5"
        exit 1
        ;;
esac

echo ""

# Memory args
MEMORY_ARGS=""
if [ "$ABLATION" != "baseline" ]; then
    MEMORY_ROOT="data/back_trs"
    [ "$DIRECTION" = "forward" ] && MEMORY_ROOT="data/fwr_trs"
    MEMORY_ARGS="--memory-root $MEMORY_ROOT --context-model $CONTEXT_MODEL"
fi

# Activate environment
source .venv-codex/bin/activate

# Run Codex
echo "Running Codex..."
echo ""

# Set Codex to use config inside codex directory
export CODEX_HOME="$(pwd)/codex/.codex-config"

# Build command - only pass --issue-ids if we have specific IDs
if [ -z "$ISSUE_IDS" ]; then
    # Empty: omit --issue-ids to use all issues from file
    PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
        --ablations "$ABLATION" \
        $MEMORY_ARGS \
        --codex-command "codex exec --sandbox danger-full-access --model $CODEX_MODEL"
else
    # Specific IDs: pass them explicitly
    PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
        --issue-ids "$ISSUE_IDS" \
        --ablations "$ABLATION" \
        $MEMORY_ARGS \
        --codex-command "codex exec --sandbox danger-full-access --model $CODEX_MODEL"
fi

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
