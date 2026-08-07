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
SAVED_OPENROUTER_KEY="$OPENROUTER_API_KEY"
SAVED_MINIMAX_KEY="$MINIMAX_API_KEY"

# Clear ALL API-related environment variables to prevent conflicts
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
unset OPENROUTER_API_KEY
unset MINIMAX_API_KEY
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

    minimax*|minimax/*)
        # MiniMax models (via OpenAI-compatible API or OpenRouter)
        # Check if MiniMax API key is set (direct), otherwise use OpenRouter
        if [ -n "$SAVED_MINIMAX_KEY" ]; then
            # Direct MiniMax API (OpenAI-compatible)
            export OPENAI_API_KEY="$SAVED_MINIMAX_KEY"
            export OPENAI_BASE_URL="https://api.minimax.chat/v1"
            CODEX_MODEL="$MODEL"
            CONTEXT_MODEL="$MODEL"
            echo "✓ Using MiniMax API (direct)"
            echo "  Model: $CODEX_MODEL"
            echo "  API Key: ${OPENAI_API_KEY:0:10}..."
        elif [ -n "$SAVED_OPENROUTER_KEY" ]; then
            # Via OpenRouter
            export OPENAI_API_KEY="$SAVED_OPENROUTER_KEY"
            export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
            # OpenRouter recommended headers (as env vars for some clients)
            export OPENROUTER_APP_NAME="Codex CI Repair"
            export OPENROUTER_SITE_URL="https://github.com/anthropics/codex"
            CODEX_MODEL="$MODEL"
            CONTEXT_MODEL="$MODEL"
            echo "✓ Using OpenRouter for MiniMax"
            echo "  Model: $CODEX_MODEL"
            echo "  API Key: ${OPENAI_API_KEY:0:10}..."
        else
            echo "✗ No MiniMax API key found"
            echo "  Set either MINIMAX_API_KEY or OPENROUTER_API_KEY in .env"
            exit 1
        fi
        ;;

    *)
        echo "✗ Unknown model: $MODEL"
        echo ""
        echo "Supported models:"
        echo "  OpenAI:    gpt-4o, gpt-4, gpt-4-turbo, o1, o1-mini, o3-mini, gpt-5-mini"
        echo "  Anthropic: anthropic/claude-opus-4, anthropic/claude-sonnet-4, anthropic/claude-sonnet-3.5"
        echo "  MiniMax:   minimax-2.7, minimax-pro, minimax-* (via direct API or OpenRouter)"
        exit 1
        ;;
esac

echo ""

# Memory args
MEMORY_ARGS=""
if [ "$ABLATION" != "baseline" ]; then
    MEMORY_ROOT="data/back_trs"
    [ "$DIRECTION" = "forward" ] && MEMORY_ROOT="data/fwr_trs"
    MEMORY_ARGS="--memory-root $MEMORY_ROOT"
fi

# ALWAYS include context-model to separate results by model
MEMORY_ARGS="$MEMORY_ARGS --context-model $CODEX_MODEL"

# Activate environment
source .venv-codex/bin/activate

# Export CODEX_MODEL so the Python test can see it
export CODEX_MODEL

# PRE-FLIGHT CHECK: Verify the specified model actually works
echo "=========================================="
echo "PRE-FLIGHT CHECK: Testing model..."
echo "=========================================="

python3 - <<'EOF'
import os
import sys
from openai import OpenAI

# Get the configured API settings from environment
api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
model = os.getenv('CODEX_MODEL')

if not api_key:
    print("✗ FATAL: No API key found in environment")
    sys.exit(1)

if not model:
    print("✗ FATAL: CODEX_MODEL not set")
    sys.exit(1)

print(f"Testing model: {model}")
print(f"API endpoint: {base_url}")
print(f"API key: {api_key[:15]}...")
print("")

try:
    # For Anthropic models, skip this test (different API)
    if model.startswith('anthropic/'):
        print("✓ Anthropic model detected - skipping OpenAI API test")
        sys.exit(0)

    # Test with OpenAI-compatible API
    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Respond with only: OK"}],
        max_tokens=10
    )

    actual_model = response.model
    response_text = response.choices[0].message.content or ""

    print(f"✓ Model responded successfully!")
    print(f"  Requested: {model}")
    print(f"  Actual:    {actual_model}")
    print(f"  Response:  {response_text[:50]}")

    # Verify we got the requested model (allow version suffixes for OpenAI models)
    # e.g., "gpt-4o" -> "gpt-4o-2024-08-06" is OK
    if actual_model != model and not actual_model.startswith(model + "-"):
        print(f"")
        print(f"✗ FATAL: Model mismatch!")
        print(f"  You requested: {model}")
        print(f"  API returned:  {actual_model}")
        print(f"")
        print(f"This means the API is using a DIFFERENT model than requested.")
        print(f"Stopping to prevent invalid results.")
        sys.exit(1)
    elif actual_model != model:
        print(f"  Note: API returned versioned model: {actual_model}")

    print("")
    print("✓ PRE-FLIGHT CHECK PASSED")
    print(f"  Confirmed: {actual_model} is working correctly")

except Exception as e:
    print(f"")
    print(f"✗ FATAL: Model test failed!")
    print(f"  Model:  {model}")
    print(f"  Error:  {e}")
    print(f"")
    print(f"The specified model cannot be used. Stopping now.")
    print(f"Fix the model name or API configuration before running.")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "✗ PRE-FLIGHT CHECK FAILED"
    echo "=========================================="
    echo "The model '$CODEX_MODEL' is not working."
    echo "Aborting to prevent running with wrong model."
    exit 1
fi

echo "=========================================="
echo ""

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
