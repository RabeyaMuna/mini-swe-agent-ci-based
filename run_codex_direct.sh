#!/bin/bash
# Run Codex with any OpenAI or Anthropic model
#
# Usage: ./run_codex_direct.sh <issue-ids> [ablation] [direction] [model] [repo_slug] [dataset] [workers]
#   <issue-ids>: Comma-separated IDs, or empty string "" to use eval_issue_ids.json, or omit to RUN ALL
#   [ablation]:  baseline|L1|L2|L3|L1+L2|L1+L2+L3|all   (default: all)
#   [direction]: backward|forward|both                   (default: both)
#   [model]:     gpt-5-mini | gpt-5.4-mini-2026-03-17 | minimax/minimax-m2.5 (default: gpt-5-mini)
#   [repo_slug]: Optional owner/repo to filter IDs from dataset (overrides <issue-ids>)
#   [dataset]:   Path to eval_set.jsonl (default: data/eval_set.jsonl)
#   [workers]:   Parallel issues per ablation (default: 1)
#
#   Examples:
#     # Run ALL ablations and BOTH directions for all issues in eval_issue_ids.json using GPT‑5‑mini
#     ./run_codex_direct.sh "" all both gpt-5-mini
#     # Run MiniMax on a single issue with backward memory L1+L2+L3
#     ./run_codex_direct.sh 129 L1+L2+L3 backward minimax/minimax-m2.5
#     # Run OpenAI snapshot across all issues for a specific repo (pulled from dataset)
#     ./run_codex_direct.sh "" all both gpt-5.4-mini-2026-03-17 octo-org/demo-repo data/eval_set.jsonl

set -e

# Handle empty string for "all issues"
if [ $# -eq 0 ]; then
    ISSUE_IDS=""          # No args: run ALL issues from eval_issue_ids.json
else
    if [ -z "$1" ]; then
        ISSUE_IDS=""      # Empty string: run ALL issues
    else
        ISSUE_IDS="$1"    # Specific issue(s)
    fi
fi

ABLATION=${2:-all}
DIRECTION=${3:-both}
MODEL=${4:-gpt-5-mini}
REPO_SLUG=${5:-}
DATASET=${6:-data/eval_set.jsonl}
WORKERS=${7:-1}

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
if [ -n "$REPO_SLUG" ]; then
    echo "Repo:      $REPO_SLUG (filter from $DATASET)"
fi
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

##############################################
# Configure API based on requested model
# - OpenAI models: use OpenAI API directly
# - OpenRouter-prefixed models: use OpenRouter
# - MiniMax: route via OpenRouter to avoid provider/model-id drift
##############################################
case "$MODEL" in
    gpt-4o|gpt-4|gpt-4-turbo|o1|o1-mini|o3-mini|gpt-5-mini|gpt-5.4-mini|gpt-5.4-mini-2026-03-17)
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
        PROVIDER="OpenAI"
        AUTH_MODE="apikey"
        API_BASE="https://api.openai.com/v1"

        # Write provider config for Codex (~/.codex)
        mkdir -p "$HOME/.codex"
        cat > "$HOME/.codex/config.toml" << 'EOF'
# Codex configuration for OpenAI (native)
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"
EOF
        cat > "$HOME/.codex/auth.json" << EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}
EOF
        chmod 600 "$HOME/.codex/auth.json"
        ;;

    openai/*)
        # OpenAI models via OpenRouter (with openai/ prefix)
        if [ -z "$SAVED_OPENROUTER_KEY" ]; then
            echo "✗ OPENROUTER_API_KEY not set in .env"
            exit 1
        fi
        export OPENAI_API_KEY="$SAVED_OPENROUTER_KEY"
        export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
        export OPENROUTER_APP_NAME="Codex CI Repair"
        export OPENROUTER_SITE_URL="https://github.com/openai/codex"
        CODEX_MODEL="$MODEL"
        CONTEXT_MODEL="$MODEL"
        echo "✓ Using OpenRouter for OpenAI models"
        echo "  Model: $CODEX_MODEL"
        echo "  API Key: ${OPENAI_API_KEY:0:10}..."
        PROVIDER="OpenRouter"
        AUTH_MODE="apikey"
        API_BASE="$OPENAI_BASE_URL"

        # Write provider config for Codex (~/.codex)
        mkdir -p "$HOME/.codex"
        cat > "$HOME/.codex/config.toml" << 'EOF'
# Codex configuration for OpenRouter
model_provider = "openrouter"
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
requires_openai_auth = true
EOF
        cat > "$HOME/.codex/auth.json" << EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}
EOF
        chmod 600 "$HOME/.codex/auth.json"
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
        PROVIDER="Anthropic"
        AUTH_MODE="apikey"
        API_BASE="https://api.anthropic.com"
        ;;

    minimax*|minimax/*)
        # Always use OpenRouter for MiniMax to ensure correct model ids
        if [ -z "$SAVED_OPENROUTER_KEY" ]; then
            echo "✗ OPENROUTER_API_KEY not set in .env (required for MiniMax)"
            exit 1
        fi
        export OPENAI_API_KEY="$SAVED_OPENROUTER_KEY"
        export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
        export OPENROUTER_APP_NAME="Codex CI Repair"
        export OPENROUTER_SITE_URL="https://github.com/openai/codex"
        CODEX_MODEL="$MODEL"
        CONTEXT_MODEL="$MODEL"
        echo "✓ Using OpenRouter for MiniMax"
        echo "  Model: $CODEX_MODEL"
        echo "  API Key: ${OPENAI_API_KEY:0:10}..."
        PROVIDER="OpenRouter"
        AUTH_MODE="apikey"
        API_BASE="$OPENAI_BASE_URL"

        # Write provider config for Codex (~/.codex)
        mkdir -p "$HOME/.codex"
        cat > "$HOME/.codex/config.toml" << 'EOF'
# Codex configuration for MiniMax via OpenRouter
model_provider = "openrouter"
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
requires_openai_auth = true
EOF
        cat > "$HOME/.codex/auth.json" << EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}
EOF
        chmod 600 "$HOME/.codex/auth.json"
        ;;

    *)
        echo "✗ Unknown model: $MODEL"
        echo ""
        echo "Supported models:"
        echo "  OpenAI (direct):    gpt-4o, gpt-4, gpt-4-turbo, o1, o1-mini, o3-mini, gpt-5-mini, gpt-5.4-mini"
        echo "  OpenAI (OpenRouter): openai/gpt-5-mini, openai/gpt-5.4-mini, openai/gpt-4o"
        echo "  Anthropic:          anthropic/claude-opus-4, anthropic/claude-sonnet-4, anthropic/claude-sonnet-3.5"
        echo "  MiniMax:            minimax/minimax-m2.5, minimax/minimax-m3, minimax-*"
        exit 1
        ;;
esac

echo ""

# Config summary banner
echo "CONFIG: auth=$AUTH_MODE | provider=$PROVIDER | codex_home=${CODEX_HOME:-$HOME/.codex} | endpoint=$API_BASE"

# Build the run function for one combo
run_one() {
    local ablation="$1"; local direction="$2"
    local memory_args=""
    if [ "$ablation" != "baseline" ]; then
        local memory_root="data/back_trs"
        [ "$direction" = "forward" ] && memory_root="data/fwr_trs"
        memory_args="--memory-root $memory_root"
    fi
    memory_args="$memory_args --context-model $CODEX_MODEL"

    echo "=========================================="
    echo "Run: ablation=$ablation | direction=$direction"
    echo "=========================================="

    if [ -z "$ISSUE_IDS" ]; then
        PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
            --ablations "$ablation" \
            $memory_args \
            --workers "$WORKERS" \
            --codex-command "codex exec --sandbox danger-full-access --model $CODEX_MODEL"
    else
        PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
            --issue-ids "$ISSUE_IDS" \
            --ablations "$ablation" \
            $memory_args \
            --workers "$WORKERS" \
            --codex-command "codex exec --sandbox danger-full-access --model $CODEX_MODEL"
    fi
}

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

def model_name_matches(requested: str, actual: str) -> bool:
    return (
        actual == requested
        or actual.startswith(requested + "-")
        or actual.endswith("/" + requested)
        or actual == requested.replace("/", "")
        or requested in actual
    )

def try_responses_api(client: OpenAI, model: str) -> tuple[str, str]:
    # Prefer 'max_output_tokens'; fall back to 'max_completion_tokens'.
    try:
        r = client.responses.create(
            model=model,
            input="Respond with only: OK",
            max_output_tokens=8,
        )
        text = getattr(r, "output_text", None)
        if not text:
            # Fallback parse
            try:
                parts = getattr(r, "output", []) or []
                if parts and getattr(parts[0], "content", None):
                    c0 = parts[0].content[0]
                    text = getattr(c0, "text", "")
            except Exception:
                text = ""
        return r.model or model, (text or "")
    except Exception as e1:
        # Retry with 'max_completion_tokens' if server expects that name
        try:
            r = client.responses.create(
                model=model,
                input="Respond with only: OK",
                max_completion_tokens=8,
            )
            text = getattr(r, "output_text", None) or ""
            return r.model or model, text
        except Exception as e2:
            raise RuntimeError(f"Responses API failed: {e1} | retry: {e2}")

def try_chat_api(client: OpenAI, model: str) -> tuple[str, str]:
    # GPT-5 family on Chat Completions expects 'max_completion_tokens'
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "Respond with only: OK"}],
    }
    if model.startswith("gpt-5") or model.startswith("o"):
        kwargs["max_completion_tokens"] = 8
    else:
        kwargs["max_tokens"] = 10
    r = client.chat.completions.create(**kwargs)
    txt = r.choices[0].message.content or ""
    return r.model, txt

try:
    # For Anthropic models, skip this test (different API)
    if model.startswith('anthropic/'):
        print("✓ Anthropic model detected - skipping OpenAI API test")
        sys.exit(0)

    client = OpenAI(base_url=base_url, api_key=api_key)

    use_responses = model.startswith('gpt-5') or model.startswith('o') or '-mini-202' in model

    actual_model = ""
    response_text = ""

    if use_responses:
        try:
            actual_model, response_text = try_responses_api(client, model)
        except Exception as e_res:
            # Fallback to chat API if responses path failed for non-Responses models
            actual_model, response_text = try_chat_api(client, model)
    else:
        try:
            actual_model, response_text = try_chat_api(client, model)
        except Exception as e_chat:
            # Some models only support Responses API
            actual_model, response_text = try_responses_api(client, model)

    print(f"✓ Model responded successfully!")
    print(f"  Requested: {model}")
    print(f"  Actual:    {actual_model}")
    print(f"  Response:  {response_text[:50]}")

    if not model_name_matches(model, actual_model):
        print("")
        print("✗ FATAL: Model mismatch!")
        print(f"  You requested: {model}")
        print(f"  API returned:  {actual_model}")
        print("")
        print("This means the API is using a DIFFERENT model than requested.")
        print("Stopping to prevent invalid results.")
        sys.exit(1)
    elif actual_model != model:
        print(f"  Note: API returned full model path: {actual_model}")

    print("")
    print("✓ PRE-FLIGHT CHECK PASSED")
    print(f"  Confirmed: {actual_model} is working correctly")

except Exception as e:
    print("")
    print("✗ FATAL: Model test failed!")
    print(f"  Model:  {model}")
    print(f"  Error:  {e}")
    print("")
    print("The specified model cannot be used. Stopping now.")
    print("Fix the model name or API configuration before running.")
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

# Allow per-model provider config written to ~/.codex
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"

# If a repo filter is provided, build ISSUE_IDS from the dataset
if [ -n "$REPO_SLUG" ]; then
    echo "Finding issues for repo: $REPO_SLUG from $DATASET"
ISSUE_IDS=$(DATASET="$DATASET" REPO_SLUG="$REPO_SLUG" python3 - << 'PY'
import json, sys, os
path = os.environ.get('DATASET') or 'data/eval_set.jsonl'
repo = os.environ.get('REPO_SLUG')
ids = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        slug = (str(row.get('repo_owner') or '').strip() + '/' + str(row.get('repo_name') or '').strip()).strip('/')
        alt  = str(row.get('repo') or '').strip()
        if slug == repo or alt == repo or alt.endswith('/'+repo.split('/')[-1]):
            iid = str(row.get('instance_id') or row.get('id') or row.get('sha_fail') or '').strip()
            if iid:
                ids.append(iid)
print(','.join(ids))
PY
)
    export ISSUE_IDS
    if [ -z "$ISSUE_IDS" ]; then
        echo "✗ No matching issues found for $REPO_SLUG in $DATASET" >&2
        exit 1
    fi
    echo "  Found issues: ${ISSUE_IDS}"
fi

# Build ablation and direction sets
if [ "$ABLATION" = "all" ]; then
    ABLATIONS_LIST=(baseline L1 L2 L3 L1+L2 L1+L2+L3)
else
    ABLATIONS_LIST=($ABLATION)
fi

if [ "$DIRECTION" = "both" ]; then
    DIRECTIONS_LIST=(backward forward)
else
    DIRECTIONS_LIST=($DIRECTION)
fi

TOTAL=0; FAILS=0
for abl in "${ABLATIONS_LIST[@]}"; do
  for dir in "${DIRECTIONS_LIST[@]}"; do
    TOTAL=$((TOTAL+1))
    if run_one "$abl" "$dir"; then
      echo "✓ Completed: $abl | $dir"
    else
      echo "✗ Failed:    $abl | $dir"
      FAILS=$((FAILS+1))
    fi
  done
done

echo ""
echo "=========================================="
echo "Completed $TOTAL runs. Failures: $FAILS"
echo "Results: results/codex/"
echo "=========================================="

exit $FAILS
