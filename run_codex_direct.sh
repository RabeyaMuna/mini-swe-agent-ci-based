#!/bin/bash
# Run Codex with any model via OpenRouter
#
# Usage: ./run_codex_direct.sh <issue-ids> [ablation] [direction] [model] [repo_filters] [dataset] [workers]
#   <issue-ids>: Comma-separated IDs, or empty string "" to use eval_issue_ids.json, or omit to RUN ALL
#   [ablation]:  baseline|L1|L2|L3|L1+L2|L1+L2+L3|all   (default: all)
#   [direction]: backward|forward|both                   (default: both)
#   [model]:     Any OpenRouter model (e.g., gpt-5-mini, deepseek-v4-flash, minimax/minimax-m2.5) (default: gpt-5-mini)
#   [repo_filters]: Optional comma-separated repo names or owner/repo slugs
#                   (overrides <issue-ids>)
#   [dataset]:   Path to eval_set.jsonl (default: data/eval_set.jsonl)
#   [workers]:   Parallel issues per ablation (default: 1)
#
#   Examples:
#     # Run ALL ablations and BOTH directions for all issues in eval_issue_ids.json using GPT‑5‑mini
#     ./run_codex_direct.sh "" all both gpt-5-mini
#     # Run DeepSeek on a single issue with backward memory L1+L2+L3
#     ./run_codex_direct.sh 129 L1+L2+L3 backward deepseek-v4-flash
#     # Run MiniMax on a single issue with backward memory L1+L2+L3
#     ./run_codex_direct.sh 129 L1+L2+L3 backward minimax/minimax-m2.5
#     # Run several repos, including all owners of the short name "agno"
#     ./run_codex_direct.sh "" L1+L2+L3 forward deepseek-v4-flash \
#       "agno,axolotl,owner/demo-repo" data/eval_set.jsonl 4

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
REPO_FILTERS=${5:-}
DATASET=${6:-data/eval_set.jsonl}
WORKERS=${7:-1}
RESULTS_ROOT=${CODEX_RESULTS_ROOT:-results/codex}

# Canonicalize model aliases before selecting the provider or CODEX_HOME.
# OpenAI's API uses unprefixed model ids; MiniMax uses its OpenRouter slug.
case "$MODEL" in
    openai/*)
        MODEL="${MODEL#openai/}"
        ;;
    minimax|minimax2.5|minimax-m2.5)
        MODEL="minimax/minimax-m2.5"
        ;;
    deepseek|deepseek-v4-flash|deepseek-v4)
        MODEL="deepseek/deepseek-v4-flash"
        ;;
esac

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
if [ -n "$REPO_FILTERS" ]; then
    echo "Repos:     $REPO_FILTERS (filter from $DATASET)"
fi
echo ""

# Load .env and save OpenRouter key
[ -f .env ] && source .env
SAVED_OPENROUTER_KEY="$OPENROUTER_API_KEY"

# Use a separate project-local Codex home for every model. Provider and auth
# settings are machine-local Codex settings, so sharing one config.toml lets a
# concurrent OpenRouter run overwrite an OpenAI run (and vice versa).
MODEL_CONFIG_NAME=$(printf '%s' "$MODEL" | tr '/.' '__' | tr -cd 'A-Za-z0-9_-')
if [ -z "$MODEL_CONFIG_NAME" ]; then
    echo "ERROR: Could not derive a safe Codex config name from model: $MODEL" >&2
    exit 1
fi
PROJECT_CODEX_HOME="$PWD/.codex-local/$MODEL_CONFIG_NAME"
export CODEX_HOME="$PROJECT_CODEX_HOME"

# Clear ALL API-related environment variables to prevent conflicts
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
unset OPENROUTER_API_KEY
unset MINIMAX_API_KEY
unset OPENAI_BASE_URL
unset OPENROUTER_BASE_URL
unset OPENROUTER_APP_NAME
unset OPENROUTER_SITE_URL
unset CODEX_PROVIDER
unset CODEX_API_BASE

##############################################
# Configure API - All models use OpenRouter
##############################################
# All models route through OpenRouter for unified access
if [ -z "$SAVED_OPENROUTER_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set in .env"
    exit 1
fi

export OPENAI_API_KEY="$SAVED_OPENROUTER_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_APP_NAME="Codex CI Repair"
export OPENROUTER_SITE_URL="https://github.com/openai/codex"
CODEX_MODEL="$MODEL"
CONTEXT_MODEL="$MODEL"
echo "Using OpenRouter"
echo "  Model: $CODEX_MODEL"
echo "  API Key: ${OPENAI_API_KEY:0:10}..."
PROVIDER="OpenRouter"
AUTH_MODE="apikey"
API_BASE="$OPENAI_BASE_URL"
export CODEX_PROVIDER="openrouter"
export CODEX_API_BASE="$API_BASE"

# Write provider config for Codex (project-local)
mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" << 'EOF'
# Codex configuration for OpenRouter (all models)
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
cat > "$CODEX_HOME/auth.json" << EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}
EOF
chmod 600 "$CODEX_HOME/auth.json"

echo ""

# Config summary banner (CODEX_HOME already set early in script)
echo "CONFIG: auth=$AUTH_MODE | provider=$PROVIDER | codex_home=$CODEX_HOME (project-local) | endpoint=$API_BASE"

# Allow tests and operators to verify routing/configuration without starting a
# benchmark or making a model request.
if [ "${CODEX_CONFIG_ONLY:-0}" = "1" ]; then
    echo "Configuration-only check complete; no API request was made."
    exit 0
fi

RESUME_ARGS=(--resume)
case "${CODEX_RESUME:-1}" in
    0|false|FALSE|no|NO)
        RESUME_ARGS=(--no-resume)
        ;;
esac

# Build the run function for one combo
run_one() {
    local ablation="$1"; local direction="$2"
    local memory_args=""
    local direction_output_root="$RESULTS_ROOT/$direction"
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
            --dataset "$DATASET" \
            --direction "$direction" \
            --output-root "$direction_output_root" \
            $memory_args \
            "${RESUME_ARGS[@]}" \
            --workers "$WORKERS" \
            --codex-command "codex exec --sandbox workspace-write --model $CODEX_MODEL"
    else
        PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
            --issue-ids "$ISSUE_IDS" \
            --ablations "$ablation" \
            --dataset "$DATASET" \
            --direction "$direction" \
            --output-root "$direction_output_root" \
            $memory_args \
            "${RESUME_ARGS[@]}" \
            --workers "$WORKERS" \
            --codex-command "codex exec --sandbox workspace-write --model $CODEX_MODEL"
    fi
}

# Activate environment
source .venv-codex/bin/activate

# Suppress tokenizers parallelism warning from sentence-transformers
export TOKENIZERS_PARALLELISM=false

# Stream Python progress immediately when stdout is redirected by nohup.
export PYTHONUNBUFFERED=1

# Export CODEX_MODEL so the Python test can see it
export CODEX_MODEL

# CODEX_HOME is already isolated by model above - do not override it.

# If repo filters are provided, build ISSUE_IDS from the dataset. A short name
# matches repo_name across all owners; owner/name matches one exact repository.
if [ -n "$REPO_FILTERS" ]; then
    echo "Finding issues for repos: $REPO_FILTERS from $DATASET"
ISSUE_IDS=$(DATASET="$DATASET" REPO_FILTERS="$REPO_FILTERS" python3 - << 'PY'
import json
import os
import sys

path = os.environ.get('DATASET') or 'data/eval_set.jsonl'
filters = [
    value.strip().lower()
    for value in os.environ.get('REPO_FILTERS', '').split(',')
    if value.strip()
]
matches = {repo_filter: 0 for repo_filter in filters}
ids = []
seen_ids = set()
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        owner = str(row.get('repo_owner') or '').strip().lower()
        name = str(row.get('repo_name') or '').strip().lower()
        alternate = str(row.get('repo') or '').strip().lower()
        slug = f'{owner}/{name}'.strip('/')
        row_matched = False
        for repo_filter in filters:
            if '/' in repo_filter:
                matched = repo_filter == slug or repo_filter == alternate
            else:
                alternate_name = alternate.rsplit('/', 1)[-1]
                matched = repo_filter == name or repo_filter == alternate_name
            if matched:
                matches[repo_filter] += 1
                row_matched = True
        if row_matched:
            iid = str(
                row.get('instance_id') or row.get('id') or row.get('sha_fail') or ''
            ).strip()
            if iid and iid not in seen_ids:
                ids.append(iid)
                seen_ids.add(iid)

unmatched = [repo_filter for repo_filter, count in matches.items() if count == 0]
if unmatched:
    print(
        'ERROR: No matching issues for repo filter(s): ' + ', '.join(unmatched),
        file=sys.stderr,
    )
    raise SystemExit(2)

print(
    f'Matched {len(ids)} unique issues from {len(filters)} repo filter(s).',
    file=sys.stderr,
)
print(','.join(ids))
PY
)
    export ISSUE_IDS
    if [ -z "$ISSUE_IDS" ]; then
        echo "ERROR: No matching issues found for $REPO_FILTERS in $DATASET" >&2
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
      echo "Completed: $abl | $dir"
    else
      echo "Failed:    $abl | $dir"
      FAILS=$((FAILS+1))
    fi
  done
done

echo ""
echo "=========================================="
echo "Completed $TOTAL runs. Failures: $FAILS"
if [ "$DIRECTION" = "both" ]; then
    echo "Results: $RESULTS_ROOT/{backward,forward}/"
else
    echo "Results: $RESULTS_ROOT/$DIRECTION/"
fi
echo "=========================================="

exit $FAILS
