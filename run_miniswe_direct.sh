#!/bin/bash
# Run Mini-SWE-Agent (CIBench) over eval issues with consistent flags
# Usage: ./run_miniswe_direct.sh <issue-ids> [ablation] [direction] [model] [repo_filters] [dataset] [workers]
#   <issue-ids>: Comma-separated IDs, or empty string "" to use data/eval_issue_ids.json, or omit to RUN ALL
#   [ablation]:  baseline|L1|L2|L3|L1+L2|L1+L2+L3|all   (default: all)
#   [direction]: none (baseline) | backward|forward|bidirectional|both|all
#                "both" runs backward+forward; "all" runs all memory directions
#   [model]:     gpt-5-mini | gpt-5.4-mini-2026-03-17 | minimax/minimax-m2.5 (default: gpt-5-mini)
#   [repo_filters]: Optional comma-separated repo names or owner/repo slugs
#                   (overrides <issue-ids>)
#   [dataset]:   Path to eval_set.jsonl (default: data/eval_set.jsonl)
#   [workers]:   Parallel issues per ablation (default: 1)

set -euo pipefail

if [ $# -eq 0 ]; then
  ISSUE_IDS=""
else
  if [ -z "${1:-}" ]; then ISSUE_IDS=""; else ISSUE_IDS="$1"; fi
fi
ABLATION=${2:-all}
DIRECTION=${3:-both}
MODEL=${4:-gpt-5-mini}
REPO_FILTERS=${5:-}
DATASET=${6:-data/eval_set.jsonl}
WORKERS=${7:-1}
RESULTS_ROOT=${MINISWE_RESULTS_ROOT:-results/miniswe-agent}

# Load provider credentials from the project file, then expose only the key
# selected by the model. Keeping the unused key present-but-empty prevents
# python-dotenv from loading it again inside the Mini-SWE process.
[ -f .env ] && source .env
SAVED_OPENAI_KEY="${OPENAI_API_KEY:-}"
SAVED_OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
SAVED_OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

unset OPENAI_API_KEY
unset OPENAI_BASE_URL
unset OPENROUTER_API_KEY
unset OPENROUTER_BASE_URL

case "$MODEL" in
  gpt-*|chatgpt-*|o[0-9]*|openai/gpt-*|openai/chatgpt-*|openai/o[0-9]*)
    if [ -z "$SAVED_OPENAI_KEY" ]; then
      echo "ERROR: OPENAI_API_KEY not set in .env" >&2
      exit 1
    fi
    export OPENAI_API_KEY="$SAVED_OPENAI_KEY"
    export OPENROUTER_API_KEY=""
    export OPENROUTER_BASE_URL=""
    PROVIDER="OpenAI"
    API_BASE="https://api.openai.com/v1"
    ;;
  minimax|minimax2.5|minimax-2.5|minimaxm2.5|minimax-m2.5|minimax_m2.5|minimax/*|openrouter/minimax/*|deepseek-v4-flash|deepseek/*|openrouter/deepseek/*)
    if [ -z "$SAVED_OPENROUTER_KEY" ]; then
      echo "ERROR: OPENROUTER_API_KEY not set in .env (required for this model)" >&2
      exit 1
    fi
    export OPENAI_API_KEY=""
    export OPENROUTER_API_KEY="$SAVED_OPENROUTER_KEY"
    export OPENROUTER_BASE_URL="$SAVED_OPENROUTER_BASE_URL"
    PROVIDER="OpenRouter"
    API_BASE="$OPENROUTER_BASE_URL"
    ;;
  *)
    echo "ERROR: Unsupported Mini-SWE model: $MODEL" >&2
    echo "Use a GPT/OpenAI, MiniMax, or DeepSeek model." >&2
    exit 1
    ;;
esac

echo "════ Mini-SWE-Agent Runner ═════════════════════════════════════"
echo "Issues:    ${ISSUE_IDS:-ALL from data/eval_issue_ids.json}"
if [ -n "$REPO_FILTERS" ]; then echo "Repos:     $REPO_FILTERS (from $DATASET)"; fi
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo "Model:     $MODEL"
echo "Provider:  $PROVIDER"
echo "Endpoint:  $API_BASE"
echo "Dataset:   $DATASET"
echo "════════════════════════════════════════════════════════════════"

if [ "${MINISWE_CONFIG_ONLY:-0}" = "1" ]; then
  echo "Configuration-only check complete; no API request was made."
  exit 0
fi

# Prefer the dedicated Mini-SWE environment; retain .venv-codex as a fallback
# for installations that intentionally share the benchmark dependencies.
if [ -f .venv-miniswe/bin/activate ]; then
  source .venv-miniswe/bin/activate
elif [ -f .venv-codex/bin/activate ]; then
  source .venv-codex/bin/activate
else
  echo "ERROR: No Mini-SWE environment found." >&2
  echo "Create .venv-miniswe as described in README.md, then rerun." >&2
  exit 1
fi

# Suppress tokenizers parallelism warning from sentence-transformers
export TOKENIZERS_PARALLELISM=false

# Expand repo filters into an issue list. Short names match all owners while
# owner/name values match only that exact repository.
if [ -n "$REPO_FILTERS" ]; then
  ISSUE_IDS=$(DATASET="$DATASET" REPO_FILTERS="$REPO_FILTERS" python3 - << 'PY'
import json
import os
import sys

path = os.environ['DATASET']
filters = [
    value.strip().lower()
    for value in os.environ['REPO_FILTERS'].split(',')
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
        matched = repo_filter == name or repo_filter == alternate.rsplit('/', 1)[-1]
      if matched:
        matches[repo_filter] += 1
        row_matched = True
    if row_matched:
      iid = str(row.get('instance_id') or row.get('id') or row.get('sha_fail') or '').strip()
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
  if [ -z "$ISSUE_IDS" ]; then echo "✗ No matching issues for $REPO_FILTERS" >&2; exit 1; fi
  echo "Selected issue IDs: $ISSUE_IDS"
fi

# If ISSUE_IDS empty, use eval_issue_ids.json
if [ -z "$ISSUE_IDS" ]; then
  if [ -f data/eval_issue_ids.json ]; then
    ISSUE_IDS=$(python3 - << 'PY'
import json
with open('data/eval_issue_ids.json','r') as f:
  arr = [str(x).strip() for x in json.load(f) if str(x).strip()]
print(','.join(arr))
PY
)
  else
    echo "✗ data/eval_issue_ids.json not found" >&2; exit 1
  fi
fi

# Build regex: ^(id1|id2|...|idN)$
ISSUE_REGEX=$(ISSUE_IDS="$ISSUE_IDS" python3 - << 'PY'
import re, os
ids = [x.strip() for x in os.environ['ISSUE_IDS'].split(',') if x.strip()]
safe = [re.escape(x) for x in ids]
print('^(' + '|'.join(safe) + ')$')
PY
)

# Translate ablation list
if [ "$ABLATION" = "all" ]; then ABLATIONS=(baseline L1 L1+L2 L1+L2+L3); else ABLATIONS=($ABLATION); fi

# Convert to lowercase (Bash 3.2 compatible)
DIRECTION_LOWER=$(echo "$DIRECTION" | tr '[:upper:]' '[:lower:]')
ABLATION_LOWER=$(echo "$ABLATION" | tr '[:upper:]' '[:lower:]')

# For BASELINE, direction is always "none" (no memory used)
if [ "$ABLATION_LOWER" = "baseline" ]; then
  DIRECTION_LOWER="none"
fi

case "$DIRECTION_LOWER" in
  none) DIRECTIONS=() ;;
  backward|forward|bidirectional) DIRECTIONS=("$DIRECTION_LOWER") ;;
  both) DIRECTIONS=(backward forward) ;;
  all) DIRECTIONS=(backward forward bidirectional) ;;
  *)
    echo "ERROR: Direction must be none, backward, forward, bidirectional, both, or all." >&2
    exit 2
    ;;
esac

# Validate: memory ablations require a direction
if [ "$ABLATION_LOWER" != "baseline" ] && [ "$DIRECTION_LOWER" = "none" ]; then
  echo "ERROR: Direction 'none' is valid only for baseline; memory ablations require a direction." >&2
  exit 2
fi

run_one() {
  local ablation="$1"; local direction="$2"
  echo "→ Mini-SWE: abl=$ablation dir=$direction"

  # For baseline, don't use direction in output path (baseline doesn't use directional memory)
  local ablation_lower=$(echo "$ablation" | tr '[:upper:]' '[:lower:]')
  local output_root="$RESULTS_ROOT"
  if [ "$ablation_lower" != "baseline" ]; then
    output_root="$RESULTS_ROOT/$direction"
  fi

  PYTHONPATH=. python3 scripts/run_miniswe_ci_bench.py \
    --dataset "$DATASET" \
    --issue_regex "$ISSUE_REGEX" \
    --ablation "$ablation" \
    --direction "$direction" \
    --model "$MODEL" \
    --output_root "$output_root" \
    --workers "$WORKERS"
}

FAILS=0; TOTAL=0
for abl in "${ABLATIONS[@]}"; do
  # Convert to lowercase (Bash 3.2 compatible)
  abl_lower=$(echo "$abl" | tr '[:upper:]' '[:lower:]')
  if [ "$abl_lower" = "baseline" ]; then
    run_directions=(none)
  else
    run_directions=("${DIRECTIONS[@]}")
  fi
  for dir in "${run_directions[@]}"; do
    TOTAL=$((TOTAL+1))
    if run_one "$abl" "$dir"; then
      echo "✓ Done: $abl | $dir"
    else
      echo "✗ Failed: $abl | $dir"; FAILS=$((FAILS+1))
    fi
  done
done

echo "════════════════════════════════════════════════════════════════"
echo "Mini-SWE runs: $TOTAL  failures: $FAILS"

# Show output locations based on ablation mode
ABLATION_LOWER=$(echo "$ABLATION" | tr '[:upper:]' '[:lower:]')
if [ "$ABLATION_LOWER" = "baseline" ]; then
  echo "Results: $RESULTS_ROOT/baseline_<model>/"
elif [ "$DIRECTION" = "both" ]; then
  echo "Results: $RESULTS_ROOT/{backward,forward}/<ablation>_<model>/"
elif [ "$DIRECTION" = "all" ]; then
  echo "Results: $RESULTS_ROOT/{backward,forward,bidirectional}/<ablation>_<model>/"
else
  echo "Results: $RESULTS_ROOT/$DIRECTION/<ablation>_<model>/"
fi
echo "════════════════════════════════════════════════════════════════"
exit $FAILS
