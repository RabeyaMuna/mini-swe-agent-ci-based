#!/bin/bash
# Run ExpeRepair with Memory Integration
#
# Usage: ./run_experepair_direct.sh [issue-ids] [ablation] [direction] [model] [repo_filters] [dataset] [workers]
#   [issue-ids]:    Comma-separated IDs, or empty "" for all (default: all from eval_issue_ids.json)
#   [ablation]:     baseline|L1|L1+L2|L1+L2+L3 (default: baseline)
#   [direction]:    none (baseline) | backward|forward|bidirectional (default: none)
#   [model]:        minimax/minimax-m2.5 | gpt-5.4-mini | deepseek-v4-flash (default: minimax/minimax-m2.5)
#   [repo_filters]: Optional comma-separated repo names or owner/repo slugs
#   [dataset]:      Path to eval_set.jsonl (default: data/eval_set.jsonl)
#   [workers]:      Parallel workers (default: 1)
#
# Examples:
#   # Baseline (no memory)
#   ./run_experepair_direct.sh "" baseline none deepseek-v4-flash "" data/eval_set.jsonl 1
#
#   # L1+L2+L3 with bidirectional
#   ./run_experepair_direct.sh "" L1+L2+L3 bidirectional deepseek-v4-flash "" data/eval_set.jsonl 1
#
#   # L1 with backward
#   ./run_experepair_direct.sh "" L1 backward minimax/minimax-m2.5 "" data/eval_set.jsonl 1

set -euo pipefail

# Parse arguments
if [ $# -eq 0 ]; then
    ISSUE_IDS=""
else
    if [ -z "${1:-}" ]; then ISSUE_IDS=""; else ISSUE_IDS="$1"; fi
fi

ABLATION=${2:-baseline}
DIRECTION=${3:-none}
MODEL=${4:-minimax/minimax-m2.5}
REPO_FILTERS=${5:-}
DATASET=${6:-data/eval_set.jsonl}
WORKERS=${7:-1}
RESULTS_ROOT=${EXPEREPAIR_RESULTS_ROOT:-results/experepair}

# Load .env for API keys
[ -f .env ] && source .env
SAVED_OPENAI_KEY="${OPENAI_API_KEY:-}"
SAVED_OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"

# Configure API based on model
case "$MODEL" in
  gpt-*|chatgpt-*|o[0-9]*)
    if [ -z "$SAVED_OPENAI_KEY" ]; then
      echo "ERROR: OPENAI_API_KEY not set in .env" >&2
      exit 1
    fi
    export OPENAI_API_KEY="$SAVED_OPENAI_KEY"
    export OPENROUTER_API_KEY=""
    PROVIDER="OpenAI"
    ;;
  minimax*|deepseek*)
    if [ -z "$SAVED_OPENROUTER_KEY" ]; then
      echo "ERROR: OPENROUTER_API_KEY not set in .env" >&2
      exit 1
    fi
    export OPENAI_API_KEY=""
    export OPENROUTER_API_KEY="$SAVED_OPENROUTER_KEY"
    export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
    PROVIDER="OpenRouter"
    ;;
  *)
    echo "ERROR: Unsupported model: $MODEL" >&2
    exit 1
    ;;
esac

# Canonicalize model name for directory
MODEL_DIR=$(echo "$MODEL" | tr '/' '_' | tr '.' '-')

# Determine output directory based on ablation and direction
if [ "$ABLATION" == "baseline" ]; then
    OUTPUT_DIR="$RESULTS_ROOT/experepair_baseline_$MODEL_DIR"
else
    # Memory ablations go under direction subdirectory
    ABL_DIR=$(echo "$ABLATION" | tr '+' '_' | tr -d ' ' | tr 'A-Z' 'a-z')
    OUTPUT_DIR="$RESULTS_ROOT/experepair_$DIRECTION/${ABL_DIR}_$MODEL_DIR"
fi

# Determine memory directory
if [ "$ABLATION" == "baseline" ]; then
    MEMORY_DIR=""
else
    case "$DIRECTION" in
        backward)
            MEMORY_DIR="data/back_trs"
            ;;
        forward)
            MEMORY_DIR="data/fwr_trs"
            ;;
        bidirectional)
            MEMORY_DIR="data/bidirect_trs"
            ;;
        *)
            echo "ERROR: Invalid direction '$DIRECTION' for ablation '$ABLATION'" >&2
            echo "       Use: backward, forward, or bidirectional" >&2
            exit 1
            ;;
    esac
fi

echo "════ ExpeRepair with Memory ════════════════════════════════════"
echo "Issues:    ${ISSUE_IDS:-ALL}"
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
if [ -n "$MEMORY_DIR" ]; then echo "Memory:    $MEMORY_DIR"; fi
if [ -n "$REPO_FILTERS" ]; then echo "Repos:     $REPO_FILTERS (from $DATASET)"; fi
echo "Model:     $MODEL"
echo "Provider:  $PROVIDER"
echo "Dataset:   $DATASET"
echo "Workers:   $WORKERS"
echo "Output:    $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════"

# Activate venv
if [ -f .venv-codex/bin/activate ]; then
    source .venv-codex/bin/activate
elif [ -f .venv-miniswe/bin/activate ]; then
    source .venv-miniswe/bin/activate
else
    echo "ERROR: No virtual environment found" >&2
    exit 1
fi

# Suppress tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Expand repo filters into issue list
if [ -n "$REPO_FILTERS" ]; then
  echo "Finding issues for repos: $REPO_FILTERS from $DATASET"
  ISSUE_IDS=$(DATASET="$DATASET" REPO_FILTERS="$REPO_FILTERS" python3 - << 'PY'
import json, os, sys
path = os.environ['DATASET']
filters = [v.strip().lower() for v in os.environ['REPO_FILTERS'].split(',') if v.strip()]
matches = {f: 0 for f in filters}
ids = []
seen = set()
with open(path, 'r') as f:
  for line in f:
    if not line.strip(): continue
    row = json.loads(line)
    owner = str(row.get('repo_owner') or '').strip().lower()
    name = str(row.get('repo_name') or '').strip().lower()
    slug = f'{owner}/{name}'.strip('/')
    matched = False
    for flt in filters:
      if '/' in flt:
        matched = flt == slug
      else:
        matched = flt == name
      if matched:
        matches[flt] += 1
        break
    if matched:
      iid = str(row.get('instance_id') or row.get('id') or row.get('sha_fail') or '').strip()
      if iid and iid not in seen:
        ids.append(iid)
        seen.add(iid)
unmatched = [f for f, c in matches.items() if c == 0]
if unmatched:
  print('ERROR: No matching issues for: ' + ', '.join(unmatched), file=sys.stderr)
  raise SystemExit(2)
print(f'Matched {len(ids)} issues.', file=sys.stderr)
print(','.join(ids))
PY
)
  if [ -z "$ISSUE_IDS" ]; then echo "✗ No matches" >&2; exit 1; fi
fi

# If still empty, use eval_issue_ids.json
if [ -z "$ISSUE_IDS" ] && [ -f data/eval_issue_ids.json ]; then
  ISSUE_IDS=$(python3 -c "import json; print(','.join(str(x).strip() for x in json.load(open('data/eval_issue_ids.json')) if str(x).strip()))")
fi

# Build command
CMD="PYTHONPATH=. python3 scripts/run_experepair_baseline.py --dataset $DATASET --model $MODEL --output $OUTPUT_DIR --workers $WORKERS --ablation $ABLATION"

if [ -n "$ISSUE_IDS" ]; then
    CMD="$CMD --issue-ids $ISSUE_IDS"
fi

if [ -n "$MEMORY_DIR" ]; then
    CMD="$CMD --memory-dir $MEMORY_DIR"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run
eval $CMD

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ ExpeRepair Complete ($ABLATION / $DIRECTION)"
echo "  Results: $OUTPUT_DIR/preds.json"
echo "════════════════════════════════════════════════════════════════"
