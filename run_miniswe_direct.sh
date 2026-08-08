#!/bin/bash
# Run Mini-SWE-Agent (CIBench) over eval issues with consistent flags
# Usage: ./run_miniswe_direct.sh <issue-ids> [ablation] [direction] [model] [repo_slug] [dataset]
#   <issue-ids>: Comma-separated IDs, or empty string "" to use data/eval_issue_ids.json, or omit to RUN ALL
#   [ablation]:  baseline|L1|L2|L3|L1+L2|L1+L2+L3|all   (default: all)
#   [direction]: backward|forward|both                   (default: both)
#   [model]:     gpt-5-mini | gpt-5.4-mini-2026-03-17 | minimax/minimax-m2.5 (default: gpt-5-mini)
#   [repo_slug]: Optional owner/repo to filter IDs from dataset (overrides <issue-ids>)
#   [dataset]:   Path to eval_set.jsonl (default: data/eval_set.jsonl)

set -euo pipefail

if [ $# -eq 0 ]; then
  ISSUE_IDS=""
else
  if [ -z "${1:-}" ]; then ISSUE_IDS=""; else ISSUE_IDS="$1"; fi
fi
ABLATION=${2:-all}
DIRECTION=${3:-both}
MODEL=${4:-gpt-5-mini}
REPO_SLUG=${5:-}
DATASET=${6:-data/eval_set.jsonl}
WORKERS=${7:-1}

echo "════ Mini-SWE-Agent Runner ═════════════════════════════════════"
echo "Issues:    ${ISSUE_IDS:-ALL from data/eval_issue_ids.json}"
if [ -n "$REPO_SLUG" ]; then echo "Repo:      $REPO_SLUG (from $DATASET)"; fi
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo "Model:     $MODEL"
echo "Dataset:   $DATASET"
echo "════════════════════════════════════════════════════════════════"

# Ensure venv
if [ -d .venv-codex ]; then source .venv-codex/bin/activate; fi

# Load env for keys
[ -f .env ] && source .env

# Expand repo filter into issue list if provided
if [ -n "$REPO_SLUG" ]; then
  ISSUE_IDS=$(DATASET="$DATASET" REPO_SLUG="$REPO_SLUG" python3 - << 'PY'
import json, os
path = os.environ['DATASET']
repo = os.environ['REPO_SLUG']
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
  if [ -z "$ISSUE_IDS" ]; then echo "✗ No matching issues for $REPO_SLUG" >&2; exit 1; fi
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
ISSUE_REGEX=$(python3 - << PY
import re, os
ids = [x.strip() for x in os.environ['ISSUE_IDS'].split(',') if x.strip()]
safe = [re.escape(x) for x in ids]
print('^(' + '|'.join(safe) + ')$')
PY
)

# Translate ablation list
if [ "$ABLATION" = "all" ]; then ABLATIONS=(baseline L1 L1+L2 L1+L2+L3); else ABLATIONS=($ABLATION); fi
if [ "$DIRECTION" = "both" ]; then DIRECTIONS=(backward forward); else DIRECTIONS=($DIRECTION); fi

run_one() {
  local ablation="$1"; local direction="$2"
  echo "→ Mini-SWE: abl=$ablation dir=$direction"
  python3 scripts/run_miniswe_ci_bench.py \
    --dataset "$DATASET" \
    --issue_regex "$ISSUE_REGEX" \
    --ablation "$ablation" \
    --direction "$direction" \
    --model "$MODEL" \
    --workers "$WORKERS"
}

FAILS=0; TOTAL=0
for abl in "${ABLATIONS[@]}"; do
  for dir in "${DIRECTIONS[@]}"; do
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
echo "Results: results/miniswe-agent/"
echo "════════════════════════════════════════════════════════════════"
exit $FAILS
