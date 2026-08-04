#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-minimax2.5}"
MEMORY_RATIO="${MEMORY_RATIO:-0.3}"
REPOS="${*:-}"

mkdir -p data/back_trs data/fwr_trs

if [ -n "$REPOS" ]; then
  python scripts/split_before_decomposition.py \
    --repos "$REPOS" \
    --memory-ratio "$MEMORY_RATIO" \
    --output-dir data
else
  python scripts/split_before_decomposition.py \
    --memory-ratio "$MEMORY_RATIO" \
    --output-dir data
fi

python backward_decomposition/decompose_ci_failure.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --model "$MODEL"

python scripts/decompose_commits.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --output-dir data/fwr_trs \
  --model "$MODEL"
