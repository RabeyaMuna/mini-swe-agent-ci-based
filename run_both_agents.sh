#!/bin/bash
# Run BOTH Codex and Mini-SWE on the same selection
# Mirrors the CLI signature of run_codex_direct.sh / run_miniswe_direct.sh
# Usage: ./run_both_agents.sh <issue-ids> [ablation] [direction] [model] [repo_slug] [dataset]

set -euo pipefail

ISSUE_IDS="${1:-}"
ABLATION=${2:-all}
DIRECTION=${3:-both}
MODEL=${4:-gpt-5-mini}
REPO_SLUG=${5:-}
DATASET=${6:-data/eval_set.jsonl}

echo "════ Running BOTH agents ═══════════════════════════════════════"
echo "Ablation=$ABLATION  Direction=$DIRECTION  Model=$MODEL"
echo "Dataset=$DATASET  Repo=${REPO_SLUG:-ALL}  Issues=${ISSUE_IDS:-ALL}"
echo "═════════════════════════════════════════════════════════════════"

# Codex first
./run_codex_direct.sh "${ISSUE_IDS}" "${ABLATION}" "${DIRECTION}" "${MODEL}" "${REPO_SLUG}" "${DATASET}" || true

# Mini-SWE second
./run_miniswe_direct.sh "${ISSUE_IDS}" "${ABLATION}" "${DIRECTION}" "${MODEL}" "${REPO_SLUG}" "${DATASET}" || true

echo "Done. See results/ under codex/ and miniswe-agent/."

