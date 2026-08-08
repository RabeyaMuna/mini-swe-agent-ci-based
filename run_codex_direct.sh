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
# Alias: map minimax names to OpenRouter slug
case "$MODEL" in
  minimax|minimax2.5|minimax-m2.5)
    MODEL="minimax/minimax-m2.5" ;;
esac
REPO_SLUG=${5:-}
DATASET=${6:-data/eval_set.jsonl}
WORKERS=${7:-1}

echo "=========================================="
echo "CODEX CLI - Multi-Model Support"
echo "=========================================="
if [ -z "$ISSUE_IDS" ]; then
    echo "Issues:    ALL (from eval_set.jsonl or eval_issue_ids.json)"
else
