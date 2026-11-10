#!/bin/bash
# Cross-Repository Evaluation: L1/L2/L3
# 
# Memory: experimental_data/bidirect_trs (110 L1, 110 L2, 302 L3)
# Eval: experimental_data/eval_issue_ids.json (first 100 TARGET issues)
# Results: results/miniswe-agent/cross_repo/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

python3 experiments/04_cross_repository_l3/prepare_cross_repo_data.py --validate-only

echo "Running Cross-Repo L1/L2/L3 with experimental_data/eval_issue_ids.json"

# Run using the same pattern as other ReCAP experiments
bash ./run_miniswe_direct.sh \
  experimental_data/eval_issue_ids.json L1+L2+L3 cross_repo deepseek-v4-flash \
  "" experimental_data/eval_set.jsonl 1
