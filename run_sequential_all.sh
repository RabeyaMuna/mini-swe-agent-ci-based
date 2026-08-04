#!/bin/bash
# Run all repos with all modes SEQUENTIALLY (one at a time)
# Use this if parallel runs cause resource issues

set -e

echo "=========================================="
echo "Sequential Run: All Repos, All Modes"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

REPOS=("crewai" "flower" "camel")
ABLATIONS=("BASELINE" "L1" "L1+L2" "L1+L2+L3")
WORKERS=8  # More workers since running sequentially

# Create logs directory
mkdir -p logs

total=$((${#REPOS[@]} * ${#ABLATIONS[@]}))
current=0

for repo in "${REPOS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
        current=$((current + 1))
        echo ""
        echo "=========================================="
        echo "[$current/$total] $repo - $ablation"
        echo "=========================================="

        python3 scripts/run_eval.py \
            --repos "$repo" \
            --ablation "$ablation" \
            --workers "$WORKERS" \
            --exclude-memory \
            2>&1 | tee "logs/${repo}_${ablation/+/_}_$(date +%Y%m%d_%H%M%S).log"

        echo "✓ Completed: $repo - $ablation"
    done
done

echo ""
echo "=========================================="
echo "✓ All Runs Complete!"
echo "=========================================="
./summarize_results.sh
