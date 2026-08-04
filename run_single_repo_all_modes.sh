#!/bin/bash
# Run a single repo with all ablation modes in parallel
#
# Usage:
#   ./run_single_repo_all_modes.sh crewai
#   ./run_single_repo_all_modes.sh flower
#   ./run_single_repo_all_modes.sh camel

set -e

REPO=${1:-flower}
WORKERS=${2:-4}
MODEL=${3:-minimax}  # Third argument or default to minimax
DIRECTION=${4:-backward}  # Fourth argument or default to backward

echo "=========================================="
echo "Running $REPO with All Ablations"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

ABLATIONS=("BASELINE" "L1" "L1+L2" "L1+L2+L3")

echo "Repo: $REPO"
echo "Model: $MODEL"
echo "Direction: $DIRECTION"
echo "Ablations: ${ABLATIONS[@]}"
echo "Workers: $WORKERS"
echo ""

# Create logs directory
mkdir -p logs

# Function to run one ablation
run_ablation() {
    local ablation=$1
    local repo=$2

    echo "[$repo] Starting: $ablation"

    python3 scripts/run_eval.py \
        --repos "$repo" \
        --ablation "$ablation" \
        --model "$MODEL" \
        --direction "$DIRECTION" \
        --workers "$WORKERS" \
        --exclude-memory \
        2>&1 | tee "logs/${repo}_${ablation/+/_}_${MODEL//./_}_${DIRECTION}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$repo] ✓ Completed: $ablation"
}

# Run all ablations in parallel
for ablation in "${ABLATIONS[@]}"; do
    run_ablation "$ablation" "$REPO" &
done

# Wait for all to complete
wait

echo ""
echo "=========================================="
echo "✓ All Runs Complete for $REPO"
echo "=========================================="
echo ""
echo "Results:"
find results -name "*.traj.json" | grep -i "$REPO" | wc -l | xargs echo "  Instances:"
