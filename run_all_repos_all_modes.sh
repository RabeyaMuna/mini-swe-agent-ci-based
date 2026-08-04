#!/bin/bash
# Run all repos (crewai, flower, camel) with all ablation modes in parallel

set -e

echo "=========================================="
echo "Running All Repos with All Ablations"
echo "=========================================="
echo ""

# Configuration
REPOS="crewai,flower,camel"
ABLATIONS=("BASELINE" "L1" "L1+L2" "L1+L2+L3")
WORKERS=4  # Parallel workers per ablation
MODEL=${MODEL:-minimax}  # Default model, can override with: MODEL=glm5.2 ./run_all_repos_all_modes.sh
DIRECTION=${DIRECTION:-backward}  # Default direction, can override with: DIRECTION=forward ./run_all_repos_all_modes.sh

# Activate venv
source .venv/bin/activate

echo "Repos: $REPOS"
echo "Model: $MODEL"
echo "Direction: $DIRECTION"
echo "Ablations: ${ABLATIONS[@]}"
echo "Workers per ablation: $WORKERS"
echo ""

# Function to run one ablation
run_ablation() {
    local ablation=$1
    echo "=========================================="
    echo "Starting: $ablation"
    echo "=========================================="

    python3 scripts/run_eval.py \
        --repos "$REPOS" \
        --ablation "$ablation" \
        --model "$MODEL" \
        --direction "$DIRECTION" \
        --workers "$WORKERS" \
        --exclude-memory \
        2>&1 | tee "logs/${ablation/+/_}_${MODEL//./_}_${DIRECTION}_$(date +%Y%m%d_%H%M%S).log"

    echo "✓ Completed: $ablation"
    echo ""
}

# Create logs directory
mkdir -p logs

# Run all ablations in parallel
echo "Starting all ablations in parallel..."
echo ""

for ablation in "${ABLATIONS[@]}"; do
    run_ablation "$ablation" &
done

# Wait for all to complete
wait

echo "=========================================="
echo "All Runs Complete!"
echo "=========================================="
echo ""
echo "Results:"
ls -lah results/
echo ""
echo "Logs:"
ls -lah logs/
