#!/bin/bash
# Ablation Study Runner - Correct Structure
# Creates: results/{baseline,L1,L1_L2,L1_L2_L3}/issue_sha/

set -e

# Configuration
DATASET="${1:-data/log_details.json}"  # Standardized location
OUTPUT_ROOT="${2:-results}"
MEMORY_ROOT="${3:-data/trs}"
WORKERS="${4:-1}"

echo "================================================================================
ABLATION STUDY
================================================================================
Dataset:      $DATASET
Output:       $OUTPUT_ROOT
Memory Root:  $MEMORY_ROOT
Workers:      $WORKERS
================================================================================
"

# Function to run one ablation
run_ablation() {
    local name=$1
    local memory_flag=$2
    local ablation_level=$3

    echo ""
    echo "========================================
Running: $name
========================================"

    local output_dir="$OUTPUT_ROOT/$name"

    if [ "$memory_flag" = "no-memory" ]; then
        # Baseline (no memory)
        python3 -m minisweagent.run.benchmarks.cibench \
            --dataset "$DATASET" \
            --output "$output_dir" \
            --workers "$WORKERS"
    else
        # With memory
        python3 -m minisweagent.run.benchmarks.cibench \
            --dataset "$DATASET" \
            --output "$output_dir" \
            --workers "$WORKERS" \
            --memory-enabled \
            --memory-root "$MEMORY_ROOT" \
            --memory-ablation "$ablation_level" \
            --no-save-memory
    fi

    echo "[OK] $name complete: $output_dir"
}

# Run all ablations
run_ablation "baseline" "no-memory" ""
run_ablation "L1" "memory" "L1"
run_ablation "L1_L2" "memory" "L1+L2"
run_ablation "L1_L2_L3" "memory" "L1+L2+L3"

echo ""
echo "================================================================================
ABLATION STUDY COMPLETE
================================================================================
Results structure:
$OUTPUT_ROOT/
|-- baseline/      <- No memory
|-- L1/            <- L1 only
|-- L1_L2/         <- L1+L2
`-- L1_L2_L3/      <- Full pipeline

Each folder contains:
|-- preds.json     <- All patches
`-- issue_sha/     <- Per-issue folders
    |-- testbed/
    `-- *.traj.json
================================================================================
"
