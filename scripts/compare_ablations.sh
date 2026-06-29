#!/bin/bash
# Compare Ablation Results

RESULTS_ROOT="${1:-results}"

echo "================================================================================
ABLATION COMPARISON
================================================================================
"

# Count issues per ablation
count_issues() {
    local dir=$1
    local name=$2

    if [ ! -d "$dir" ]; then
        echo "  $name: NOT RUN"
        return
    fi

    # Count directories (excluding preds.json and logs)
    local count=$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

    # Count patches in preds.json
    if [ -f "$dir/preds.json" ]; then
        local patched=$(cat "$dir/preds.json" | jq 'length' 2>/dev/null || echo "0")
        echo "  $name: $patched issues patched, $count issue folders"
    else
        echo "  $name: $count issue folders"
    fi
}

echo "Issue counts:"
count_issues "$RESULTS_ROOT/baseline" "Baseline"
count_issues "$RESULTS_ROOT/L1" "L1      "
count_issues "$RESULTS_ROOT/L1_L2" "L1+L2   "
count_issues "$RESULTS_ROOT/L1_L2_L3" "L1+L2+L3"

echo ""
echo "================================================================================
"

# Find common issues across ablations
echo "Comparing patches for same issues..."
echo ""

# Get list of issues from baseline
if [ -d "$RESULTS_ROOT/baseline" ]; then
    for issue_dir in "$RESULTS_ROOT/baseline"/*; do
        if [ ! -d "$issue_dir" ]; then
            continue
        fi

        issue_sha=$(basename "$issue_dir")

        # Skip preds.json and log files
        if [[ "$issue_sha" == "preds.json" ]] || [[ "$issue_sha" == *.log ]]; then
            continue
        fi

        echo "Issue: $issue_sha"

        # Check if exists in each ablation
        [ -d "$RESULTS_ROOT/baseline/$issue_sha" ] && echo "  ✅ Baseline" || echo "  ❌ Baseline"
        [ -d "$RESULTS_ROOT/L1/$issue_sha" ] && echo "  ✅ L1" || echo "  ❌ L1"
        [ -d "$RESULTS_ROOT/L1_L2/$issue_sha" ] && echo "  ✅ L1+L2" || echo "  ❌ L1+L2"
        [ -d "$RESULTS_ROOT/L1_L2_L3/$issue_sha" ] && echo "  ✅ L1+L2+L3" || echo "  ❌ L1+L2+L3"

        echo ""
    done
fi

echo "================================================================================
SUMMARY
================================================================================

Structure:
$RESULTS_ROOT/
├── baseline/
├── L1/
├── L1_L2/
└── L1_L2_L3/

To view a specific issue's patch:
  cat $RESULTS_ROOT/L1_L2_L3/ISSUE_SHA/ISSUE_SHA.traj.json

To compare patches across ablations:
  diff $RESULTS_ROOT/L1/ISSUE_SHA/*.traj.json \\
       $RESULTS_ROOT/L1_L2/ISSUE_SHA/*.traj.json

================================================================================
"
