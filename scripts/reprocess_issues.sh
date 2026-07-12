#!/bin/bash
# reprocess_issues.sh - Reprocess problematic issues
#
# Usage:
#   bash scripts/reprocess_issues.sh data/trs/problematic_issues.txt
#   bash scripts/reprocess_issues.sh data/trs/problematic_issues.txt --dry-run

set -e

ISSUE_LIST=${1:-"data/trs/problematic_issues.txt"}
OUTPUT_DIR="data/trs"
MODEL="${MEMCI_LLM_MODEL:-openrouter/minimax/minimax-m2.5}"
DRY_RUN=false

# Check for dry-run flag
if [[ "$2" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "================================================================================"
echo "Reprocess Problematic Issues"
echo "================================================================================"
echo "Issue list: $ISSUE_LIST"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo ""

# Check if file exists
if [ ! -f "$ISSUE_LIST" ]; then
    echo "ERROR: Issue list not found: $ISSUE_LIST"
    echo ""
    echo "Generate the list first:"
    echo "  python scripts/track_problem_issues.py --fix-list problematic_issues.txt"
    exit 1
fi

# Count issues
TOTAL_ISSUES=$(wc -l < "$ISSUE_LIST" | tr -d ' ')
echo "Found $TOTAL_ISSUES issues to reprocess"
echo ""

if [ $DRY_RUN = true ]; then
    echo "DRY RUN - Would reprocess:"
    cat "$ISSUE_LIST"
    echo ""
    echo "To actually reprocess, run without --dry-run:"
    echo "  bash scripts/reprocess_issues.sh $ISSUE_LIST"
    exit 0
fi

echo "Starting reprocessing..."
echo ""

CURRENT=0
SUCCEEDED=0
FAILED=0

while IFS= read -r issue_id; do
    ((CURRENT++))

    echo "[$CURRENT/$TOTAL_ISSUES] Processing issue $issue_id..."

    # Run decomposition
    if python3 scripts/decompose_ci_failure.py \
        --issue-id "$issue_id" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" 2>&1 | tee -a reprocess.log | grep -q "OK Saved to decomposed_issues.json"; then

        ((SUCCEEDED++))
        echo "  ✓ Success"
    else
        ((FAILED++))
        echo "  ✗ Failed"
    fi

    echo ""

    # Rate limiting
    sleep 2

done < "$ISSUE_LIST"

echo "================================================================================"
echo "REPROCESSING COMPLETE"
echo "================================================================================"
echo "Total: $TOTAL_ISSUES"
echo "Succeeded: $SUCCEEDED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "⚠ Some issues failed. Check reprocess.log for details"
fi

echo "Run quality check again:"
echo "  python scripts/track_problem_issues.py --output-dir $OUTPUT_DIR"
echo "================================================================================"
