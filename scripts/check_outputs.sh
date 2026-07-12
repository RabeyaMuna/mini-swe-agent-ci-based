#!/bin/bash
# check_outputs.sh
# Verify that the workflow completed successfully

OUTPUT_DIR="${1:-data/trs}"

echo "================================================================================"
echo "CI Memory System - Output Verification"
echo "================================================================================"
echo "Checking directory: $OUTPUT_DIR"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

check_file() {
    file=$1
    desc=$2

    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ MISSING${NC} $desc"
        echo "   File: $file"
        return 1
    fi

    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)

    if [ "$size" -eq 0 ]; then
        echo -e "${YELLOW}⚠ EMPTY${NC} $desc"
        echo "   File: $file (0 bytes)"
        return 2
    fi

    # Format size nicely
    if [ "$size" -lt 1024 ]; then
        size_str="${size}B"
    elif [ "$size" -lt 1048576 ]; then
        size_str="$(($size / 1024))KB"
    else
        size_str="$(($size / 1048576))MB"
    fi

    echo -e "${GREEN}✓ OK${NC} $desc"
    echo "   File: $file ($size_str)"
    return 0
}

echo "STEP 1: Issues Loading"
echo "-----------------------"
check_file "$OUTPUT_DIR/filtered_issues.jsonl" "Filtered issues from HuggingFace"
echo ""

echo "STEP 2: Decomposition"
echo "---------------------"
check_file "$OUTPUT_DIR/decomposed_issues.json" "All issues decomposed into atomic problems"

if [ -f "$OUTPUT_DIR/decomposed_issues.json" ]; then
    count=$(cat "$OUTPUT_DIR/decomposed_issues.json" | jq 'length' 2>/dev/null || echo "?")
    echo "   → Contains $count decomposed issues"
fi
echo ""

echo "STEP 3: Similarity & Split"
echo "--------------------------"
check_file "$OUTPUT_DIR/memory_set.jsonl" "Memory set (30%)"
check_file "$OUTPUT_DIR/eval_set.jsonl" "Evaluation set (70%)"
check_file "$OUTPUT_DIR/similarity_analysis.json" "Similarity analysis"
check_file "$OUTPUT_DIR/split_metadata.json" "Split metadata"

if [ -f "$OUTPUT_DIR/split_metadata.json" ]; then
    echo ""
    echo -e "${BLUE}Split Statistics:${NC}"
    cat "$OUTPUT_DIR/split_metadata.json" | jq -r '
        "   Total issues: \(.total_issues)",
        "   Memory set: \(.memory_set_size) (\(.memory_ratio * 100 | floor)%)",
        "   Eval set: \(.eval_set_size) (\((1 - .memory_ratio) * 100 | floor)%)"
    ' 2>/dev/null || echo "   (Could not parse metadata)"
fi
echo ""

echo "STEP 4: L1/L2/L3 Memory Files"
echo "-----------------------------"
l1_ok=0
l2_ok=0
l3_ok=0

check_file "$OUTPUT_DIR/failure_memory.json" "L1: File-level problems" && l1_ok=1
if [ $l1_ok -eq 1 ] && command -v jq &> /dev/null; then
    count=$(cat "$OUTPUT_DIR/failure_memory.json" | jq 'length' 2>/dev/null || echo "?")
    echo "   → Contains $count file-level problems"
fi

check_file "$OUTPUT_DIR/repo_memory.json" "L2: Repair sequences" && l2_ok=1
if [ $l2_ok -eq 1 ] && command -v jq &> /dev/null; then
    count=$(cat "$OUTPUT_DIR/repo_memory.json" | jq 'length' 2>/dev/null || echo "?")
    echo "   → Contains $count repair sequences"
fi

check_file "$OUTPUT_DIR/cross_memory.json" "L3: Universal patterns" && l3_ok=1
if [ $l3_ok -eq 1 ] && command -v jq &> /dev/null; then
    count=$(cat "$OUTPUT_DIR/cross_memory.json" | jq 'length' 2>/dev/null || echo "?")
    echo "   → Contains $count universal patterns"
fi
echo ""

echo "CACHES (Optional)"
echo "-----------------"
check_file "$OUTPUT_DIR/log_details.json" "CI log analysis cache" || echo -e "${YELLOW}⚠${NC} No cache (will regenerate on next run)"
check_file "$OUTPUT_DIR/workflow_validation_cache.json" "Workflow validation cache" || echo -e "${YELLOW}⚠${NC} No cache (will regenerate on next run)"
echo ""

# Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"

if [ $l1_ok -eq 1 ] && [ $l2_ok -eq 1 ] && [ $l3_ok -eq 1 ]; then
    echo -e "${GREEN}✓ SUCCESS!${NC} All memory files generated correctly."
    echo ""
    echo "Your system is ready to use. The memory files can now be used by the agent."
    echo ""
    echo "Next steps:"
    echo "  1. Inspect memory:"
    echo "     cat $OUTPUT_DIR/failure_memory.json | jq '.[] | {file, problem}' | head"
    echo ""
    echo "  2. Run agent with memory:"
    echo "     python scripts/run_agent.py --issue-id 184 --memory-dir $OUTPUT_DIR"
    echo ""
    echo "  3. Evaluate on eval set:"
    echo "     python scripts/run_eval.py --dataset $OUTPUT_DIR/eval_set.jsonl --memory-dir $OUTPUT_DIR"
    echo ""
elif [ ! -f "$OUTPUT_DIR/memory_set.jsonl" ]; then
    echo -e "${RED}✗ INCOMPLETE${NC} - Workflow stopped at Step 3 (Similarity & Split)"
    echo ""
    echo "The split step failed. Check if decomposed_issues.json exists and contains data."
    echo ""
    echo "To continue, run:"
    echo "  python scripts/prepare_memory_train_test_split.py \\"
    echo "    --dataset $OUTPUT_DIR/filtered_issues.jsonl \\"
    echo "    --output-dir $OUTPUT_DIR"
    echo ""
elif [ $l1_ok -eq 0 ] || [ $l2_ok -eq 0 ] || [ $l3_ok -eq 0 ]; then
    echo -e "${YELLOW}⚠ INCOMPLETE${NC} - L1/L2/L3 memory files missing"
    echo ""
    echo "Step 4 (Memory building) failed or didn't run."
    echo ""
    echo "To generate memory files, run:"
    echo "  python scripts/decompose_ci_failure.py \\"
    echo "    --dataset $OUTPUT_DIR/memory_set.jsonl \\"
    echo "    --output-dir $OUTPUT_DIR"
    echo ""
else
    echo -e "${GREEN}✓ COMPLETE${NC} - System ready!"
    echo ""
fi

echo "================================================================================"
