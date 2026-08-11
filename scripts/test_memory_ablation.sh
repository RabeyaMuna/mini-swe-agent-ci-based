#!/bin/bash
# Test memory ablation with a single flower issue (ID 123)
# Tests: l1, l1+l2, l1+l2+l3

set -e

ISSUE_ID="123"
MODEL="glm5.2"
MEMORY_DIR="data/back_trs"
OUTPUT_BASE="data/ablation_test"

echo "========================================================================"
echo "Testing Memory Ablation with Flower Issue ${ISSUE_ID}"
echo "========================================================================"
echo "Model: ${MODEL}"
echo "Memory Dir: ${MEMORY_DIR}"
echo "Output: ${OUTPUT_BASE}"
echo "========================================================================"
echo ""

# Test 1: L1 only
echo "========================================================================"
echo "TEST 1: L1 Memory Only (Failure Sequences)"
echo "========================================================================"
python3 scripts/decompose_backward.py \
  --issue-id "${ISSUE_ID}" \
  --use-huggingface \
  --model "${MODEL}" \
  --use-memory-retrieval \
  --memory-mode l1 \
  --memory-dir "${MEMORY_DIR}" \
  --output-dir "${OUTPUT_BASE}/l1_only" \
  2>&1 | tee "${OUTPUT_BASE}/l1_only.log"

echo ""
echo "✓ L1 test complete"
echo ""

# Test 2: L1+L2
echo "========================================================================"
echo "TEST 2: L1+L2 Memory (Failure Sequences + Repair Strategies)"
echo "========================================================================"
python3 scripts/decompose_backward.py \
  --issue-id "${ISSUE_ID}" \
  --use-huggingface \
  --model "${MODEL}" \
  --use-memory-retrieval \
  --memory-mode l1+l2 \
  --memory-dir "${MEMORY_DIR}" \
  --output-dir "${OUTPUT_BASE}/l1_l2" \
  2>&1 | tee "${OUTPUT_BASE}/l1_l2.log"

echo ""
echo "✓ L1+L2 test complete"
echo ""

# Test 3: L1+L2+L3
echo "========================================================================"
echo "TEST 3: L1+L2+L3 Memory (Full STAIR)"
echo "========================================================================"
python3 scripts/decompose_backward.py \
  --issue-id "${ISSUE_ID}" \
  --use-huggingface \
  --model "${MODEL}" \
  --use-memory-retrieval \
  --memory-mode l1+l2+l3 \
  --memory-dir "${MEMORY_DIR}" \
  --output-dir "${OUTPUT_BASE}/l1_l2_l3" \
  2>&1 | tee "${OUTPUT_BASE}/l1_l2_l3.log"

echo ""
echo "✓ L1+L2+L3 test complete"
echo ""

# Summary
echo "========================================================================"
echo "SUMMARY: Memory Ablation Test Results"
echo "========================================================================"
echo ""

echo "Test outputs saved to:"
echo "  - ${OUTPUT_BASE}/l1_only/"
echo "  - ${OUTPUT_BASE}/l1_l2/"
echo "  - ${OUTPUT_BASE}/l1_l2_l3/"
echo ""

echo "Logs saved to:"
echo "  - ${OUTPUT_BASE}/l1_only.log"
echo "  - ${OUTPUT_BASE}/l1_l2.log"
echo "  - ${OUTPUT_BASE}/l1_l2_l3.log"
echo ""

# Extract key metrics
echo "Key Metrics:"
echo ""

for level in "l1_only" "l1_l2" "l1_l2_l3"; do
    log="${OUTPUT_BASE}/${level}.log"
    if [ -f "$log" ]; then
        echo "[$level]:"

        # Extract STAGE 0 (decomposition cache hit/miss)
        grep -i "STAGE 0.*cached\|STAGE 0.*Decomposing" "$log" | head -1 || echo "  No STAGE 0 info"

        # Extract memory retrieval stats
        grep -i "STAGE 1.*Retrieved\|No matches found" "$log" | head -1 || echo "  No retrieval info"

        # Extract common patterns
        grep -i "STAGE 6.*Found.*common" "$log" | head -1 || echo "  No common patterns info"

        # Extract final problem count
        grep -i "Final.*problems\|STAGE 8" "$log" | tail -1 || echo "  No final count"

        echo ""
    else
        echo "[$level]: Log not found"
        echo ""
    fi
done

echo "========================================================================"
echo "To compare results:"
echo "  diff ${OUTPUT_BASE}/l1_only.log ${OUTPUT_BASE}/l1_l2.log"
echo "  diff ${OUTPUT_BASE}/l1_l2.log ${OUTPUT_BASE}/l1_l2_l3.log"
echo "========================================================================"
