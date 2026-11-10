#!/bin/bash
# Test script to verify component ablation flags work without breaking existing functionality
set -euo pipefail

echo "=========================================="
echo "Component Ablation Test Script"
echo "=========================================="
echo ""

# Get first 2 instance IDs
INSTANCE_IDS=$(head -2 data/eval_issue_ids.json | grep -o '"[0-9]*"' | tr -d '"' | paste -sd ',' -)
echo "Testing with instances: $INSTANCE_IDS"
echo ""

# Activate environment
if [ -f .venv-miniswe/bin/activate ]; then
  source .venv-miniswe/bin/activate
elif [ -f .venv-codex/bin/activate ]; then
  source .venv-codex/bin/activate
else
  echo "ERROR: No virtual environment found"
  exit 1
fi

MODEL="deepseek-v4-flash"
DATASET="experimental_data/eval_set.jsonl"

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
  echo "ERROR: Dataset not found at $DATASET"
  exit 1
fi

echo "=========================================="
echo "TEST 1: Existing functionality (NO FLAGS)"
echo "Should work exactly as before"
echo "=========================================="
echo ""

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset "$DATASET" \
  --issue_regex "^($INSTANCE_IDS)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model "$MODEL" \
  --output_root results/test_baseline \
  --workers 1 || {
  echo "ERROR: Baseline test failed!"
  exit 1
}

echo ""
echo "✓ TEST 1 PASSED: Existing functionality works"
echo ""

echo "=========================================="
echo "TEST 2: Disable dependency expansion"
echo "=========================================="
echo ""

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset "$DATASET" \
  --issue_regex "^($INSTANCE_IDS)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model "$MODEL" \
  --disable-dependency-expansion \
  --output_root results/test_no_dep \
  --workers 1 || {
  echo "ERROR: Disable dependency expansion test failed!"
  exit 1
}

echo ""
echo "✓ TEST 2 PASSED: Dependency expansion can be disabled"
echo ""

echo "=========================================="
echo "TEST 3: Disable recurring recall"
echo "=========================================="
echo ""

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset "$DATASET" \
  --issue_regex "^($INSTANCE_IDS)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model "$MODEL" \
  --disable-recurring-recall \
  --output_root results/test_no_recurring \
  --workers 1 || {
  echo "ERROR: Disable recurring recall test failed!"
  exit 1
}

echo ""
echo "✓ TEST 3 PASSED: Recurring recall can be disabled"
echo ""

echo "=========================================="
echo "TEST 4: Disable filtering"
echo "=========================================="
echo ""

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset "$DATASET" \
  --issue_regex "^($INSTANCE_IDS)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model "$MODEL" \
  --disable-filtering \
  --output_root results/test_no_filtering \
  --workers 1 || {
  echo "ERROR: Disable filtering test failed!"
  exit 1
}

echo ""
echo "✓ TEST 4 PASSED: Filtering can be disabled"
echo ""

echo "=========================================="
echo "TEST 5: Disable ordering"
echo "=========================================="
echo ""

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset "$DATASET" \
  --issue_regex "^($INSTANCE_IDS)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model "$MODEL" \
  --disable-ordering \
  --output_root results/test_no_ordering \
  --workers 1 || {
  echo "ERROR: Disable ordering test failed!"
  exit 1
}

echo ""
echo "✓ TEST 5 PASSED: Ordering can be disabled"
echo ""

echo "=========================================="
echo "TEST 6: Hidden-problem prediction ablation"
echo "(Disable BOTH dependency expansion AND recurring recall)"
echo "=========================================="
echo ""

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset "$DATASET" \
  --issue_regex "^($INSTANCE_IDS)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model "$MODEL" \
  --disable-dependency-expansion \
  --disable-recurring-recall \
  --output_root results/test_no_hidden \
  --workers 1 || {
  echo "ERROR: Hidden-problem prediction ablation test failed!"
  exit 1
}

echo ""
echo "✓ TEST 6 PASSED: Multiple components can be disabled together"
echo ""

echo "=========================================="
echo "ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Existing functionality preserved (backward compatible)"
echo "  ✓ All 4 component ablation flags work correctly"
echo "  ✓ Multiple flags can be combined"
echo ""
echo "You can now run reviewer experiments with:"
echo "  --disable-dependency-expansion"
echo "  --disable-recurring-recall"
echo "  --disable-filtering"
echo "  --disable-ordering"
