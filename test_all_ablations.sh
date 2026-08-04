#!/bin/bash
# Test issue 125 with all ablation modes

set -e

echo "=========================================="
echo "Testing Issue 125 with All Ablations"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

ISSUE_ID="125"

echo "Test 1/4: BASELINE (no memory)"
echo "------------------------------------------"
python3 scripts/run_eval.py --issue-ids $ISSUE_ID --ablation BASELINE --workers 1
echo ""
echo "✓ BASELINE complete"
echo ""

echo "Test 2/4: L1 (file-level memory)"
echo "------------------------------------------"
python3 scripts/run_eval.py --issue-ids $ISSUE_ID --ablation L1 --workers 1
echo ""
echo "✓ L1 complete"
echo ""

echo "Test 3/4: L1+L2 (file + sequences)"
echo "------------------------------------------"
python3 scripts/run_eval.py --issue-ids $ISSUE_ID --ablation L1+L2 --workers 1
echo ""
echo "✓ L1+L2 complete"
echo ""

echo "Test 4/4: L1+L2+L3 (full memory)"
echo "------------------------------------------"
python3 scripts/run_eval.py --issue-ids $ISSUE_ID --ablation L1+L2+L3 --workers 1
echo ""
echo "✓ L1+L2+L3 complete"
echo ""

echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="
echo ""
echo "Results:"
ls -lah results/*/
echo ""
echo "Check logs in results/<mode>/<sha>/"
