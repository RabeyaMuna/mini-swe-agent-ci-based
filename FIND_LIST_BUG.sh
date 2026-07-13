#!/bin/bash
# Find the exact location of the 'list' object error
# Run this on your SERVER

cd ~/Documents/rabeya/mini-swe-agent-ci-based

echo "========================================="
echo "Running Test to Get Full Traceback"
echo "========================================="
echo ""

# Run with Python -X dev for better debugging
python3 -X dev -u scripts/run_eval.py \
    --issue-ids 72 \
    --ablation L1+L2+L3 \
    --workers 1 \
    2>&1 | tee /tmp/debug_output.txt

echo ""
echo "========================================="
echo "Searching for Error Location"
echo "========================================="

# Extract the traceback
grep -A30 "'list' object" /tmp/debug_output.txt

echo ""
echo "Full log saved to /tmp/debug_output.txt"
echo ""
echo "To see more context:"
echo "  grep -B20 -A40 \"'list' object\" /tmp/debug_output.txt"
