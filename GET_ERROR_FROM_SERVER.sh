#!/bin/bash
# Extract the full error traceback from cibench.log on the SERVER
# Run this ON THE SERVER

cd ~/Documents/rabeya/mini-swe-agent-ci-based

echo "========================================="
echo "Extracting Full Error Traceback"
echo "========================================="
echo ""

# Check if log exists
if [ ! -f "results/L1_L2_L3/cibench.log" ]; then
    echo "❌ Log file not found: results/L1_L2_L3/cibench.log"
    echo "Run the evaluation first!"
    exit 1
fi

echo "📋 Found log file. Extracting error..."
echo ""

# Extract the full traceback (Python exception includes Traceback lines)
grep -B50 "'list' object has no attribute 'get'" results/L1_L2_L3/cibench.log | tail -60

echo ""
echo "========================================="
echo ""
echo "If you see 'Traceback (most recent call last):' above,"
echo "that shows the EXACT file and line number!"
echo ""
echo "Otherwise, the error might be caught silently."
echo "Try running with more verbose logging:"
echo ""
echo "  export LOG_LEVEL=DEBUG"
echo "  python3 scripts/run_eval.py --issue-ids 73 --ablation L1+L2+L3 --workers 1"
