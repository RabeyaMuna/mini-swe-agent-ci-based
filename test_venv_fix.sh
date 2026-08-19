#!/bin/bash
# Quick test to verify venv fix

set -e

echo "Testing venv creation fix..."
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv-codex/bin/activate

# Clean up any old test results
rm -rf results/test_venv_fix/

# Run on ONE instance
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --output results/test_venv_fix/ \
  --slice 0:1 \
  2>&1 | tee test_venv_fix.log

echo ""
echo "============================================"
echo "Checking where venv was created..."
echo "============================================"

# Find all venvs
VENV_LOCATIONS=$(find results/test_venv_fix -name ".venv*" -type d 2>/dev/null)

if [ -z "$VENV_LOCATIONS" ]; then
    echo "❌ No venv found!"
    exit 1
fi

echo "Found venv(s):"
echo "$VENV_LOCATIONS"
echo ""

# Check if venv is at testbed root (correct) or nested (wrong)
CORRECT_VENV=$(echo "$VENV_LOCATIONS" | grep -E "testbed/\.venv-[^/]+$" | head -1)
NESTED_VENV=$(echo "$VENV_LOCATIONS" | grep -E "testbed/.+/testbed/\.venv" | head -1)

if [ -n "$CORRECT_VENV" ]; then
    echo "✅ SUCCESS! Venv at correct location:"
    echo "   $CORRECT_VENV"
    exit 0
elif [ -n "$NESTED_VENV" ]; then
    echo "❌ FAILED! Venv still in nested location:"
    echo "   $NESTED_VENV"
    exit 1
else
    echo "⚠️  UNKNOWN! Venv in unexpected location:"
    echo "$VENV_LOCATIONS"
    exit 1
fi
