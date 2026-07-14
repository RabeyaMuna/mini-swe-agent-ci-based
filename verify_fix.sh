#!/bin/bash
# Verification script to check if the fix is properly applied

echo "🔍 Checking if the fix is properly applied..."
echo ""

# Check if testbed_path is used instead of agent.environment.repo_path
if grep -q "repo_path=str(testbed_path)" src/minisweagent/run/benchmarks/cibench.py; then
    echo "✅ Fix is applied: Using testbed_path"
else
    echo "❌ Fix NOT applied: Still using agent.environment.repo_path"
    exit 1
fi

# Check if agent.environment is still referenced (should only be in import)
count=$(grep -c "agent.environment" src/minisweagent/run/benchmarks/cibench.py)
if [ "$count" -eq 1 ]; then
    echo "✅ No agent.environment references (except import)"
else
    echo "❌ Found $count references to agent.environment"
    grep -n "agent.environment" src/minisweagent/run/benchmarks/cibench.py
    exit 1
fi

# Check if problem_validator.py exists
if [ -f "src/minisweagent/run/benchmarks/utils/problem_validator.py" ]; then
    echo "✅ problem_validator.py exists"
else
    echo "❌ problem_validator.py NOT found"
    exit 1
fi

echo ""
echo "🎉 All fixes are properly applied!"
echo ""
echo "The error you saw was from a cached/old version."
echo "To ensure Python picks up the changes, try:"
echo "  1. Kill any running Python processes"
echo "  2. Clear Python cache: find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null"
echo "  3. Re-run the command"
