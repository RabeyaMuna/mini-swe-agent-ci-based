#!/bin/bash
# Test all 3 fixes

echo "=========================================="
echo "Testing All Fixes"
echo "=========================================="

# Clear cache
echo "1. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "OK Done"

# Test fixes are present
echo ""
echo "2. Verifying fixes in code..."
python test_simple.py
if [ $? -ne 0 ]; then
    echo "ERROR Fixes not present in code"
    exit 1
fi

# Run actual test
echo ""
echo "3. Running real benchmark test..."
echo ""

python -m minisweagent.run.benchmarks.cibench \
  --instances data/trs/eval_issues_filtered.jsonl \
  --run_name test_all_fixes \
  --memory_root data/trs \
  --memory_enabled \
  --ablation_levels L1+L2+L3 \
  --max_workers 1

# Check results
echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="

# Check logs for errors
echo ""
echo "Checking for critical errors..."
if grep -q "Memory synthesis LLM failed" results/test_all_fixes/cibench.log; then
    echo "WARNING:  LLM synthesis still failing - check logs:"
    grep "Memory synthesis" results/test_all_fixes/cibench.log
    echo ""
    echo "Debug: Check with --log-level DEBUG to see raw LLM output"
else
    echo "OK No LLM synthesis failures"
fi

# Check instance ID
echo ""
echo "Checking instance ID format..."
python -c "
import json
preds = json.load(open('results/test_all_fixes/preds.json'))
for key in preds.keys():
    print(f'Prediction key: {key}')
    if key.startswith('adap/flower@'):
        print('ERROR Still using repo@sha format instead of benchmark ID')
    elif key == '102':
        print('OK Using correct benchmark ID format')
    else:
        print(f'WARNING:  Using unexpected ID format: {key}')
"

# Check problem statement
echo ""
echo "Checking problem statement content..."
python -c "
import json
traj_file = 'results/test_all_fixes/07c55e7c8bee5b60d9d9647d647f89bec137ef4d/07c55e7c8bee5b60d9d9647d647f89bec137ef4d.traj.json'
try:
    traj = json.load(open(traj_file))
    messages = traj.get('messages', [])
    if len(messages) > 1:
        ps = messages[1].get('content', '')
        print(f'Problem statement length: {len(ps)} chars')
        print(f'  Has Memory Context: {\"## Memory Context\" in ps}')
        print(f'  Has Repair Plan: {\"## Suggested Repair Plan\" in ps}')
        print(f'  Has hidden files section: {\"Files to fix — including those NOT in the log\" in ps}')

        # Check if synthesis worked
        if 'using deterministic guidance' in ps.lower() or 'Relevant L1 memory match' in ps:
            print('WARNING:  Deterministic fallback was used (LLM synthesis failed)')
        else:
            print('OK LLM synthesis likely succeeded')
except FileNotFoundError:
    print('ERROR Trajectory file not found')
except Exception as e:
    print(f'ERROR Error: {e}')
"

echo ""
echo "Full logs: results/test_all_fixes/cibench.log"
