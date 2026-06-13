#!/bin/bash
# Test the fixes with issue 121 (multi-problem: numpy + RST + taplo)

set -e

echo "=========================================="
echo "Testing Multi-Problem CI Repair"
echo "Issue 121: numpy + RST + taplo"
echo "=========================================="

# Step 1: Clear Python cache
echo ""
echo "1. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "OK Cache cleared"

# Step 2: Create test set with issue 121
echo ""
echo "2. Creating test set with issue 121..."
python -c "
import json
import sys

try:
    with open('data/trs/eval_issues.json') as f:
        all_issues = json.load(f)

    issue_121 = [i for i in all_issues if str(i.get('id')) == '121']

    if not issue_121:
        print('ERROR Issue 121 not found in eval_issues.json')
        print('Available IDs:', [i.get('id') for i in all_issues[:10]])
        sys.exit(1)

    with open('data/trs/eval_issues_121.jsonl', 'w') as f:
        f.write(json.dumps(issue_121[0]) + '\n')

    print(f'OK Created eval_issues_121.jsonl')
    print(f'   ID: {issue_121[0].get(\"id\")}')
    print(f'   SHA: {issue_121[0].get(\"sha_fail\")}')
    print(f'   Repo: {issue_121[0].get(\"repo_owner\")}/{issue_121[0].get(\"repo_name\")}')

except FileNotFoundError:
    print('ERROR eval_issues.json not found at data/trs/')
    sys.exit(1)
except Exception as e:
    print(f'ERROR Error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Failed to create test set"
    exit 1
fi

# Step 3: Check memory exists
echo ""
echo "3. Checking memory files..."
if [ ! -f "data/trs/repo_memory.json" ]; then
    echo "ERROR Memory not found. Run: python scripts/build_memory_from_decomposed.py --decomposed data/trs/decomposed_issues.json --output-dir data/trs"
    exit 1
fi

python -c "
import json
l2 = json.load(open('data/trs/repo_memory.json'))
print(f'OK L2 memory loaded: {len(l2)} entries')

# Check if issue 121 is in memory
issue_121_mem = [e for e in l2 if e.get('id') == 'flower_121']
if issue_121_mem:
    print(f'OK Issue 121 found in memory')
    entry = issue_121_mem[0]
    print(f'   Problems: {len(entry.get(\"atomic_problems\", []))}')
    print(f'   Has trajectory: {\"repair_trajectory_summary\" in entry}')
else:
    print(f'ERROR Issue 121 NOT in memory')
    print(f'   Available IDs: {[e.get(\"id\") for e in l2[:5]]}')
"

# Step 4: Run test
echo ""
echo "4. Running benchmark test..."
echo ""

python -m minisweagent.run.benchmarks.cibench \
  --instances data/trs/eval_issues_121.jsonl \
  --run_name test_issue_121_fixed \
  --memory_root data/trs \
  --memory_enabled \
  --ablation_levels L1+L2+L3 \
  --max_workers 1

# Step 5: Check results
echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="

if [ -f "results/test_issue_121_fixed/preds.json" ]; then
    echo ""
    echo "Generated patch:"
    python -c "
import json
preds = json.load(open('results/test_issue_121_fixed/preds.json'))
for key, value in preds.items():
    print(f'\nPrediction ID: {key}')
    print(f'Actual ID: {value.get(\"id\")}')

    diff = value.get('diff', '')
    if diff:
        # Count files changed
        files = [line for line in diff.split('\n') if line.startswith('diff --git')]
        print(f'\n Files changed: {len(files)}')
        for f in files[:10]:
            print(f'   {f}')
        if len(files) > 10:
            print(f'   ... and {len(files) - 10} more')

        # Check for expected changes
        has_numpy = 'ndarrays_arithmetic.py' in diff
        has_rst = '.rst' in diff
        has_toml = 'pyproject.toml' in diff

        print(f'\nOK Expected changes:')
        print(f'   numpy type fix: {\"OK\" if has_numpy else \"ERROR\"}')
        print(f'   RST formatting: {\"OK\" if has_rst else \"ERROR\"}')
        print(f'   taplo enable:   {\"OK\" if has_toml else \"ERROR\"}')

        if has_numpy and has_rst and has_toml:
            print(f'\n SUCCESS: All 3 problems fixed!')
        elif has_numpy or has_rst or has_toml:
            print(f'\nWARNING:  PARTIAL: Some problems fixed')
        else:
            print(f'\nERROR FAILURE: No expected changes found')
    else:
        print('ERROR No diff generated')
"
else
    echo "ERROR No predictions generated"
    echo "Check results/test_issue_121_fixed/ for logs"
fi

echo ""
echo "Full results: results/test_issue_121_fixed/"
