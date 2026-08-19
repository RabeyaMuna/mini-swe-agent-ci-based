#!/bin/bash
# Test script to verify venv creation with debug logging

set -e

echo "============================================"
echo "Testing venv creation for single instance"
echo "============================================"
echo ""

# Activate the main project venv
source .venv-codex/bin/activate

# Create a minimal test dataset with one issue
cat > /tmp/test_single_issue.json << 'EOF'
[
  {
    "id": "test-001",
    "instance_id": "test-001",
    "repo_owner": "RabeyaMuna",
    "repo_name": "taipy",
    "sha_fail": "358dc4d954c07b9725c93750029a413768a9fc16",
    "workflow_path": ".github/workflows/test.yml",
    "workflow_name": "test",
    "workflow": "name: test\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v2\n      - run: echo 'test'",
    "logs": "Test failed"
  }
]
EOF

echo "Created test dataset: /tmp/test_single_issue.json"
echo ""

# Run mini-swe-agent with debug output
echo "Running mini-swe-agent on single test instance..."
echo "Look for DEBUG lines in the output to track venv paths"
echo ""

cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based

python -m minisweagent.run.benchmarks.cibench \
  --dataset /tmp/test_single_issue.json \
  --output results/test_venv_debug/ \
  --config miniswe-agent/configs/default.yaml \
  --slice 0:1 \
  2>&1 | tee /tmp/test_venv_output.log | grep -E "(DEBUG|Creating isolated|Virtual environment ready|testbed)"

echo ""
echo "============================================"
echo "Test complete!"
echo ""
echo "Full log saved to: /tmp/test_venv_output.log"
echo ""
echo "To check where the venv actually is:"
echo "  find results/test_venv_debug -name '.venv*' -type d"
echo "============================================"
