#!/bin/bash
# Test script for commit-based decomposition

echo "========================================"
echo "Testing Commit-Based Decomposition"
echo "========================================"

# Test with just 3 issues first
python3 commit_decomposition/run_commit_decomposition.py \
    --dataset data/trs/filtered_issues.jsonl \
    --limit 3 \
    --output data/trs/commit_decomposed_issues_test.json

echo ""
echo "========================================"
echo "Test complete!"
echo "Check output: data/trs/commit_decomposed_issues_test.json"
echo "========================================"
