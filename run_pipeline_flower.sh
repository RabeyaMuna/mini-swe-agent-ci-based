#!/bin/bash
# Complete pipeline for Flower issues only

echo "=================================="
echo "MEMORY BUILDING PIPELINE - FLOWER"
echo "=================================="
echo ""

python scripts/build_memory_pipeline.py \
  --repo flower \
  --raw-issues data/trs/memory_seed_issues.json \
  --verified-patches data/generated_patches_success_only.json \
  --output-dir data/trs_flower \
  --model openrouter/minimax/minimax-m2.5

echo ""
echo "=================================="
echo "DONE!"
echo "=================================="
echo ""
echo "Output saved to: data/trs_flower/"
echo ""
echo "Inspect L2 output:"
echo "  cat data/trs_flower/repo_memory.json | jq '.[0]'"
echo ""
echo "Check backward conditioning:"
echo "  cat data/trs_flower/repo_memory.json | jq '.[0].atomic_problems[0].file_changes[0].why_this_file'"
echo ""
echo "Check sequential metadata:"
echo "  cat data/trs_flower/repo_memory.json | jq '.[0].sequential_workflow_metadata'"
echo ""
