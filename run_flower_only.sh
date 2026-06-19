#!/bin/bash
# Build L2 memory for Flower issues only with backward conditioning

python scripts/build_memory_from_decomposed.py \
  --decomposed data/trs/decomposed_issues_flower_only.json \
  --verified-patches data/generated_patches_success_only.json \
  --output-dir data/trs_flower_only \
  --model openrouter/minimax/minimax-m2.5

echo ""
echo "=================================="
echo "Output saved to: data/trs_flower_only/"
echo "  - failure_memory.json (L1)"
echo "  - repo_memory.json (L2)"  
echo "  - cross_memory.json (L3)"
echo "=================================="
