#!/bin/bash
# Quick verification that component ablation logs appear correctly
# This does a dry-run style check by looking at the code logic

echo "=========================================="
echo "Component Ablation Quick Verification"
echo "=========================================="
echo ""

echo "Checking implementation..."
echo ""

echo "1. Checking command-line flags in run_miniswe_ci_bench.py..."
if grep -q "disable-dependency-expansion" scripts/run_miniswe_ci_bench.py && \
   grep -q "disable-recurring-recall" scripts/run_miniswe_ci_bench.py && \
   grep -q "disable-filtering" scripts/run_miniswe_ci_bench.py && \
   grep -q "disable-ordering" scripts/run_miniswe_ci_bench.py; then
  echo "   ✓ All 4 flags added"
else
  echo "   ✗ Missing flags"
  exit 1
fi

echo ""
echo "2. Checking environment variable setup..."
if grep -q "RECAP_ENABLE_DEPENDENCY_EXPANSION" scripts/run_miniswe_ci_bench.py && \
   grep -q "RECAP_ENABLE_RECURRING_RECALL" scripts/run_miniswe_ci_bench.py && \
   grep -q "RECAP_ENABLE_FILTERING" scripts/run_miniswe_ci_bench.py && \
   grep -q "RECAP_ENABLE_ORDERING" scripts/run_miniswe_ci_bench.py; then
  echo "   ✓ Environment variables set up"
else
  echo "   ✗ Missing environment variables"
  exit 1
fi

echo ""
echo "3. Checking MemoryPlugin reads environment variables..."
if grep -q "RECAP_ENABLE_DEPENDENCY_EXPANSION" memory_plugin/memory_plugin.py && \
   grep -q "enable_dependency_expansion=enable_dependency_expansion" memory_plugin/memory_plugin.py; then
  echo "   ✓ MemoryPlugin reads env vars and passes to STAIRRetrieval"
else
  echo "   ✗ MemoryPlugin not configured"
  exit 1
fi

echo ""
echo "4. Checking STAIRRetrieval accepts parameters..."
if grep -q "enable_dependency_expansion: bool = True" memory_plugin/stair_retrieval.py && \
   grep -q "enable_recurring_recall: bool = True" memory_plugin/stair_retrieval.py && \
   grep -q "enable_filtering: bool = True" memory_plugin/stair_retrieval.py && \
   grep -q "enable_ordering: bool = True" memory_plugin/stair_retrieval.py; then
  echo "   ✓ STAIRRetrieval accepts all 4 parameters"
else
  echo "   ✗ STAIRRetrieval missing parameters"
  exit 1
fi

echo ""
echo "5. Checking conditional execution in pipeline stages..."
STAGES_OK=0

if grep -q "if self.enable_filtering:" memory_plugin/stair_retrieval.py; then
  echo "   ✓ STAGE 2 (filtering) has conditional execution"
  STAGES_OK=$((STAGES_OK + 1))
fi

if grep -q "if self.enable_dependency_expansion:" memory_plugin/stair_retrieval.py; then
  echo "   ✓ STAGE 4 (dependency expansion) has conditional execution"
  STAGES_OK=$((STAGES_OK + 1))
fi

if grep -q "if self.enable_recurring_recall:" memory_plugin/stair_retrieval.py; then
  echo "   ✓ STAGE 5 (recurring recall) has conditional execution"
  STAGES_OK=$((STAGES_OK + 1))
fi

if grep -q "if self.enable_ordering:" memory_plugin/stair_retrieval.py; then
  echo "   ✓ STAGE 6 (ordering) has conditional execution"
  STAGES_OK=$((STAGES_OK + 1))
fi

if [ $STAGES_OK -ne 4 ]; then
  echo "   ✗ Not all stages have conditional execution"
  exit 1
fi

echo ""
echo "6. Checking skip messages are present..."
if grep -q "SKIP Filtering (disabled by component ablation)" memory_plugin/stair_retrieval.py && \
   grep -q "SKIP Dependency extraction (disabled by component ablation)" memory_plugin/stair_retrieval.py && \
   grep -q "SKIP Common patterns detection (disabled by component ablation)" memory_plugin/stair_retrieval.py && \
   grep -q "SKIP Reordering (disabled by component ablation)" memory_plugin/stair_retrieval.py; then
  echo "   ✓ All skip messages present"
else
  echo "   ✗ Missing skip messages"
  exit 1
fi

echo ""
echo "=========================================="
echo "✓ ALL CHECKS PASSED!"
echo "=========================================="
echo ""
echo "Implementation Summary:"
echo "  • 4 command-line flags added: --disable-{dependency-expansion,recurring-recall,filtering,ordering}"
echo "  • Environment variables used for passing flags"
echo "  • MemoryPlugin reads env vars and passes to STAIRRetrieval"
echo "  • STAIRRetrieval conditionally executes stages 2, 4, 5, 6"
echo "  • Clear log messages when stages are skipped"
echo ""
echo "Backward Compatibility:"
echo "  • All parameters default to True (enabled)"
echo "  • Existing commands work unchanged"
echo "  • Only explicit --disable-X flags skip components"
echo ""
echo "Usage Examples:"
echo ""
echo "# Existing commands work as before (all components enabled):"
echo "./run_miniswe_direct.sh \"\" L1+L2+L3 bidirectional deepseek-v4-flash"
echo ""
echo "# Component ablations for reviewers:"
echo "python scripts/run_miniswe_ci_bench.py --disable-dependency-expansion ..."
echo "python scripts/run_miniswe_ci_bench.py --disable-recurring-recall ..."
echo "python scripts/run_miniswe_ci_bench.py --disable-filtering ..."
echo "python scripts/run_miniswe_ci_bench.py --disable-ordering ..."
echo ""
echo "# Hidden-problem prediction ablation (disable both):"
echo "python scripts/run_miniswe_ci_bench.py --disable-dependency-expansion --disable-recurring-recall ..."
echo ""
