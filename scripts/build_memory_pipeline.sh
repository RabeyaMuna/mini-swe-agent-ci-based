#!/bin/bash
#
# build_memory_pipeline.sh - Complete Memory Building Pipeline
# ============================================================
#
# Builds L1/L2/L3 memory using professor's approach:
# 1. Reverse engineer atomic problems from CI failure + diff
# 2. Build L1 (per-file with reasoning)
# 3. Build L2 (per-issue with multi-problem decomposition)
# 4. Build L3 (cross-repo with hierarchical abstraction)
#
# Usage:
#   # Test on 1 issue
#   ./scripts/build_memory_pipeline.sh --test
#
#   # Build full memory
#   ./scripts/build_memory_pipeline.sh
#
#   # Build for first 5 issues
#   ./scripts/build_memory_pipeline.sh --limit 5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
EVAL_ISSUES="data/trs/eval_issues.json"
OUTPUT_DIR="data/trs_memory_v2"
MODEL="minimax/minimax-m2.5"
LIMIT=""
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            LIMIT="1"
            OUTPUT_DIR="test_memory_v2"
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--limit N] [--model MODEL] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Memory Building Pipeline (Professor's Approach)           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Eval issues:  $EVAL_ISSUES"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Model:        $MODEL"
[[ -n "$LIMIT" ]] && echo "  Limit:        $LIMIT issues"
[[ "$TEST_MODE" == true ]] && echo -e "  ${YELLOW}TEST MODE (1 issue only)${NC}"
echo ""

# Check if eval issues exist
if [[ ! -f "$EVAL_ISSUES" ]]; then
    echo -e "${RED}✗ Eval issues not found: $EVAL_ISSUES${NC}"
    exit 1
fi

TOTAL_ISSUES=$(python3 -c "import json; print(len(json.load(open('$EVAL_ISSUES'))))")
echo -e "${GREEN}✓ Found $TOTAL_ISSUES issues in eval file${NC}"
echo ""

# Step 1: Reverse Engineer Atomic Problems
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 1: Reverse Engineer Atomic Problems (CI log + diff)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "This step uses LLM to infer hidden problems from the diff."
echo "The CI log shows only the FIRST failure, but the diff fixes ALL problems."
echo ""

DECOMPOSE_CMD="python scripts/decompose_ci_failure.py --eval-issues $EVAL_ISSUES --model $MODEL"
[[ -n "$LIMIT" ]] && DECOMPOSE_CMD="$DECOMPOSE_CMD --limit $LIMIT"

DECOMPOSED_OUTPUT="data/trs/decomposed_issues.json"
if [[ "$TEST_MODE" == true ]]; then
    DECOMPOSED_OUTPUT="test_decomposed_issues.json"
fi
DECOMPOSE_CMD="$DECOMPOSE_CMD --output $DECOMPOSED_OUTPUT"

echo "Running: $DECOMPOSE_CMD"
echo ""

if $DECOMPOSE_CMD; then
    echo ""
    echo -e "${GREEN}✓ Step 1 complete: Atomic problems identified${NC}"
else
    echo ""
    echo -e "${RED}✗ Step 1 failed${NC}"
    exit 1
fi

# Step 2: Build L1/L2/L3 Memory
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 2: Build L1/L2/L3 Memory with Abstraction${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Building:"
echo "  L1: Per-file memory with reasoning (WHY + HOW)"
echo "  L2: Per-issue with atomic problems + dependencies"
echo "  L3: Cross-repo with 3-level hierarchical abstraction"
echo ""

BUILD_CMD="python scripts/build_memory_from_decomposed.py --decomposed $DECOMPOSED_OUTPUT --output-dir $OUTPUT_DIR --model $MODEL"

echo "Running: $BUILD_CMD"
echo ""

if $BUILD_CMD; then
    echo ""
    echo -e "${GREEN}✓ Step 2 complete: L1/L2/L3 memory built${NC}"
else
    echo ""
    echo -e "${RED}✗ Step 2 failed${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✓ Memory building complete!${NC}"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "  - failure_memory.json (L1: per-file)"
echo "  - repo_memory.json (L2: per-issue with atomic problems)"
echo "  - cross_memory.json (L3: cross-repo with abstraction)"
echo ""

# Check output
python3 << EOF
import json
from pathlib import Path

output = Path("$OUTPUT_DIR")

l1 = json.loads((output / "failure_memory.json").read_text()) if (output / "failure_memory.json").exists() else []
l2 = json.loads((output / "repo_memory.json").read_text()) if (output / "repo_memory.json").exists() else []
l3 = json.loads((output / "cross_memory.json").read_text()) if (output / "cross_memory.json").exists() else []

print("Memory Statistics:")
print(f"  L1 (per-file):     {len(l1)} entries")
print(f"  L2 (per-issue):    {len(l2)} entries")
if l2:
    total_atomic = sum(m.get("total_atomic_problems", 0) for m in l2)
    print(f"    Atomic problems: {total_atomic}")
print(f"  L3 (cross-repo):   {len(l3)} principles")

# Check L2 structure
if l2:
    visible = sum(
        sum(1 for p in m.get("atomic_problems", []) if p.get("visibility") == "visible_in_log")
        for m in l2
    )
    hidden = sum(
        sum(1 for p in m.get("atomic_problems", []) if p.get("visibility") == "hidden")
        for m in l2
    )
    print(f"\nAtomic Problem Visibility:")
    print(f"  Visible (in CI log): {visible}")
    print(f"  Hidden (inferred):   {hidden}")

# Check L3 abstraction
if l3:
    with_hierarchy = sum(1 for p in l3 if "abstraction_hierarchy" in p)
    print(f"\nL3 Abstraction:")
    print(f"  Principles with 3-level hierarchy: {with_hierarchy}/{len(l3)}")
EOF

echo ""

if [[ "$TEST_MODE" == true ]]; then
    echo -e "${YELLOW}TEST MODE:${NC} Memory built for 1 issue only."
    echo ""
    echo "Next steps:"
    echo "  1. Inspect: cat $DECOMPOSED_OUTPUT | python -m json.tool | head -100"
    echo "  2. Check L2: cat $OUTPUT_DIR/repo_memory.json | python -m json.tool | head -100"
    echo "  3. If good, run full: ./scripts/build_memory_pipeline.sh"
else
    echo "Next steps:"
    echo "  1. Precompute embeddings: python scripts/precompute_embeddings.py --memory-root $OUTPUT_DIR"
    echo "  2. Verify structure: python scripts/verify_memory_structure.py --memory-root $OUTPUT_DIR"
    echo "  3. Run evaluation: scripts/run_cibench_minimax_openrouter.sh --slice 0:15"
fi
echo ""
