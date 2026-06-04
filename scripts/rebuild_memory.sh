#!/bin/bash
#
# rebuild_memory.sh - Complete Memory Rebuild Pipeline
# =====================================================
#
# Rebuilds high-quality memory using CI-REPAIR-BENCH approach.
# This is the recommended way to build memory for evaluation.
#
# Usage:
#   # Test on 1 issue first
#   ./scripts/rebuild_memory.sh --test
#
#   # Build full memory for all eval issues
#   ./scripts/rebuild_memory.sh
#
#   # Resume from checkpoint
#   ./scripts/rebuild_memory.sh --resume
#
#   # Custom slice (e.g., first 5 issues)
#   ./scripts/rebuild_memory.sh --slice 0:5

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
SEED_FILE="data/trs/eval_issues.json"
OUTPUT_DIR="data/trs_rebuilt"
MODEL="minimax/minimax-m2.5"
SLICE=""
RESUME=""
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            SLICE="0:1"
            OUTPUT_DIR="test_memory"
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --slice)
            SLICE="$2"
            shift 2
            ;;
        --seed-file)
            SEED_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--resume] [--slice 0:5] [--seed-file FILE] [--output-dir DIR] [--model MODEL]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       MEMORY REBUILD PIPELINE (CI-REPAIR-BENCH Style)         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Seed file:    $SEED_FILE"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Model:        $MODEL"
[[ -n "$SLICE" ]] && echo "  Slice:        $SLICE"
[[ -n "$RESUME" ]] && echo "  Mode:         RESUME"
[[ "$TEST_MODE" == true ]] && echo -e "  ${YELLOW}TEST MODE (1 issue only)${NC}"
echo ""

# Check if seed file exists
if [[ ! -f "$SEED_FILE" ]]; then
    echo -e "${RED}✗ Seed file not found: $SEED_FILE${NC}"
    exit 1
fi

# Count issues
TOTAL_ISSUES=$(python3 -c "import json; print(len(json.load(open('$SEED_FILE'))))")
echo -e "${GREEN}✓ Found $TOTAL_ISSUES issues in seed file${NC}"
echo ""

# Step 1: Build Memory Bank
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 1: Build Memory Bank (L1/L2/L3)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

BUILD_CMD="python scripts/build_memory_bank.py --seed-file $SEED_FILE --output-dir $OUTPUT_DIR --model $MODEL"
[[ -n "$SLICE" ]] && BUILD_CMD="$BUILD_CMD --slice $SLICE"
[[ -n "$RESUME" ]] && BUILD_CMD="$BUILD_CMD $RESUME"

echo "Running: $BUILD_CMD"
echo ""

if $BUILD_CMD; then
    echo ""
    echo -e "${GREEN}✓ Memory bank built successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Memory building failed${NC}"
    exit 1
fi

# Step 2: Verify Structure
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 2: Verify Memory Structure${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if python scripts/verify_memory_structure.py --memory-root "$OUTPUT_DIR"; then
    echo ""
    echo -e "${GREEN}✓ Memory structure verified${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Memory structure verification failed (check warnings above)${NC}"
fi

# Step 3: Precompute Embeddings
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 3: Precompute Embeddings${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if python scripts/precompute_embeddings.py --memory-root "$OUTPUT_DIR"; then
    echo ""
    echo -e "${GREEN}✓ Embeddings precomputed${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Embedding computation failed (will compute on-the-fly during retrieval)${NC}"
fi

# Step 4: Quality Check
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 4: Quality Check${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

python3 << EOF
import json
from pathlib import Path

output = Path("$OUTPUT_DIR")

# Load memory files
l1 = json.loads((output / "failure_memory.json").read_text()) if (output / "failure_memory.json").exists() else []
l2 = json.loads((output / "repo_memory.json").read_text()) if (output / "repo_memory.json").exists() else []
l3 = json.loads((output / "cross_memory.json").read_text()) if (output / "cross_memory.json").exists() else []

print(f"Memory Statistics:")
print(f"  L1 (per-file):     {len(l1)} records")
print(f"  L2 (per-failure):  {len(l2)} records")
print(f"  L3 (principles):   {len(l3)} records")
print()

# Check L2 structure
if l2:
    cf = l2[0].get("changed_files", [])
    if cf and isinstance(cf[0], dict) and "fix_strategy" in cf[0]:
        print("✅ L2 structure: CORRECT (4-field objects)")
    else:
        print("❌ L2 structure: INCORRECT (strings or incomplete)")

    has_emb = sum(1 for r in l2 if "_embedding" in r or "embedding" in r)
    print(f"  Embeddings: {has_emb}/{len(l2)} records")
else:
    print("⚠️  No L2 records found")

print()

# Unique repos/commits
if l2:
    repos = {r.get("repo_name", r.get("repo", "?")) for r in l2}
    commits = {r.get("sha_fail", "?") for r in l2}
    print(f"Coverage:")
    print(f"  Repositories: {len(repos)}")
    print(f"  Unique commits: {len(commits)}")
EOF

echo ""

# Final Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✓ Memory rebuild complete!${NC}"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

if [[ "$TEST_MODE" == true ]]; then
    echo -e "${YELLOW}TEST MODE:${NC} Memory built for 1 issue only."
    echo ""
    echo "Next steps:"
    echo "  1. Inspect output:  ls -lh $OUTPUT_DIR/"
    echo "  2. Check structure: python scripts/verify_memory_structure.py --memory-root $OUTPUT_DIR"
    echo "  3. If good, rebuild all issues: ./scripts/rebuild_memory.sh"
else
    echo "Next steps:"
    echo "  1. Check similarities: python scripts/check_similarity.py --issue-1 407 --issue-2 410"
    echo "  2. Replace old memory: mv data/trs data/trs_OLD && mv $OUTPUT_DIR data/trs"
    echo "  3. Re-evaluate: scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2 --slice 0:15"
fi
echo ""
