#!/bin/bash
# Run evaluation on repos with 100% memory coverage
#
# This script runs codex or miniswe-agent on the memory-complete eval subset
# (122 issues from agno, axolotl, crewai, camel, django-import-export)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Memory-Complete Eval Set Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <tool> <model> [ablation] [num_workers]"
    echo ""
    echo "Arguments:"
    echo "  tool         : 'codex' or 'miniswe'"
    echo "  model        : Model name (e.g., 'gpt-5.4-mini', 'claude-sonnet-4.5')"
    echo "  ablation     : Ablation type (default: 'baseline')"
    echo "  num_workers  : Number of parallel workers (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 codex gpt-5.4-mini baseline 4"
    echo "  $0 miniswe claude-sonnet-4.5 memory 2"
    exit 1
fi

TOOL=$1
MODEL=$2
ABLATION=${3:-baseline}
NUM_WORKERS=${4:-1}

# Set eval file
EVAL_FILE="data/eval_set_memory_complete.jsonl"

if [ ! -f "$EVAL_FILE" ]; then
    echo -e "${YELLOW}⚠️  Eval file not found: $EVAL_FILE${NC}"
    echo "Run: python scripts/create_memory_complete_eval.py"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Tool:        $TOOL"
echo "  Model:       $MODEL"
echo "  Ablation:    $ABLATION"
echo "  Workers:     $NUM_WORKERS"
echo "  Eval file:   $EVAL_FILE"
echo ""

# Count issues
TOTAL_ISSUES=$(wc -l < "$EVAL_FILE" | tr -d ' ')
echo "  Total issues: $TOTAL_ISSUES (memory-complete repos only)"
echo ""

# Run based on tool
if [ "$TOOL" == "codex" ]; then
    echo -e "${GREEN}Running CODEX...${NC}"
    echo ""
    bash ./run_codex_direct.sh "" "$ABLATION" backward "$MODEL" "" "$EVAL_FILE" "$NUM_WORKERS"

elif [ "$TOOL" == "miniswe" ]; then
    echo -e "${GREEN}Running MINISWE-AGENT...${NC}"
    echo ""
    bash ./run_minisweagent.sh "" "$ABLATION" backward "$MODEL" "" "$EVAL_FILE" "$NUM_WORKERS"

else
    echo -e "${YELLOW}❌ Unknown tool: $TOOL${NC}"
    echo "Use 'codex' or 'miniswe'"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Evaluation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
