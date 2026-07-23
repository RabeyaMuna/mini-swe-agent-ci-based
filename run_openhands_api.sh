#!/bin/bash
# Quick start script for running CI-Bench with OpenHands API

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}  CI-Bench OpenHands API Runner${NC}"
echo -e "${GREEN}==============================================================${NC}"

# Check if OpenHands server is running
echo -e "\n${YELLOW}Checking OpenHands server...${NC}"
if curl -s http://localhost:3000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OpenHands server is running${NC}"
else
    echo -e "${RED}✗ OpenHands server is not running${NC}"
    echo -e "${YELLOW}Start it with: openhands start --port 3000${NC}"
    echo -e "${YELLOW}Or with Docker: docker run -it -p 3000:3000 ghcr.io/all-hands-ai/openhands:latest${NC}"
    exit 1
fi

# Default values
MODE=${MODE:-baseline}
SLICE=${SLICE:-}
EVAL_ISSUES=${EVAL_ISSUES:-data/trs/eval_set.jsonl}
DECOMPOSED=${DECOMPOSED:-data/trs/decomposed_issues.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/openhands-api}

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Eval Issues: $EVAL_ISSUES"
echo "  Output: $OUTPUT_DIR/$MODE"
if [ -n "$SLICE" ]; then
    echo "  Slice: $SLICE"
fi
if [ "$MODE" = "memory" ]; then
    echo "  Decomposed Issues: $DECOMPOSED"
fi

# Build command
CMD="python openhands/api_runner.py \
  --eval-issues $EVAL_ISSUES \
  --mode $MODE \
  --openhands-url http://localhost:3000 \
  --output $OUTPUT_DIR/$MODE"

if [ "$MODE" = "memory" ] && [ -f "$DECOMPOSED" ]; then
    CMD="$CMD --decomposed-issues $DECOMPOSED"
fi

if [ -n "$SLICE" ]; then
    CMD="$CMD --slice $SLICE"
fi

echo -e "\n${YELLOW}Running:${NC}"
echo "$CMD"
echo ""

# Run
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}==============================================================${NC}"
    echo -e "${GREEN}✓ Completed successfully${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "\nResults saved to: ${GREEN}$OUTPUT_DIR/$MODE/preds.json${NC}"
    echo -e "\nEvaluate with:"
    echo -e "  ${YELLOW}python scripts/evaluate_ablation_preds.py $OUTPUT_DIR/$MODE/preds.json${NC}"
else
    echo -e "\n${RED}==============================================================${NC}"
    echo -e "${RED}✗ Failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}==============================================================${NC}"
    exit $EXIT_CODE
fi
