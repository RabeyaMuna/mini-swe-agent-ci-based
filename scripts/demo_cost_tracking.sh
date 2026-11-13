#!/bin/bash
###############################################################################
# demo_cost_tracking.sh - Quick Demo of Offline Pipeline Cost Tracking
###############################################################################
#
# This script demonstrates the complete cost tracking workflow:
# 1. Process 3 test instances
# 2. Analyze costs
# 3. Display results
#
# Usage:
#   bash scripts/demo_cost_tracking.sh
#
###############################################################################

set -e  # Exit on error

# Configuration
MODEL="deepseek-v4-flash"
NUM_INSTANCES=3
OUTPUT_DIR="results/metrics/demo_$(date +%Y%m%d_%H%M%S)"
DATASET="data/cibench_test.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Offline Extraction Pipeline - Cost Tracking Demo                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}ERROR: Dataset not found: $DATASET${NC}"
    echo "Please ensure you have a test dataset at $DATASET"
    exit 1
fi

# Set pricing file environment variable
export RUN_METRICS_PRICING_FILE="config/model_pricing.json"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Instances: $NUM_INSTANCES"
echo "  Output: $OUTPUT_DIR"
echo "  Dataset: $DATASET"
echo ""

# Step 1: Process instances
echo -e "${GREEN}Step 1: Processing $NUM_INSTANCES instances...${NC}"
echo "────────────────────────────────────────────────────────────────────────"

python scripts/batch_offline_metrics.py \
    --dataset "$DATASET" \
    --num-instances "$NUM_INSTANCES" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Batch processing failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Batch processing complete${NC}"
echo ""

# Step 2: Analyze results
echo -e "${GREEN}Step 2: Analyzing costs...${NC}"
echo "────────────────────────────────────────────────────────────────────────"

python scripts/analyze_offline_costs.py \
    --batch-summary "$OUTPUT_DIR/batch_summary.json" \
    --detailed-tracker-dir "$OUTPUT_DIR/per_instance" \
    --output-dir "$OUTPUT_DIR/analysis"

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Cost analysis failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Cost analysis complete${NC}"
echo ""

# Step 3: Display results
echo -e "${GREEN}Step 3: Results Summary${NC}"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Display the visualization
cat "$OUTPUT_DIR/analysis/cost_visualization.txt"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}Demo Complete!${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. View detailed analysis: cat $OUTPUT_DIR/analysis/cost_analysis.json"
echo "  2. Open CSV in spreadsheet: open $OUTPUT_DIR/batch_costs.csv"
echo "  3. Run on 20 instances for accurate projections:"
echo "     python scripts/batch_offline_metrics.py \\"
echo "       --dataset $DATASET \\"
echo "       --num-instances 20 \\"
echo "       --model $MODEL \\"
echo "       --output-dir results/metrics/batch_20"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
