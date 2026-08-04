#!/bin/bash
# Run Codex via Harbor using HuggingFace dataset directly
#
# Usage:
#   ./run_codex_hf.sh glm5.2 125,126,127
#   ./run_codex_hf.sh minimax 125

set -e

MODEL=${1:-minimax}
ISSUE_IDS=${2:-125}
ABLATION=${3:-baseline}
DIRECTION=${4:-backward}

echo "=========================================="
echo "Running Codex via Harbor (HuggingFace)"
echo "=========================================="
echo "Model: $MODEL"
echo "Issue IDs: $ISSUE_IDS"
echo "Ablation: $ABLATION"
echo "Direction: $DIRECTION"
echo ""

# Load environment
if [ -f .env ]; then
    source .env
fi

# Check proxy running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  LiteLLM proxy not running!"
    echo "Starting proxy..."
    ./start_litellm_proxy.sh &
    sleep 3
fi

echo "✓ LiteLLM proxy running"
echo ""

# Create temporary dataset file with selected issues
echo "Preparing dataset for issues: $ISSUE_IDS"
python3 -c "
from datasets import load_dataset
import json
from pathlib import Path

# Load dataset
ds = load_dataset('ci-benchmark-user/ci-repair-bench', split='train')

# Filter by issue IDs
issue_ids = [id.strip() for id in '$ISSUE_IDS'.split(',')]
filtered = [item for item in ds if str(item.get('id', '')) in issue_ids]

print(f'Found {len(filtered)} issues')

# Save to temp file
Path('temp_dataset.json').write_text(json.dumps(filtered, indent=2))
"

# Determine output directory
OUTPUT_DIR="results/codex/${ABLATION}_${MODEL//./_}_${DIRECTION}"

echo ""
echo "Starting Codex via Harbor..."
echo "Output: $OUTPUT_DIR"
echo ""

# Run Harbor with the dataset
# Harbor should be able to read the JSON directly
harbor run \
    --dataset temp_dataset.json \
    --agent codex \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --n-concurrent 1

echo ""
echo "=========================================="
echo "✓ Complete!"
echo "=========================================="
echo "Results: $OUTPUT_DIR/"

# Cleanup
rm -f temp_dataset.json
