#!/bin/bash
# Complete setup for running Codex via Harbor with any model

set -e

echo "=========================================="
echo "Codex Setup via Harbor"
echo "=========================================="
echo ""

# Step 1: Check .env file
echo "Step 1: Checking .env file..."
if [ ! -f .env ]; then
    echo "  ✗ .env file not found!"
    echo "  Create one with:"
    echo "    OPENAI_API_KEY=sk-dummy"
    echo "    OPENAI_BASE_URL=http://localhost:8000/v1"
    echo "    GLM_API_KEY=your-key"
    echo "    OPENROUTER_API_KEY=your-key"
    exit 1
fi

# Load .env
source .env

echo "  ✓ .env file found"
echo "    OPENAI_BASE_URL: ${OPENAI_BASE_URL:-not set}"
echo "    GLM_API_KEY: ${GLM_API_KEY:+set}"
echo "    OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+set}"
echo ""

# Step 2: Check Harbor installed
echo "Step 2: Checking Harbor installation..."
if ! command -v harbor &> /dev/null; then
    echo "  ✗ Harbor not found!"
    echo "  Install with: uv tool install harbor"
    exit 1
fi

echo "  ✓ Harbor installed: $(harbor --version 2>&1 || echo 'installed')"
echo ""

# Step 3: Check LiteLLM proxy running
echo "Step 3: Checking LiteLLM proxy..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✗ LiteLLM proxy not running!"
    echo "  Start with: ./start_litellm_proxy.sh"
    echo ""
    echo "  Starting proxy now..."
    ./start_litellm_proxy.sh &
    sleep 3

    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✗ Failed to start proxy!"
        exit 1
    fi
fi

echo "  ✓ LiteLLM proxy running at http://localhost:8000"
echo ""

# Step 4: Convert dataset
echo "Step 4: Preparing Harbor dataset..."
if [ ! -d "harbor_dataset" ]; then
    echo "  Converting HuggingFace dataset to Harbor format..."
    python3 convert_to_harbor_dataset.py --issue-ids 125,126,127 --output harbor_dataset
else
    echo "  ✓ harbor_dataset/ already exists"
fi
echo ""

echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Run Codex with:"
echo ""
echo "  # GLM 5.2"
echo "  harbor run -d harbor_dataset -a codex -m glm5.2"
echo ""
echo "  # Minimax"
echo "  harbor run -d harbor_dataset -a codex -m minimax"
echo ""
echo "  # Any model from litellm_config.yaml"
echo "  harbor run -d harbor_dataset -a codex -m <model-name>"
echo ""
