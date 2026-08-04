#!/bin/bash
# Complete setup for running CI repair with any model

set -e

echo "=========================================="
echo "Complete Setup for CI Repair System"
echo "=========================================="
echo ""

# Step 1: Check .env file
echo "Step 1/5: Checking .env file..."
if [ ! -f .env ]; then
    echo "  ✗ .env file not found!"
    echo "  Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo "  ⚠️  IMPORTANT: Edit .env and add your API keys!"
    echo "  Required keys:"
    echo "    - GLM_API_KEY (for glm5.2, glm4)"
    echo "    - OPENROUTER_API_KEY (for minimax)"
    echo ""
    echo "  Run: nano .env"
    echo ""
    exit 1
else
    echo "  ✓ .env file exists"
fi
echo ""

# Step 2: Check API keys are set
echo "Step 2/5: Checking API keys..."
source .env

if [ -z "$GLM_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo "  ✗ No API keys configured in .env!"
    echo "  Edit .env and add at least GLM_API_KEY or OPENROUTER_API_KEY"
    exit 1
fi

echo "  Configured keys:"
if [ -n "$GLM_API_KEY" ]; then
    echo "    ✓ GLM_API_KEY (for glm5.2, glm4)"
fi
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "    ✓ OPENROUTER_API_KEY (for minimax)"
fi
if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo "    ✓ DEEPSEEK_API_KEY (for deepseek-chat)"
fi
if [ -n "$OPENAI_API_KEY" ]; then
    echo "    ✓ OPENAI_API_KEY (for gpt-4, gpt-3.5-turbo)"
fi
echo ""

# Step 3: Install Python dependencies
echo "Step 3/5: Installing Python dependencies..."
pip install -q litellm[proxy] python-dotenv
echo "  ✓ litellm[proxy] installed"
echo "  ✓ python-dotenv installed"
echo ""

# Step 4: Configure Codex
echo "Step 4/5: Configuring Codex..."
./setup_codex_config.sh > /dev/null
echo "  ✓ Codex configured to use LiteLLM proxy"
echo ""

# Step 5: Test setup
echo "Step 5/5: Testing setup..."

# Check if proxy is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ LiteLLM proxy is already running"
else
    echo "  ℹ️  LiteLLM proxy not running (start with: ./start_litellm_proxy.sh)"
fi
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start LiteLLM proxy (Terminal 1):"
echo "   ./start_litellm_proxy.sh"
echo ""
echo "2. Run repair (Terminal 2):"
echo "   ./run_repair.sh baseline glm5.2 125"
echo "   ./run_repair.sh L1+L2+L3 minimax 125"
echo ""
echo "Available models:"
echo "  - glm5.2     (requires GLM_API_KEY)"
echo "  - glm4       (requires GLM_API_KEY)"
echo "  - minimax    (requires OPENROUTER_API_KEY)"
echo "  - deepseek-chat (requires DEEPSEEK_API_KEY)"
echo "  - gpt-4      (requires OPENAI_API_KEY)"
echo ""
echo "=========================================="
