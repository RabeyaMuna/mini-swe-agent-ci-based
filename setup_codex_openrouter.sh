#!/bin/bash
# Setup Codex to work with OpenRouter (minimax, GLM, etc.)

set -e

echo "=========================================="
echo "Setting Up Codex for OpenRouter"
echo "=========================================="
echo ""

# Step 1: Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠ OPENROUTER_API_KEY not found in environment"
    echo ""
    echo "Please add it to .env file:"
    echo "  echo 'OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY' >> .env"
    echo ""
    echo "Or export it:"
    echo "  export OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY"
    echo ""

    # Check .env file
    if [ -f .env ] && grep -q "OPENROUTER_API_KEY" .env; then
        echo "✓ Found in .env file, loading..."
        set -a
        source .env
        set +a
    else
        exit 1
    fi
fi

echo "✓ OpenRouter API key found"
echo ""

# Step 2: Create Codex config
CODEX_CONFIG_DIR="$HOME/.codex"
CODEX_CONFIG_FILE="$CODEX_CONFIG_DIR/config.toml"

echo "Creating Codex configuration..."
mkdir -p "$CODEX_CONFIG_DIR"

cat > "$CODEX_CONFIG_FILE" << 'EOF'
# Codex configuration for OpenRouter

[provider.openai]
# Use OpenRouter endpoint for non-Anthropic models
api_base = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
EOF

echo "✓ Created $CODEX_CONFIG_FILE"
echo ""

# Step 3: Export environment variables for current session
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY

echo "✓ Environment configured for this session"
echo ""

# Step 4: Verify Codex is installed
if ! command -v codex &> /dev/null; then
    echo "⚠ Codex CLI not installed!"
    echo "Install with: uv tool install codex-cli"
    exit 1
fi

echo "✓ Codex CLI installed: $(codex --version)"
echo ""

echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run Codex with OpenRouter models:"
echo ""
echo "  # Using the wrapper:"
echo "  ./run_codex.sh minimax baseline backward 125"
echo ""
echo "  # Or directly:"
echo "  PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \\"
echo "    --issue-ids 125 \\"
echo "    --ablations baseline \\"
echo "    --codex-command 'codex exec --full-auto --model openrouter/minimax/minimax-01'"
echo ""
echo "Available models:"
echo "  - minimax         → openrouter/minimax/minimax-01"
echo "  - glm5.2          → openrouter/zhipuai/glm-4-plus"
echo "  - glm4            → openrouter/zhipuai/glm-4-0520"
echo "  - deepseek-chat   → openrouter/deepseek/deepseek-chat"
echo "  - gpt-4o          → openrouter/openai/gpt-4o"
echo ""
