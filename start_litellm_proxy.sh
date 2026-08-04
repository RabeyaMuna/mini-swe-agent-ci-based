#!/bin/bash
# Start LiteLLM proxy for all configured models

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "No .env file found. Using existing environment variables."
    echo "Copy .env.example to .env and configure your API keys."
    echo ""
fi

echo "========================================"
echo "Starting LiteLLM Proxy"
echo "========================================"
echo "Endpoint: http://localhost:8000"
echo ""
echo "Configured models:"
echo "  - glm5.2 (GLM_API_KEY: ${GLM_API_KEY:+✓ set}${GLM_API_KEY:-✗ not set})"
echo "  - minimax (OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+✓ set}${OPENROUTER_API_KEY:-✗ not set})"
echo "  - deepseek-chat (DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:+✓ set}${DEEPSEEK_API_KEY:-✗ not set})"
echo "  - gpt-4 (OPENAI_API_KEY: ${OPENAI_API_KEY:+✓ set}${OPENAI_API_KEY:-✗ not set})"
echo "========================================"
echo ""

# Check required keys
if [ -z "$GLM_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  WARNING: No API keys configured!"
    echo "Set at least GLM_API_KEY or OPENROUTER_API_KEY in .env file"
    echo ""
fi

# Start LiteLLM
litellm --config litellm_config.yaml --port 8000
