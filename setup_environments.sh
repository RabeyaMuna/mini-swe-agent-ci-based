#!/bin/bash
# Setup script for 3 virtual environments
# Run from project root: bash setup_environments.sh

set -e  # Exit on error

PROJECT_ROOT="/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based"

echo "========================================"
echo "  Multi-Agent CI-Bench Setup"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Setup ROOT .venv
echo -e "${YELLOW}[1/3] Setting up ROOT .venv (shared tools)${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Created .venv/"
else
    echo "✓ .venv/ already exists"
fi

source .venv/bin/activate
echo "✓ Activated ROOT .venv"

echo "  Installing shared tools..."
pip install -q -r requirements-shared.txt
echo -e "${GREEN}✓ ROOT .venv setup complete${NC}"
deactivate

# 2. Setup miniswe-agent/.venv
echo ""
echo -e "${YELLOW}[2/3] Setting up miniswe-agent/.venv${NC}"
cd miniswe-agent

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Created miniswe-agent/.venv"
else
    echo "✓ miniswe-agent/.venv already exists"
fi

source .venv/bin/activate
echo "✓ Activated miniswe-agent/.venv"

echo "  Installing mini-swe-agent..."
pip install -q -e .
echo "  Installing sentence-transformers for memory..."
pip install -q sentence-transformers
echo -e "${GREEN}✓ Mini-SWE-Agent setup complete${NC}"
deactivate

# 3. Setup openhands/.venv
echo ""
echo -e "${YELLOW}[3/3] Setting up openhands/.venv${NC}"
cd ../openhands

if [ ! -d ".venv" ]; then
    # Check if Python 3.12 is available
    if command -v python3.12 &> /dev/null; then
        python3.12 -m venv .venv
        echo "✓ Created openhands/.venv (Python 3.12)"
    else
        echo "⚠️  Python 3.12 not found, using default Python 3"
        python3 -m venv .venv
        echo "✓ Created openhands/.venv"
    fi
else
    echo "✓ openhands/.venv already exists"
fi

source .venv/bin/activate
echo "✓ Activated openhands/.venv"

echo "  Installing poetry..."
pip install -q poetry

echo "  Installing OpenHands (this may take a while)..."
poetry install --quiet || echo "⚠️  OpenHands installation incomplete (continue manually)"

echo -e "${GREEN}✓ OpenHands setup complete${NC}"
deactivate

# Summary
cd "$PROJECT_ROOT"
echo ""
echo "========================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Virtual environments created:"
echo "  ✓ .venv/                 (shared tools)"
echo "  ✓ miniswe-agent/.venv/   (Mini-SWE-Agent)"
echo "  ✓ openhands/.venv/       (OpenHands)"
echo ""
echo "Next steps:"
echo "  1. Activate ROOT venv:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Build memory:"
echo "     python scripts/decompose_ci_failure.py \\"
echo "       --eval-issues data/trs/eval_issues.json \\"
echo "       --output-dir data/trs"
echo ""
echo "  3. Run mini-swe-agent:"
echo "     cd miniswe-agent"
echo "     source .venv/bin/activate"
echo "     python -m minisweagent cibench \\"
echo "       --dataset ../data/trs/eval_set.jsonl \\"
echo "       --model minimax \\"
echo "       --slice 0:5 \\"
echo "       --output ../results/miniswe-agent/minimax/test"
echo ""
echo "See FINAL_SETUP_SUMMARY.md for complete guide"
echo "========================================"
