#!/bin/bash
# Installation script for CI Repair with Memory Plugin
# Creates a clean virtual environment with NO dependency conflicts
#
# Usage: bash INSTALL.sh

set -e  # Exit on error

# Always operate on the checkout containing this script, even when INSTALL.sh
# is invoked through an absolute path from another working directory.
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   CI Repair Environment Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Step 1: Check Python version
# ============================================================================
echo -e "${YELLOW}[1/9]${NC} Checking Python version..."

if ! command -v python3.13 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3.13 not found!${NC}"
    echo "Please install Python 3.13 first."
    exit 1
fi

PYTHON_VERSION=$(python3.13 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} found"
echo ""

# ============================================================================
# Step 2: Deactivate current environment
# ============================================================================
echo -e "${YELLOW}[2/9]${NC} Checking current environment..."

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${YELLOW}⚠️  Deactivating current virtual environment: $(basename $VIRTUAL_ENV)${NC}"
    deactivate 2>/dev/null || true
fi
echo -e "${GREEN}✓${NC} Ready to create new environment"
echo ""

# ============================================================================
# Step 3: Set virtual environment name
# ============================================================================
echo -e "${YELLOW}[3/9]${NC} Setting virtual environment name..."

# Use .venv-codex normally. A different project-local name is useful for
# isolated installation checks without replacing the working environment.
VENV_NAME=${CI_REPAIR_VENV_NAME:-.venv-codex}
if [[ ! "$VENV_NAME" =~ ^\.venv-[A-Za-z0-9._-]+$ ]]; then
    echo -e "${RED}ERROR: CI_REPAIR_VENV_NAME must be a project-local .venv-* name.${NC}"
    exit 2
fi

echo -e "${GREEN}✓${NC} Virtual environment name: ${VENV_NAME}"
echo ""

# ============================================================================
# Step 4: Clean up old environments
# ============================================================================
echo -e "${YELLOW}[4/9]${NC} Cleaning up old environments..."

# Remove only the selected environment. Do not delete unrelated environments.
if [ -d "$VENV_NAME" ]; then
    echo -e "   Removing old ${VENV_NAME}..."
    rm -rf -- "$VENV_NAME"
fi
echo -e "${GREEN}✓${NC} Clean slate ready"
echo ""

# ============================================================================
# Step 5: Create virtual environment
# ============================================================================
echo -e "${YELLOW}[5/9]${NC} Creating ${VENV_NAME} virtual environment..."

python3.13 -m venv "$VENV_NAME"
echo -e "${GREEN}✓${NC} Virtual environment created: ${VENV_NAME}"
echo ""

# ============================================================================
# Step 6: Activate and upgrade pip
# ============================================================================
echo -e "${YELLOW}[6/9]${NC} Activating environment and upgrading pip..."

source "${VENV_NAME}/bin/activate"
pip install pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 --quiet
echo -e "${GREEN}✓${NC} Pip upgraded to $(pip --version | awk '{print $2}')"
echo ""

# ============================================================================
# Step 7: Install PyTorch (CPU version)
# ============================================================================
echo -e "${YELLOW}[7/9]${NC} Installing PyTorch (CPU version)..."
echo "   This may take a few minutes..."

# Uninstall any existing PyTorch first to avoid version conflicts
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Try PyTorch index first, fallback to PyPI if network issue
# PyTorch 2.5.0 is the latest stable version (not 2.13.0)
if pip install 'torch==2.5.0' 'torchvision==0.20.0' \
    --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Installed from PyTorch index"
else
    echo -e "${YELLOW}⚠️  PyTorch index unreachable, trying PyPI...${NC}"
    # Fallback to PyPI (may not have CPU-only wheels, but works on most systems)
    pip install 'torch==2.5.0' 'torchvision==0.20.0' --quiet
fi

if [ $? -eq 0 ]; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    TORCHVISION_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓${NC} PyTorch ${TORCH_VERSION} installed"
    echo -e "${GREEN}✓${NC} torchvision ${TORCHVISION_VERSION} installed"
else
    echo -e "${RED}✗ Failed to install PyTorch${NC}"
    echo -e "${YELLOW}Tip: Check your internet connection or use a proxy${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Step 7.5: Pin the Python 3.13 NumPy/SciPy-compatible range
# ============================================================================
echo -e "${YELLOW}[7.5/9]${NC} Pinning NumPy for Python 3.13 and SciPy..."

pip install 'numpy==2.2.6' --quiet

NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
echo -e "${GREEN}✓${NC} NumPy ${NUMPY_VERSION} (compatible with Python 3.13 and SciPy)"

# Keep the CPU PyTorch family and NumPy versions selected above stable while
# pip resolves the remaining project and editable-package dependencies.
CONSTRAINT_FILE="${VENV_NAME}/install-constraints.txt"
python3 - "$CONSTRAINT_FILE" << 'CONSTRAINTS_EOF'
from importlib.metadata import version
from pathlib import Path
import sys

packages = ("numpy", "torch", "torchvision")
Path(sys.argv[1]).write_text(
    "".join(f"{package}=={version(package)}\n" for package in packages),
    encoding="utf-8",
)
CONSTRAINTS_EOF
echo ""

# ============================================================================
# Step 8: Install project dependencies
# ============================================================================
echo -e "${YELLOW}[8/9]${NC} Installing project dependencies..."
echo "   This may take a few minutes..."

pip install -r requirements-codex.txt -c "$CONSTRAINT_FILE" --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All dependencies installed"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Step 9: Install mini-swe-agent
# ============================================================================
echo -e "${YELLOW}[9/9]${NC} Installing mini-swe-agent..."

if [ -d "miniswe-agent" ]; then
    pip install -e ./miniswe-agent -c "$CONSTRAINT_FILE" --quiet
    echo -e "${GREEN}✓${NC} mini-swe-agent installed"
else
    echo -e "${YELLOW}⚠️  miniswe-agent directory not found, skipping...${NC}"
fi
echo ""

# ============================================================================
# Verification
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Verifying Installation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

"${VENV_NAME}/bin/python" scripts/verify_installation.py
"${VENV_NAME}/bin/python" -m pip check

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}   Installation Complete!${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Virtual environment:${NC} ${GREEN}${VENV_NAME}${NC}"
    echo ""
    echo -e "${YELLOW}To activate this environment:${NC}"
    echo -e "  ${GREEN}source ${VENV_NAME}/bin/activate${NC}"
    echo "  (INSTALL.sh runs in a child shell, so activation does not persist.)"
    echo ""
    echo -e "${YELLOW}Or run a repository script without activating:${NC}"
    echo -e "  ${GREEN}${VENV_NAME}/bin/python scripts/split_before_decomposition.py${NC}"
    echo ""
    echo -e "${YELLOW}To run experiments:${NC}"
    echo -e "  ${GREEN}bash ./run_miniswe_direct.sh \"\" L1+L2+L3 backward minimax2.5 \"\" data/eval_set.jsonl 1${NC}"
    echo ""
    echo -e "${YELLOW}To deactivate:${NC}"
    echo -e "  ${GREEN}deactivate${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}   Installation Failed!${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Please check the errors above and try again.${NC}"
    exit 1
fi
