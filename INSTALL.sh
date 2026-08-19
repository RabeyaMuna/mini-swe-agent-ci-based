#!/bin/bash
# Installation script for CI Repair with Memory Plugin
# Creates a clean virtual environment with NO dependency conflicts
#
# Usage: bash install.sh

set -e  # Exit on error

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

# Always use .venv-codex as the environment name
VENV_NAME=".venv-codex"

echo -e "${GREEN}✓${NC} Virtual environment name: ${VENV_NAME}"
echo ""

# ============================================================================
# Step 4: Clean up old environments
# ============================================================================
echo -e "${YELLOW}[4/9]${NC} Cleaning up old environments..."

# Remove existing .venv-codex and other common venv names
for old_venv in ".venv-codex" ".venv"; do
    if [ -d "$old_venv" ]; then
        echo -e "   Removing old ${old_venv}..."
        rm -rf "$old_venv"
    fi
done
echo -e "${GREEN}✓${NC} Clean slate ready"
echo ""

# ============================================================================
# Step 5: Create virtual environment
# ============================================================================
echo -e "${YELLOW}[5/9]${NC} Creating .venv-codex virtual environment..."

python3.13 -m venv "$VENV_NAME"
echo -e "${GREEN}✓${NC} Virtual environment created: ${VENV_NAME}"
echo ""

# ============================================================================
# Step 6: Activate and upgrade pip
# ============================================================================
echo -e "${YELLOW}[6/9]${NC} Activating environment and upgrading pip..."

source "${VENV_NAME}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓${NC} Pip upgraded to $(pip --version | awk '{print $2}')"
echo ""

# ============================================================================
# Step 6.5: Remove conflicting packages
# ============================================================================
echo -e "${YELLOW}[6.5/9]${NC} Removing packages that conflict with scipy..."

# These packages require numpy>=2.0 which breaks scipy
CONFLICT_PACKAGES="fastembed ml-dtypes opencv-python opencv-python-headless"
for pkg in $CONFLICT_PACKAGES; do
    if pip show $pkg &> /dev/null; then
        echo "   Removing $pkg..."
        pip uninstall -y $pkg --quiet 2>/dev/null || true
    fi
done

echo -e "${GREEN}✓${NC} Conflicting packages removed"
echo ""

# ============================================================================
# Step 7: Install PyTorch (CPU version)
# ============================================================================
echo -e "${YELLOW}[7/9]${NC} Installing PyTorch (CPU version)..."
echo "   This may take a few minutes..."

# Install PyTorch components together with --upgrade to ensure version compatibility
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

if [ $? -eq 0 ]; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    TORCHVISION_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓${NC} PyTorch ${TORCH_VERSION} installed"
    echo -e "${GREEN}✓${NC} torchvision ${TORCHVISION_VERSION} installed"
else
    echo -e "${RED}✗ Failed to install PyTorch${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Step 7.5: CRITICAL - Force NumPy <2.0 (PyTorch pulls in 2.x)
# ============================================================================
echo -e "${YELLOW}[7.5/9]${NC} Forcing NumPy <2.0 (required for scipy)..."

pip install 'numpy>=1.26.0,<2.0.0' --force-reinstall --quiet

NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo -e "${RED}✗ NumPy still 2.x after force reinstall!${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} NumPy ${NUMPY_VERSION} (compatible with scipy)"
fi
echo ""

# ============================================================================
# Step 8: Install project dependencies
# ============================================================================
echo -e "${YELLOW}[8/9]${NC} Installing project dependencies..."
echo "   This may take a few minutes..."

pip install -r requirements-codex.txt --quiet

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
    cd miniswe-agent
    pip install -e . --quiet
    cd ..
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

python3 << 'VERIFY_EOF'
import sys

def check_package(name, import_name=None):
    """Check if a package is installed and get its version."""
    if import_name is None:
        import_name = name.replace("-", "_")

    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {name:25s} {version}")
        return True
    except Exception as e:
        print(f"✗ {name:25s} FAILED: {e}")
        return False

print("Core Dependencies:")
check_package("numpy")
check_package("scipy")
check_package("pandas")
check_package("scikit-learn", "sklearn")
print()

print("LLM & API:")
check_package("openai")
check_package("litellm")
check_package("langchain-openai", "langchain_openai")
print()

print("Embeddings & Transformers:")
check_package("torch")
check_package("transformers")
check_package("sentence-transformers", "sentence_transformers")
print()

print("Memory Plugin:")
try:
    from memory_plugin.memory_plugin import MemoryPlugin
    print("✓ memory_plugin            OK")
except Exception as e:
    print(f"✗ memory_plugin            FAILED: {e}")
print()

# Critical compatibility test
print("Critical Tests:")
try:
    import numpy as np
    import scipy.stats

    # Check numpy version
    if np.__version__.startswith("2."):
        print("✗ NumPy version             WRONG (2.x detected, need <2.0)")
        sys.exit(1)
    else:
        print(f"✓ NumPy version             OK ({np.__version__})")

    # Test scipy works with numpy
    scipy.stats.norm.pdf(0)
    print("✓ SciPy compatibility       OK")
except Exception as e:
    print(f"✗ Compatibility test        FAILED: {e}")
    sys.exit(1)

# Memory retrieval test
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(["Test sentence for memory retrieval"])
    print(f"✓ Memory retrieval          READY! (embeddings shape: {embeddings.shape})")
except Exception as e:
    print(f"✗ Memory retrieval          FAILED: {e}")
    sys.exit(1)

VERIFY_EOF

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
