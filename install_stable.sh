#!/bin/bash
# Install stable, tested package versions
# This avoids version conflicts and compatibility issues

set -e  # Exit on error

echo "=========================================="
echo "Installing Stable Package Versions"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv-codex" ]; then
    echo "Error: Virtual environment .venv-codex not found"
    echo "Please create it first with: python -m venv .venv-codex"
    exit 1
fi

# Activate virtual environment
echo "Step 1: Activating virtual environment..."
source .venv-codex/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Uninstall problematic packages first
echo "Step 2: Removing potentially conflicting packages..."
pip uninstall -y torch torchvision torchaudio transformers tokenizers sentence-transformers litellm openai 2>/dev/null || true
echo ""

# Install from stable requirements
echo "Step 3: Installing stable package versions..."
pip install -r requirements-stable.txt
echo ""

# Verify critical packages
echo "Step 4: Verifying installation..."
python -c "
import sys

print('Checking critical packages...')
print('')

# Check PyTorch
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
except ImportError as e:
    print(f'✗ PyTorch failed: {e}')
    sys.exit(1)

# Check transformers
try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except ImportError as e:
    print(f'✗ Transformers failed: {e}')
    sys.exit(1)

# Check sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    print(f'✓ Sentence-transformers loaded')
    # Test model loading
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f'✓ Model loaded successfully')
    # Test encoding
    embeddings = model.encode(['test'])
    print(f'✓ Embeddings computed successfully')
except Exception as e:
    print(f'✗ Sentence-transformers failed: {e}')
    sys.exit(1)

# Check LLM libraries
try:
    import openai
    print(f'✓ OpenAI {openai.__version__}')
except ImportError as e:
    print(f'✗ OpenAI failed: {e}')
    sys.exit(1)

try:
    import litellm
    print(f'✓ LiteLLM {litellm.__version__}')
except ImportError as e:
    print(f'✗ LiteLLM failed: {e}')
    sys.exit(1)

print('')
print('━' * 50)
print('SUCCESS: All packages installed correctly!')
print('━' * 50)
"

VERIFICATION_STATUS=$?

echo ""
echo "=========================================="
if [ $VERIFICATION_STATUS -eq 0 ]; then
    echo "✓ Installation Complete"
    echo "=========================================="
    echo ""
    echo "All packages are installed and working!"
    echo ""
    echo "You can now run:"
    echo "  bash ./run_miniswe_direct.sh \"\" L1+L2+L3 backward minimax2.5"
else
    echo "✗ Installation Failed"
    echo "=========================================="
    echo ""
    echo "Some packages failed to install or verify."
    echo "Check the error messages above."
    exit 1
fi
