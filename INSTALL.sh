#!/bin/bash
# Simple installation for Python 3.13
# Creates fresh environment and installs all dependencies

set -e

echo "=========================================="
echo "Fresh Installation - Python 3.13"
echo "=========================================="
echo ""

# 1. Remove old environment
if [ -d ".venv-codex" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv-codex
fi

# 2. Create fresh environment
echo "Creating fresh Python 3.13 virtual environment..."
python3.13 -m venv .venv-codex

# 3. Activate it
source .venv-codex/bin/activate

# 4. Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch
echo ""
echo "Step 1/3: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install from requirements
echo ""
echo "Step 2/3: Installing dependencies from requirements-codex.txt..."
pip install -r requirements-codex.txt

# 7. Install mini-swe-agent
echo ""
echo "Step 3/3: Installing mini-swe-agent..."
cd miniswe-agent
pip install -e .
cd ..

# 8. Verify
echo ""
echo "Verifying installation..."
python << 'EOF'
import sys
import torch
from sentence_transformers import SentenceTransformer

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

print("\nTesting sentence-transformers...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['test'])

print("\n" + "="*50)
print("✓ SUCCESS! All packages installed correctly.")
print("="*50)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "Installation complete!"
    echo ""
    echo "To activate: source .venv-codex/bin/activate"
    echo "To run: bash ./run_miniswe_direct.sh \"\" L1+L2+L3 backward minimax2.5"
else
    echo ""
    echo "Installation verification failed - see errors above"
    exit 1
fi
