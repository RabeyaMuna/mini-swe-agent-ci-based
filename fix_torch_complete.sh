#!/bin/bash
# Complete fix for torch/transformers compatibility
# Removes everything and installs matching versions

set -e  # Exit on error

echo "=========================================="
echo "Complete PyTorch/Transformers Fix"
echo "=========================================="
echo ""

# Activate virtual environment
source .venv-codex/bin/activate

echo "Step 1: Completely removing all torch/transformers packages..."
pip uninstall -y torch torchvision torchaudio transformers tokenizers sentence-transformers huggingface-hub safetensors accelerate torchao 2>/dev/null || true
echo ""

echo "Step 2: Installing matching PyTorch stack (CPU version)..."
# Install specific matching versions
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
echo ""

echo "Step 3: Installing compatible transformers (NO torchao)..."
# Install transformers version that doesn't depend on torchao
pip install transformers==4.40.2 tokenizers==0.19.1 --no-deps
echo ""

echo "Step 4: Installing transformers dependencies..."
pip install huggingface-hub==0.23.4 safetensors==0.4.3 pyyaml requests tqdm regex packaging
echo ""

echo "Step 5: Installing sentence-transformers..."
pip install sentence-transformers==2.7.0
echo ""

echo "Step 6: Verifying installation..."
python << 'PYEOF'
import sys

print("Checking installation...")
print("")

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
    sys.exit(1)

# Check torchvision
try:
    import torchvision
    print(f"✓ Torchvision {torchvision.__version__}")
except Exception as e:
    print(f"✗ Torchvision failed: {e}")
    sys.exit(1)

# Check transformers
try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers failed: {e}")
    sys.exit(1)

# Check sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    print(f"✓ Sentence-transformers loaded")

    # Test model loading
    print("  Loading test model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  ✓ Model loaded")

    # Test encoding
    embeddings = model.encode(['test sentence'])
    print(f"  ✓ Embeddings computed")

except Exception as e:
    print(f"✗ Sentence-transformers failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("")
print("="*50)
print("SUCCESS! All packages working correctly.")
print("="*50)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation Complete!"
    echo ""
    echo "You can now run your evaluation with memory retrieval."
else
    echo ""
    echo "✗ Installation Failed"
    echo ""
    echo "If this still doesn't work, there may be a system-level issue."
    echo "Try creating a fresh virtual environment:"
    echo "  python -m venv .venv-new"
    echo "  source .venv-new/bin/activate"
    echo "  bash ./fix_torch_complete.sh"
fi
