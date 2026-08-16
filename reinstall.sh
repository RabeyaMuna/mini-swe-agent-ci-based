#!/bin/bash
# Reinstall packages with stable versions from requirements-codex.txt
# Run this to fix package version conflicts

set -e  # Exit on error

echo "=========================================="
echo "Reinstalling with Stable Versions"
echo "=========================================="
echo ""

# Activate virtual environment
if [ ! -f ".venv-codex/bin/activate" ]; then
    echo "Error: Virtual environment .venv-codex not found"
    echo "Create it first with: python3.12 -m venv .venv-codex"
    exit 1
fi

source .venv-codex/bin/activate

echo "Step 1: Removing conflicting packages..."
pip uninstall -y torch torchvision torchaudio transformers tokenizers sentence-transformers huggingface-hub safetensors torchao 2>/dev/null || true
echo ""

echo "Step 2: Installing PyTorch (stable version)..."
echo "NOTE: You'll see warnings about missing packages - this is normal!"
echo "      Step 3 will install them."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
echo ""

echo "Step 3: Installing remaining packages from requirements-codex.txt..."
echo "This will fix the warnings from Step 2..."
pip install -r requirements-codex.txt
echo ""

echo "Step 4: Verifying installation..."
python << 'EOF'
import sys

print("Verifying critical packages...")
print("")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")

    import transformers
    print(f"✓ Transformers {transformers.__version__}")

    from sentence_transformers import SentenceTransformer
    print(f"✓ Sentence-transformers loaded")

    # Quick test
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(['test'])
    print(f"✓ Model works correctly")

    print("")
    print("="*50)
    print("SUCCESS! All packages installed correctly.")
    print("="*50)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation Complete!"
    echo ""
    echo "You can now run your evaluation:"
    echo "  bash ./run_miniswe_direct.sh \"\" L1+L2+L3 backward minimax2.5"
else
    echo ""
    echo "✗ Installation Failed - see errors above"
    exit 1
fi
