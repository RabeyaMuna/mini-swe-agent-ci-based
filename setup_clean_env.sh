#!/bin/bash
# Setup clean environment for CI Repair
# This script creates a fresh virtual environment with NO dependency conflicts

set -e  # Exit on error

echo "🔧 Setting up clean environment for CI Repair..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3.13 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# 1. Deactivate current environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "⚠️  Deactivating current virtual environment..."
    deactivate 2>/dev/null || true
fi

# 2. Remove old clean environment if exists
if [ -d ".venv-clean" ]; then
    echo "🗑️  Removing old .venv-clean..."
    rm -rf .venv-clean
fi

# 3. Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python3.13 -m venv .venv-clean

# 4. Activate it
echo "🔌 Activating environment..."
source .venv-clean/bin/activate

# 5. Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# 6. Install PyTorch (CPU version - much faster, no CUDA conflicts)
echo "🔥 Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

# 7. Install clean requirements
echo "📚 Installing compatible dependencies..."
pip install -r requirements-clean.txt --quiet

# 8. Install mini-swe-agent
echo "🤖 Installing mini-swe-agent..."
cd miniswe-agent
pip install -e . --quiet
cd ..

# 9. Verify installation
echo ""
echo "✅ Verifying installation..."
python3 << 'EOF'
import numpy
import scipy
import sentence_transformers
import transformers
import litellm
import openai

print(f"✅ NumPy: {numpy.__version__}")
print(f"✅ SciPy: {scipy.__version__}")
print(f"✅ sentence-transformers: {sentence_transformers.__version__}")
print(f"✅ transformers: {transformers.__version__}")
print(f"✅ litellm: {litellm.__version__}")
print(f"✅ openai: {openai.__version__}")

# Test memory imports
try:
    from memory_plugin.memory_plugin import MemoryPlugin
    print("✅ Memory plugin: OK")
except Exception as e:
    print(f"❌ Memory plugin: {e}")

# Test that scipy doesn't crash on numpy
try:
    import scipy.stats
    print("✅ SciPy stats: OK")
except Exception as e:
    print(f"❌ SciPy stats: {e}")
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use this environment:"
echo "  source .venv-clean/bin/activate"
echo ""
echo "Then run your experiments:"
echo "  bash ./run_miniswe_direct.sh \"\" L1+L2+L3 backward minimax2.5 \"\" data/eval_set.jsonl 1"
echo ""
