#!/bin/bash
# Fix PyTorch/transformers/torchao compatibility issue
# The issue: torchao (dependency of transformers) is incompatible with PyTorch CPU version

echo "=== Fixing PyTorch/transformers compatibility issue ==="
echo ""

# Activate virtual environment
source .venv-codex/bin/activate

echo "Step 1: Uninstalling problematic packages..."
pip uninstall -y torchao transformers tokenizers sentence-transformers

echo ""
echo "Step 2: Installing compatible versions..."
# Install transformers without torchao dependency
pip install 'transformers==4.46.3' --no-deps
pip install 'tokenizers==0.20.3'
pip install 'sentence-transformers==3.3.1'

# Reinstall missing transformers dependencies (without torchao)
pip install 'huggingface-hub>=0.16.4'
pip install 'numpy>=1.17'
pip install 'packaging>=20.0'
pip install 'pyyaml>=5.1'
pip install 'regex!=2019.12.17'
pip install 'requests'
pip install 'safetensors>=0.4.1'
pip install 'tqdm>=4.27'

echo ""
echo "Step 3: Verifying installation..."
python -c "
try:
    from sentence_transformers import SentenceTransformer
    print('✓ sentence-transformers loaded successfully')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✓ Model loaded successfully')
    embeddings = model.encode(['test'])
    print('✓ Embeddings computed successfully')
    print('')
    print('SUCCESS: Installation fixed!')
except Exception as e:
    print(f'✗ Error: {e}')
    print('')
    print('FAILED: Please try the alternative fix below')
"

echo ""
echo "=== Fix complete ==="
echo ""
echo "Alternative fix if above doesn't work:"
echo "1. Reinstall PyTorch with full CUDA support (not CPU-only):"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "2. Or use a different embedding model that doesn't depend on transformers:"
echo "   Edit memory_plugin/stair_retrieval.py and change embedding_model to use sklearn instead"
