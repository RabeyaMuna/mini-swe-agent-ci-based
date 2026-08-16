#!/bin/bash
set -e

echo "=========================================="
echo "Fixing tokenizers dependency conflict"
echo "=========================================="

# Activate venv
if [ -f .venv-codex/bin/activate ]; then
    source .venv-codex/bin/activate
    echo "✓ Activated .venv-codex"
else
    echo "❌ Error: .venv-codex not found"
    exit 1
fi

echo ""
echo "Current versions:"
pip list | grep -E "transformers|tokenizers|sentence-transformers"

echo ""
echo "Fixing tokenizers version..."
pip install 'tokenizers>=0.21.0,<0.22.0' --force-reinstall

echo ""
echo "Verifying installation..."
pip list | grep -E "transformers|tokenizers|sentence-transformers"

echo ""
echo "Testing import..."
python3 -c "
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
print('✓ All imports successful!')
"

echo ""
echo "=========================================="
echo "Fix complete!"
echo "=========================================="
echo ""
echo "You can now run your command:"
echo "bash ./run_miniswe_direct.sh \"\" L1+L2+L3 backward minimax2.5 \"\" data/eval_set.jsonl"
