# Simple Guide: Install Stable Package Versions

## The Problem
Package version conflicts cause errors like:
- `operator torchvision::nms does not exist`
- `No module named 'torch._inductor.kernel.flex_attention'`

## The Solution: Use Stable, Tested Versions

### One-Command Fix

On your server, run:

```bash
cd /home/ubuntu/Documents/rabeya/mini-swe-agent-ci-based
bash ./fix_torch_complete.sh
```

**This script will:**
1. Remove all conflicting packages
2. Install PyTorch 2.3.1 (CPU version) + matching torchvision/torchaudio
3. Install transformers 4.40.2 (NO torchao)  
4. Install sentence-transformers 2.7.0
5. Verify everything works

**Expected output:**
```
✓ PyTorch 2.3.1
✓ Torchvision 0.18.1
✓ Transformers 4.40.2
✓ Sentence-transformers loaded
  ✓ Model loaded
  ✓ Embeddings computed

SUCCESS! All packages working correctly.
```

### Why These Specific Versions?

| Package | Version | Reason |
|---------|---------|--------|
| PyTorch | 2.3.1 | Stable, no inductor issues |
| torchvision | 0.18.1 | Exact match for PyTorch 2.3.1 |
| transformers | 4.40.2 | Last version before torchao dependency |
| sentence-transformers | 2.7.0 | Compatible with transformers 4.40.2 |

These versions are **tested together** and guaranteed to work.

---

## Alternative: Fresh Virtual Environment

If the script still fails (due to system conflicts), create a fresh environment:

```bash
cd /home/ubuntu/Documents/rabeya/mini-swe-agent-ci-based

# Create new virtual environment
python3.12 -m venv .venv-fresh

# Activate it
source .venv-fresh/bin/activate

# Run the fix script
bash ./fix_torch_complete.sh

# Install other requirements
pip install -r requirements-stable.txt
```

---

## After Installation

**Test it works:**
```bash
source .venv-codex/bin/activate  # or .venv-fresh
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('SUCCESS!')
"
```

**Run your evaluation:**
```bash
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5
```

You should now see:
```
[Memory] Starting memory retrieval for X CI problems
[Memory]   Processing CI problem 1: ...
[Memory]     ✓ Enriched with repair strategy
```

Instead of:
```
ERROR: operator torchvision::nms does not exist
```

---

## Key Principle: Pin Everything

The root cause was **loose version constraints** in `requirements-codex.txt`:
- `torch>=2.0.0` → Could install ANY version ≥2.0.0
- `transformers>=4.35.0` → Could install incompatible versions

**Solution:** Use exact versions (`==`) in `requirements-stable.txt`:
- `torch==2.3.1` → Exact, tested version
- `transformers==4.40.2` → Exact, tested version

This eliminates version conflicts.

---

## Summary

1. **Run once:** `bash ./fix_torch_complete.sh`
2. **Test:** `python -c "from sentence_transformers import SentenceTransformer; print('OK')"`
3. **Use:** Run your evaluations with L1+L2+L3 memory enabled

No more version conflicts!
