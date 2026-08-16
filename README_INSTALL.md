# Installation Guide

## Quick Fix for Package Conflicts

### On Your Server:

```bash
cd /home/ubuntu/Documents/rabeya/mini-swe-agent-ci-based
bash ./reinstall.sh
```

That's it! This script:
1. Removes all conflicting packages
2. Installs PyTorch 2.3.1 (stable, CPU version)
3. Installs all other packages from `requirements-codex.txt`
4. Verifies everything works

---

## What Changed

**ONE file with stable versions:**
- ✅ `requirements-codex.txt` - Updated with exact versions (==)
- ❌ No more `requirements-stable.txt` or multiple fix scripts

**Key package versions:**
- PyTorch: `2.3.1` (stable, no conflicts)
- Transformers: `4.40.2` (NO torchao dependency)
- Sentence-transformers: `2.7.0` (compatible)

---

## If Reinstall Fails

Create a fresh virtual environment:

```bash
# Remove old environment
rm -rf .venv-codex

# Create fresh one
python3.12 -m venv .venv-codex

# Run reinstall
bash ./reinstall.sh
```

---

## Manual Installation (if script doesn't work)

```bash
source .venv-codex/bin/activate

# Step 1: Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install everything else
pip install -r requirements-codex.txt

# Step 3: Verify
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

---

## After Installation

Run your evaluation:

```bash
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5
```

You should see:
```
[Memory] Starting memory retrieval for X CI problems
✓ Enriched with repair strategy
```

Not:
```
ERROR: operator torchvision::nms does not exist
```

---

## Why Exact Versions?

**Before (Broken):**
```
torch>=2.0.0          # Could be 2.3, 2.5, 2.10 (incompatible!)
transformers>=4.35.0  # Could install 4.55 (needs torchao!)
```

**After (Working):**
```
torch==2.3.1          # Exact, tested
transformers==4.40.2  # Exact, tested
```

Exact versions (`==`) prevent conflicts and ensure reproducibility.
