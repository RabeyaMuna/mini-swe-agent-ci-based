# Installation Guide

## Quick Installation (Python 3.13)

### On Your Server - One Command:

```bash
cd /home/ubuntu/Documents/rabeya/mini-swe-agent-ci-based
bash ./INSTALL.sh
```

This script:
1. ✅ Creates fresh Python 3.13 environment
2. ✅ Installs PyTorch (CPU version)
3. ✅ Installs all packages from `requirements-codex.txt`
4. ✅ Installs mini-swe-agent (`pip install -e .`)
5. ✅ Verifies everything works

---

## Manual Installation

If you prefer to install manually:

```bash
cd /home/ubuntu/Documents/rabeya/mini-swe-agent-ci-based

# 1. Create fresh environment
rm -rf .venv-codex
python3.13 -m venv .venv-codex
source .venv-codex/bin/activate

# 2. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install -r requirements-codex.txt

# 4. Install mini-swe-agent
cd miniswe-agent
pip install -e .
cd ..

# 5. Verify
python -c "from sentence_transformers import SentenceTransformer; print('✓ OK')"
```

---

## What's Installed

**From `requirements-codex.txt`:**
- PyTorch (latest CPU version)
- Transformers 4.40.2 (NO torchao)
- Sentence-transformers 2.7.0
- LiteLLM >=1.75.0,<1.82.7
- All other dependencies with exact versions

**Why exact versions?**
- Prevents version conflicts
- Ensures reproducibility
- All tested together

---

## After Installation

Run your evaluation:

```bash
source .venv-codex/bin/activate
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5
```

You should see:
```
[Memory] Starting memory retrieval for X CI problems
✓ Enriched with repair strategy
```

---

## Files

- `requirements-codex.txt` - Stable package versions
- `INSTALL.sh` - One-command installation
- `README_INSTALL.md` - This file
