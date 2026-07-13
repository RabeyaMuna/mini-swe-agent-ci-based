# Installation Guide

## Quick Install (Recommended)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd mini-swe-agent-ci-based

# 2. Install with pip (installs ALL dependencies)
pip install -e .

# 3. Verify installation
python3 scripts/verify_installation.py

# 4. Configure API keys
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
EOF

# 5. Ready to run!
python3 scripts/run_eval.py --help
```

---

## What Gets Installed

### Core Dependencies:
- `pyyaml`, `requests`, `jinja2`
- `pydantic>=2.0`
- `litellm`, `openai`
- `rich`, `typer`, `python-dotenv`

### Memory & Embeddings (CRITICAL for L1+L2+L3):
- ✅ `sentence-transformers` - Embedding model
- ✅ `accelerate` - Required by sentence-transformers
- ✅ `scikit-learn` - Clustering & similarity
- ✅ `numpy` - Numerical operations

### Optional:
- `chromadb` - Vector database (can use JSON backend instead)
- `fastembed` - Alternative embedding library

---

## Manual Install (Alternative)

If `pip install -e .` doesn't work, install manually:

```bash
# Core
pip install pyyaml requests jinja2 pydantic litellm openai rich python-dotenv typer

# CRITICAL for L1+L2+L3
pip install sentence-transformers accelerate scikit-learn numpy

# Optional
pip install chromadb fastembed datasets
```

---

## Verify Installation

```bash
# Run verification script
python3 scripts/verify_installation.py

# Expected output:
# ✓ pyyaml
# ✓ requests
# ...
# ✓ sentence-transformers (embedding size: 384)
# ✓ accelerate
# ✓ scikit-learn
# ✓ numpy
# ✅ All dependencies installed correctly!
```

---

## Common Issues

### Issue: "No module named 'accelerate'"

**Fix:**
```bash
pip install accelerate
```

### Issue: "sentence-transformers load failed"

**Fix:**
```bash
pip install sentence-transformers accelerate
```

### Issue: "memory retrieval disabled"

**Cause:** Embedding packages not installed

**Fix:**
```bash
pip install sentence-transformers accelerate scikit-learn numpy
python3 scripts/verify_installation.py
```

---

## Server Installation

For server deployment, see [SERVER_DEPLOYMENT_GUIDE.md](docs/SERVER_DEPLOYMENT_GUIDE.md)

Quick server setup:

```bash
# On server
cd ~/mini-swe-agent-ci-based
git pull
pip install -e .
python3 scripts/verify_installation.py
```

---

## Development Installation

For development with testing tools:

```bash
pip install -e ".[dev]"

# Includes: pytest, pytest-cov, pytest-asyncio, ruff
```

---

## Minimal Installation (BASELINE only)

If you only need BASELINE (no memory), you can skip embedding packages:

```bash
pip install pyyaml requests jinja2 pydantic litellm openai rich python-dotenv typer datasets

# This works for BASELINE but NOT for L1/L1+L2/L1+L2+L3
```

---

## Requirements Files

Two options available:

1. **pyproject.toml** (recommended):
   ```bash
   pip install -e .
   ```

2. **requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```

Both install the same packages.

---

## Next Steps

After installation:

1. **Build memory:**
   ```bash
   bash scripts/run_memory_split_workflow.sh
   ```

2. **Test:**
   ```bash
   python3 scripts/run_eval.py --issue-ids 113,121,123 --ablation L1+L2+L3 --workers 1
   ```

3. **Run full evaluation:**
   ```bash
   python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3 --workers 4
   ```

---

## Help

- Installation issues: Run `python3 scripts/verify_installation.py`
- Server setup: See [SERVER_SETUP_QUICK.md](SERVER_SETUP_QUICK.md)
- Full guide: See [SERVER_DEPLOYMENT_GUIDE.md](docs/SERVER_DEPLOYMENT_GUIDE.md)
