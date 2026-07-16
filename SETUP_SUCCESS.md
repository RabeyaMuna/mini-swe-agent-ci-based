# ✅ Setup Complete!

## What's Working

✅ ROOT `.venv/` - Installed successfully
  - sentence-transformers ✓
  - numpy, pandas ✓
  - matplotlib, jupyter ✓
  - All shared tools ✓

✅ `miniswe-agent/.venv/` - Installed successfully
  - mini-swe-agent package ✓
  - All dependencies ✓

## How to Use

### Activate Environments

```bash
# ROOT (for scripts, memory building, evaluation)
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Mini-SWE-Agent (for running experiments)
cd miniswe-agent
source .venv/bin/activate
```

### Commands Available

**In `miniswe-agent/.venv/`**:
- `mini` - Main CLI (same as `mini-swe-agent`)
- `mini-swe-agent` - Full command name
- `mini-extra` - Extra tools

**Check version**:
```bash
cd miniswe-agent
source .venv/bin/activate
mini --help
```

## Next Steps

1. **Read documentation**: See [FINAL_SETUP_SUMMARY.md](FINAL_SETUP_SUMMARY.md)

2. **Setup API keys**:
   ```bash
   cd miniswe-agent
   cp ../.env .env  # If you have .env at root
   # Or create new .env with your API keys
   ```

3. **Run first experiment** (following old docs):
   ```bash
   cd miniswe-agent
   source .venv/bin/activate
   
   # Check if cibench benchmark config exists
   ls src/minisweagent/config/benchmarks/cibench*.yaml
   
   # Run using config file directly
   mini -c src/minisweagent/config/benchmarks/cibench.yaml \
        --yolo
   ```

## Troubleshooting

### Issue: `mini` command not found

**Solution**: Make sure you're in miniswe-agent dir and venv is activated:
```bash
cd miniswe-agent
source .venv/bin/activate
which mini  # Should show: .../miniswe-agent/.venv/bin/mini
```

### Issue: Wrong venv activated

**Solution**: 
```bash
# Manually specify the python
./miniswe-agent/.venv/bin/mini --help
```

### Issue: Module not found

This is expected when using wrong venv. Each venv is isolated:
- ROOT venv: Has numpy, pandas, jupyter (NO minisweagent)
- miniswe-agent venv: Has minisweagent (NO jupyter by default)

## Success Indicators

✅ ROOT venv works:
```bash
source .venv/bin/activate
python -c "import sentence_transformers, numpy, pandas; print('OK')"
```

✅ Mini-SWE-Agent works:
```bash
cd miniswe-agent
source .venv/bin/activate
mini --help  # Shows usage
```

---

**Everything is set up!** Read FINAL_SETUP_SUMMARY.md for complete guide.
