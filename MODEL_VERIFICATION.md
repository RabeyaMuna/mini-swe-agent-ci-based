# Model Verification in Codex Runs

## Pre-Flight Check

Before running any evaluation, the script **validates the specified model actually works**.

This ensures:
- ✅ No silent fallback to other models
- ✅ Results are truly from the requested model
- ✅ Invalid model names fail fast (before running 60 issues)
- ✅ API authentication is working

## What Gets Checked

1. **Model Name**: Exact match verification
   - Request: `minimax/minimax-m2.5`
   - Response: Must return `minimax/minimax-m2.5`
   - If different → **FAILS** (prevents silent model substitution)

2. **API Connectivity**: Test call to the API
   - Sends a simple test message
   - Verifies response comes back
   - If error → **FAILS** (prevents running with broken API)

3. **Model Availability**: Confirms model exists
   - Invalid model names (e.g., `minimax-2.5` instead of `minimax/minimax-m2.5`)
   - Non-existent models
   - If not found → **FAILS**

## Example: Valid Model

```bash
$ ./run_codex_direct.sh 129 baseline backward minimax/minimax-m2.5

==========================================
PRE-FLIGHT CHECK: Testing model...
==========================================
Testing model: minimax/minimax-m2.5
API endpoint: https://openrouter.ai/api/v1
API key: sk-or-v1-62524d...

✓ Model responded successfully!
  Requested: minimax/minimax-m2.5
  Actual:    minimax/minimax-m2.5
  Response:  

✓ PRE-FLIGHT CHECK PASSED
  Confirmed: minimax/minimax-m2.5 is working correctly
==========================================

Running Codex...
```

## Example: Invalid Model

```bash
$ ./run_codex_direct.sh 129 baseline backward minimax-2.5

==========================================
PRE-FLIGHT CHECK: Testing model...
==========================================
Testing model: minimax-2.5
API endpoint: https://openrouter.ai/api/v1

✗ FATAL: Model test failed!
  Model:  minimax-2.5
  Error:  Error code: 400 - minimax-2.5 is not a valid model ID

The specified model cannot be used. Stopping now.
Fix the model name or API configuration before running.
```

**Script exits immediately** - does NOT proceed to "Running Codex..."

## Results Organization

Results are now **separated by model** in the output directory:

```
results/codex/
├── baseline_gpt-5-mini/           # GPT-5-mini results
│   ├── 129/
│   ├── 185/
│   └── predictions.json
│
├── baseline_minimax_minimax-m2_5/ # MiniMax 2.5 results
│   ├── 129/
│   ├── 185/
│   └── predictions.json
│
└── baseline_anthropic_claude-sonnet-4/  # Claude Sonnet 4 results
    ├── 129/
    └── predictions.json
```

## Comparing Models

After running evaluations:

```bash
# Run with MiniMax
./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5

# Run with GPT-5-mini
./run_codex_direct.sh "" baseline backward gpt-5-mini

# Compare results
diff -r results/codex/baseline_minimax_minimax-m2_5/ \
        results/codex/baseline_gpt-5-mini/
```

## Why This Matters

**Before this fix:**
- ❌ All models saved to same `baseline/` folder
- ❌ Results would overwrite each other
- ❌ No way to know if wrong model was used
- ❌ Silent failures if model name was wrong

**After this fix:**
- ✅ Each model has separate results folder
- ✅ Pre-flight check validates model works
- ✅ Script fails fast if model is invalid
- ✅ Model mismatch detection (requested vs actual)
- ✅ Can confidently compare model performance
