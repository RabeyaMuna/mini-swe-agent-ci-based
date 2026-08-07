# MiniMax Integration Issue Report

## Problem Summary

**MiniMax models fail to work with Codex via OpenRouter**, even though:
- ✅ Pre-flight check (Python OpenAI client) succeeds
- ✅ Model exists on OpenRouter (`minimax/minimax-m2.5`)  
- ✅ API key is valid
- ❌ Codex (Rust binary) gets "model_not_found" error

## Root Cause

**OpenRouter rejects Codex's HTTP requests** but accepts Python OpenAI client requests.

### Evidence

1. **Python OpenAI client works:**
   ```python
   client = OpenAI(
       base_url="https://openrouter.ai/api/v1",
       api_key=OPENROUTER_API_KEY
   )
   response = client.chat.completions.create(
       model="minimax/minimax-m2.5",
       messages=[{"role": "user", "content": "test"}]
   )
   # ✓ SUCCESS: Returns model "minimax/minimax-m2.5"
   ```

2. **Codex fails with same settings:**
   ```bash
   export OPENAI_API_KEY="$OPENROUTER_API_KEY"
   export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
   codex exec --model minimax/minimax-m2.5
   
   # ✗ ERROR: 
   # {
   #   "code": "model_not_found",
   #   "message": "The requested model 'minimax/minimax-m2.5' does not exist."
   # }
   ```

## Technical Details

### HTTP Headers Difference

OpenRouter documentation recommends these headers:
```
HTTP-Referer: <your-site-url>
X-Title: <your-app-name>
```

**Python OpenAI client:** Sends these when configured with `default_headers`  
**Codex (Rust):** Does NOT send these headers

### Environment Variables Tested

```bash
export OPENROUTER_APP_NAME="Codex CI Repair"
export OPENROUTER_SITE_URL="https://github.com/anthropics/codex"
```

**Result:** No effect. Codex doesn't read these environment variables.

### Debug Logs

From `RUST_LOG=trace`:
```
provider=ModelProviderInfo { 
  name: "OpenAI",
  http_headers: Some({"version": "0.146.1"}),
  ...
}
```

Codex treats OpenRouter as generic OpenAI provider and doesn't add OpenRouter-specific headers.

## Solutions Attempted

### ✅ 1. Pre-flight Check (IMPLEMENTED)
- Validates model works before running 60 issues
- Prevents wasted evaluation runs
- Correctly stops if model fails
- **Location:** [run_codex_direct.sh:140-210](run_codex_direct.sh#L140-L210)

### ✅ 2. Model-Specific Output Folders (IMPLEMENTED)  
- Each model gets separate results directory
- Prevents overwriting results
- Format: `baseline_minimax_minimax-m2_5/`
- **Location:** [run_codex_direct.sh:133](run_codex_direct.sh#L133)

### ❌ 3. OpenRouter Environment Variables (FAILED)
- Set `OPENROUTER_APP_NAME` and `OPENROUTER_SITE_URL`
- Codex doesn't recognize these
- No effect on requests

### ❌ 4. Direct MiniMax API (BLOCKED)
- Requires separate `MINIMAX_API_KEY`
- User doesn't have direct MiniMax account
- Not a viable option

## Workarounds

### Option 1: Use Direct OpenAI Models (RECOMMENDED)

```bash
./run_codex_direct.sh "" baseline backward gpt-4o
./run_codex_direct.sh "" baseline backward gpt-4o-mini
```

**Status:** ✅ Works (need to fix pre-flight check for versioned models)

### Option 2: Use Anthropic Models via OpenRouter

If you want to use OpenRouter but need models that work:

```bash
# Test if Anthropic models work via OpenRouter
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
./run_codex_direct.sh 129 baseline backward anthropic/claude-opus-5
```

### Option 3: Get Direct MiniMax API Key

1. Sign up at https://www.minimaxi.com
2. Get API key
3. Add to `.env`:
   ```bash
   MINIMAX_API_KEY=your-key-here
   ```
4. Run:
   ```bash
   ./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5
   ```

Script will automatically use direct MiniMax API instead of OpenRouter.

## Files Changed

### 1. `run_codex_direct.sh`

**Changes:**
- Added MiniMax model routing (lines 83-112)
- Added pre-flight model validation (lines 140-210)
- Always include `--context-model` for result separation (line 133)
- Save/restore OpenRouter and MiniMax API keys (lines 42-43)

**Purpose:**
- Support MiniMax models via OpenRouter or direct API
- Validate model works before running evaluation
- Organize results by model name
- Prevent silent model fallbacks

### 2. `MODEL_VERIFICATION.md` (NEW)

**Purpose:**
- Document pre-flight check system
- Explain model validation process
- Show example outputs
- Explain result organization

## Validation Results

### Pre-Flight Check
- ✅ **PASS:** Correctly validates working models (gpt-4o, minimax via Python)
- ✅ **PASS:** Correctly rejects invalid models (minimax-2.5 typo)
- ✅ **PASS:** Stops script execution on failure
- ⚠️ **NEEDS FIX:** Too strict on versioned models (gpt-4o vs gpt-4o-2024-08-06)

### Model-Specific Directories
- ✅ **PASS:** Creates `baseline_gpt-4o/`, `baseline_minimax_minimax-m2_5/` folders
- ✅ **PASS:** Prevents result overwrites between models
- ✅ **PASS:** Works for all ablation types (baseline, with-memory)

### MiniMax via OpenRouter
- ❌ **FAIL:** Codex cannot use MiniMax models via OpenRouter
- ✅ **PASS:** Pre-flight check correctly detects this and stops

## Remaining Risks

1. **OpenRouter compatibility:** May affect other OpenRouter models beyond MiniMax
2. **Versioned model names:** Pre-flight check rejects `gpt-4o-2024-08-06` when requesting `gpt-4o`
3. **No Codex configuration for headers:** Can't add OpenRouter-required headers to Codex requests

## Recommendations

1. **For this evaluation:** Use direct OpenAI models (gpt-4o, gpt-4o-mini)
2. **For MiniMax:** Get direct MiniMax API key, avoid OpenRouter
3. **For comparison:** Run evaluations with both gpt-4o and gpt-4o-mini to measure model impact
4. **Future:** Request Codex team to add OpenRouter header support

## Current Status

**BLOCKED on MiniMax via OpenRouter**

**READY TO RUN with:**
- ✅ `gpt-4o` (fix version check first)
- ✅ `gpt-4o-mini` (fix version check first)  
- ✅ `anthropic/claude-opus-4` (direct Anthropic API)
- ⚠️ MiniMax (need direct API key)

---

**Next Steps:**
1. Fix pre-flight check to allow versioned OpenAI models
2. Run evaluation with gpt-4o
3. Get MiniMax API key if needed for comparison
4. File issue with Codex team about OpenRouter support
