# CI Repair System - Complete Setup Guide

## Quick Start (3 Commands)

```bash
# 1. Setup (one time)
./COMPLETE_SETUP.sh

# 2. Start proxy (Terminal 1 - keep running)
./start_litellm_proxy.sh

# 3. Run repair (Terminal 2)
./run_repair.sh baseline glm5.2 125
```

---

## One-Time Setup

### Run Setup Script

```bash
./COMPLETE_SETUP.sh
```

This will:
1. ✅ Check/create `.env` file
2. ✅ Verify API keys are set
3. ✅ Install `litellm[proxy]` and `python-dotenv`
4. ✅ Configure Codex to use LiteLLM proxy
5. ✅ Test setup

### Edit .env File (First Time)

```bash
nano .env
```

Add your API keys:

```bash
# Required for GLM models
GLM_API_KEY=your-actual-glm-key
GLM_BASE_URL=https://api.z.ai/api/paas/v4

# Required for Minimax
OPENROUTER_API_KEY=your-actual-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

## Running CI Repair

### Terminal 1: Start Proxy

```bash
./start_litellm_proxy.sh
```

**Keep this running!** You should see:

```
========================================
Starting LiteLLM Proxy
========================================
Endpoint: http://localhost:8000

Configured models:
  - glm5.2 (GLM_API_KEY: ✓ set)
  - minimax (OPENROUTER_API_KEY: ✓ set)
========================================

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Run Repair

```bash
# Single issue, baseline mode
./run_repair.sh baseline glm5.2 125

# Single issue, full memory
./run_repair.sh L1+L2+L3 glm5.2 125

# Single issue, minimax
./run_repair.sh baseline minimax 125

# Multiple issues
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125 126 127 \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --context-model glm5.2 \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model glm5.2"

# All issues from file
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --context-model glm5.2 \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model glm5.2"
```

---

## Available Models

| Model | API Key Required | Use Case |
|-------|-----------------|----------|
| `glm5.2` | `GLM_API_KEY` | Most capable GLM, recommended |
| `glm4` | `GLM_API_KEY` | Faster, cheaper GLM |
| `minimax` | `OPENROUTER_API_KEY` | Minimax 2.5 via OpenRouter |
| `deepseek-chat` | `DEEPSEEK_API_KEY` | DeepSeek (add key to .env) |
| `gpt-4` | `OPENAI_API_KEY` | OpenAI GPT-4 (add key to .env) |

---

## Available Modes

| Mode | Memory Used | Command Flag |
|------|-------------|--------------|
| Baseline | None | `--ablations baseline` |
| L1 | Repo+Workflow | `--ablations L1 --memory-root data/back_trs` |
| L1+L2 | Repo+Workflow+Repo-level | `--ablations L1+L2 --memory-root data/back_trs` |
| L1+L2+L3 | Full (+ Universal patterns) | `--ablations L1+L2+L3 --memory-root data/back_trs` |

---

## Troubleshooting

### Error: "model not supported"

**Cause:** LiteLLM proxy not running

**Fix:**
```bash
# Terminal 1
./start_litellm_proxy.sh
```

### Error: "Connection refused"

**Cause:** Proxy not started

**Fix:**
```bash
curl http://localhost:8000/health
# If fails, start proxy:
./start_litellm_proxy.sh
```

### Error: "Invalid API key"

**Cause:** API key not set in .env

**Fix:**
```bash
nano .env
# Add your actual API keys
# Re-run setup:
./COMPLETE_SETUP.sh
```

### Verify Everything Works

```bash
# 1. Check .env loaded
cat .env | grep GLM_API_KEY

# 2. Check proxy running
curl http://localhost:8000/health
# Should return: {"status":"ok"}

# 3. Check Codex config
cat ~/.config/codex/config.json
# Should show: "baseURL": "http://localhost:8000/v1"

# 4. Test Codex can reach proxy
echo "Test" | codex exec --model glm5.2 --sandbox none
# Should NOT error!
```

---

## Complete Example Session

```bash
# ====================
# ONE-TIME SETUP
# ====================

# 1. Run setup
./COMPLETE_SETUP.sh

# 2. Edit .env with your keys
nano .env

# ====================
# EVERY TIME YOU RUN
# ====================

# Terminal 1: Start proxy
./start_litellm_proxy.sh

# Terminal 2: Run repairs

# Single issue
./run_repair.sh baseline glm5.2 125
./run_repair.sh L1 glm5.2 125
./run_repair.sh L1+L2 minimax 125
./run_repair.sh L1+L2+L3 glm5.2 125

# Multiple issues, all modes
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125 126 127 \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --context-model glm5.2 \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model glm5.2"

# All issues from eval_issue_ids.json
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --context-model minimax \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model minimax"
```

---

## File Structure

```
mini-swe-agent-ci-based/
├── .env                      # Your API keys (created from .env.example)
├── .env.example              # Template
├── COMPLETE_SETUP.sh         # One-time setup script ⭐
├── setup_codex_config.sh     # Configure Codex for proxy
├── start_litellm_proxy.sh    # Start LiteLLM proxy (Terminal 1)
├── run_repair.sh             # Run single repair (Terminal 2)
├── litellm_config.yaml       # Model mappings
├── README_FINAL.md           # This file
└── codex/scripts/
    └── run_codex_ci_repair.py # Main script
```

---

## Summary

### Setup (one time):
```bash
./COMPLETE_SETUP.sh
nano .env  # Add your API keys
```

### Run (every time):
```bash
# Terminal 1
./start_litellm_proxy.sh

# Terminal 2
./run_repair.sh <mode> <model> <issue>
```

**That's it!** 🚀

---

## Quick Commands Reference

```bash
# ==================
# SINGLE ISSUE
# ==================
./run_repair.sh baseline glm5.2 125
./run_repair.sh L1 glm5.2 125
./run_repair.sh L1+L2 glm5.2 125
./run_repair.sh L1+L2+L3 glm5.2 125
./run_repair.sh baseline minimax 125

# ==================
# MULTIPLE ISSUES
# ==================
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125 126 127 \
    --ablations baseline \
    --context-model glm5.2 \
    --codex-command "codex exec --full-auto --model glm5.2"

# ==================
# ALL ABLATIONS
# ==================
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125 126 127 \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --context-model glm5.2 \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model glm5.2"

# ==================
# ALL ISSUES
# ==================
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --ablations baseline,L1,L1+L2,L1+L2+L3 \
    --context-model glm5.2 \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model glm5.2"
```
