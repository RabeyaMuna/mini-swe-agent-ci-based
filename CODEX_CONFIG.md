# Codex Configuration - Project vs Global

## Overview

This project uses **project-local** Codex configuration to prevent API key leakage to other projects.

## Global Codex Config (`~/.codex/`)

**Purpose**: Used by all OTHER Codex projects on your machine

**Current Setup**:
- **Auth Mode**: `chat_auth` (ChatGPT Plus)
- **No API Keys**: Clean configuration, no OpenAI or OpenRouter keys
- **Location**: 
  - `~/.codex/config.toml`
  - `~/.codex/auth.json`

**What this means**: When you use Codex in other projects, it uses your ChatGPT Plus subscription without consuming API credits.

---

## Project-Local Config (`.codex-local/`)

**Purpose**: Used ONLY for this project (mini-swe-agent-ci-based)

**Location**: 
- `/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/.codex-local/`
- Set via `CODEX_HOME` environment variable in `run_codex_direct.sh`

**How it works**:
1. `run_codex_direct.sh` sets: `export CODEX_HOME="$PWD/.codex-local"`
2. Creates `.codex-local/config.toml` and `.codex-local/auth.json`
3. Uses API keys from `.env` file in this project

**Configuration by Model**:

### GPT-5-mini / GPT-5.4-mini-2026-03-17
```toml
# .codex-local/config.toml
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"
```

```json
// .codex-local/auth.json
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "<from .env file>"
}
```

### minimax/minimax-m2.5 (via OpenRouter)
```toml
# .codex-local/config.toml
model_provider = "openrouter"
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
requires_openai_auth = true
```

```json
// .codex-local/auth.json
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "<OPENROUTER_API_KEY from .env>"
}
```

---

## Required .env Variables

Create `.env` in project root with:

```ini
# OpenAI API (for GPT-5 models)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# OpenRouter (for MiniMax M2.5)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional: GLM for decomposition only
GLM_API_KEY=...
```

---

## Security

✅ `.codex-local/` is in `.gitignore` (API keys won't be committed)
✅ Global config has NO API keys (safe for other projects)
✅ API keys only in `.env` file (already in `.gitignore`)

---

## Verification

### Check global config (should use ChatGPT Plus):
```bash
cat ~/.codex/auth.json
# Should show: {"auth_mode": "chat_auth"}
```

### Check project-local config (created after first run):
```bash
cat .codex-local/auth.json
# Should show: {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-..."}
```

### Verify environment variable:
```bash
bash run_codex_direct.sh "" baseline backward gpt-5-mini "" data/eval_set.jsonl 1
# Look for banner: "CONFIG: ... codex_home=<path>/.codex-local (project-local) ..."
```

---

## Backups

Global config backups created on 2026-08-08:
- `~/.codex/auth.json.backup-20260808-*`
- `~/.codex/config.toml.backup-20260808-*`

To restore old global config (if needed):
```bash
cp ~/.codex/auth.json.backup-<timestamp> ~/.codex/auth.json
cp ~/.codex/config.toml.backup-<timestamp> ~/.codex/config.toml
```

---

## Summary

| Location | Auth | Models | Keys Source | Used By |
|----------|------|--------|-------------|---------|
| `~/.codex/` | ChatGPT Plus | Any | None | Other projects |
| `.codex-local/` | API Key | GPT-5-mini, GPT-5.4-mini, MiniMax M2.5 | `.env` file | This project only |

**Result**: 
- ✅ This project uses OpenAI/OpenRouter API keys from `.env`
- ✅ Other projects use ChatGPT Plus (no API charges)
- ✅ No configuration leakage between projects
