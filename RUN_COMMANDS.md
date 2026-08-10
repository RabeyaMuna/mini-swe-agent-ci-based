# How to Run CODEX CI Repair with GPT-5.4-mini

## Prerequisites

Ensure your `.env` file has:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

> **Security Note**: Never commit `.env` file to git. It's already in `.gitignore`.

## Quick Start (Recommended)

### Using the Wrapper Script
```bash
# Automatically loads .env and runs with GPT-5.4-mini
./run_gpt54.sh

# With custom parameters
./run_gpt54.sh baseline backward data/eval_set.jsonl 1
```

## Manual Commands

### Option 1: Auto-load from .env (Recommended)
```bash
# Source the .env file
set -a
source .env
set +a

# Run the command
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 1
```

### Option 2: Using shell export
```bash
# Load environment variables from .env
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)

# Run the command
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 1
```

## Running on Server

### Setup (one-time)
```bash
# SSH to server
ssh your-server

# Navigate to project
cd /path/to/mini-swe-agent-ci-based

# Sync latest code
git pull

# Ensure .env exists with OPENAI_API_KEY
# (Copy from secure location, do not commit to git)
```

### Run Command
```bash
# Use the wrapper script (easiest)
./run_gpt54.sh

# Or manually source .env
set -a; source .env; set +a
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 1
```

## Model Name Aliases

All these names map to the latest `gpt-5.4-mini`:
- `gpt-5.4-mini` ✓ Recommended
- `gpt-5.4`
- `gpt5.4`
- `gpt5.4-mini`

## Troubleshooting

### "Invalid API key" error
```bash
# Check if environment variable is set
echo $OPENAI_API_KEY

# Should output your API key (not empty)
# If empty, source .env again
```

### "Model not found" error
```bash
# Test API access
curl https://api.openai.com/v1/models/gpt-5.4-mini \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Should return: {"id": "gpt-5.4-mini", "object": "model", ...}
```

### Sync between local and server
```bash
# On local - commit code changes only (not .env!)
git add utilities/model_registry.py run_gpt54.sh
git commit -m "Update to use latest GPT-5.4-mini"
git push

# On server - pull changes
git pull

# Copy .env separately (secure method)
```

## Consistent Setup for Both Environments

1. ✅ `.env` file with `OPENAI_API_KEY` (keep secure, don't commit)
2. ✅ Same `model_registry.py` (sync via git)
3. ✅ Use wrapper script `./run_gpt54.sh` or source .env before running
4. ✅ Use `gpt-5.4-mini` in all commands
