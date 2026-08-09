# CI‑Repair Bench – Clean Setup & Run (Mini‑SWE + Codex)

This project runs two agents over a CI‑repair benchmark, with or without memory, across three models. One agent at a time.
- mini‑swe‑agent
- codex (OpenAI Codex CLI)

Supported agent models (exact list):
- gpt-5-mini (OpenAI)
- gpt-5.4-mini-2026-03-17 (OpenAI snapshot)
- minimax/minimax-m2.5 (MiniMax M2.5 via OpenRouter)

Ablations: BASELINE, L1, L1+L2, L1+L2+L3
Directions: backward (data/back_trs) or forward (data/fwr_trs)

Notes
- GLM 5.2 may be used only for decomposition/memory build, not as an agent model.
- predictions.json is updated incrementally after each issue so crashes do not lose patches.

## 0) Install Once

### macOS / Local Development Setup

- Create env and install shared deps
```bash
python3 -m venv .venv-codex
source .venv-codex/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements-shared.txt -r requirements-codex.txt python-dotenv litellm
# Install Mini‑SWE‑Agent (for the Mini‑SWE runner)
pip install -e miniswe-agent
```

- Install Codex CLI
```bash
npm install -g @openai/codex-cli
codex --version
```

- Put keys in .env at repo root
```ini
# OpenAI (for GPT‑5 models)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# OpenRouter (for MiniMax M2.5)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM-5.2 via Z.ai (optional, for decomposition/memory build)
GLM_API_KEY=...
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL_NAME=glm-5.2

# HuggingFace (for dataset loading)
HUGGINGFACE_TOKEN=hf_...
```

### Ubuntu Server / Remote Setup

**Prerequisites:**
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- Node.js 18+ (for Codex CLI)
- Git

**Step 1: Clone Repository**
```bash
# SSH into your server
ssh ubuntu@your-server-ip

# Clone the repo
cd ~/Documents
git clone https://github.com/your-username/mini-swe-agent-ci-based.git
cd mini-swe-agent-ci-based
```

**Step 2: Install System Dependencies**
```bash
# Update package list
sudo apt update

# Install Python 3.10+ and pip
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install Node.js 18+ (for Codex CLI)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Git if not present
sudo apt install -y git curl
```

**Step 3: Create Python Virtual Environment**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip wheel
pip install -r requirements-shared.txt -r requirements-codex.txt python-dotenv litellm

# Install Mini-SWE-Agent
pip install -e miniswe-agent
```

**Step 4: Install Codex CLI**
```bash
# Install globally
npm install -g @openai/codex-cli

# Verify installation
codex --version
```

**Step 5: Configure Environment Variables**
```bash
# Create .env file (copy API keys from your local setup)
cat > .env << 'EOF'
# OpenAI API Key
OPENAI_API_KEY=sk-proj-...
OPENAI_BASE_URL=https://api.openai.com/v1

# OpenRouter API Key (for MiniMax M2.5)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM-5.2 via Z.ai Platform
GLM_API_KEY=...
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL_NAME=glm-5.2

# HuggingFace Token
HUGGINGFACE_TOKEN=hf_...
EOF

# Secure the .env file
chmod 600 .env
```

**Step 6: Configure Codex (Ubuntu-specific)**
```bash
# Create Codex configuration directory
mkdir -p .codex

# Create config.toml for Codex
cat > .codex/config.toml << 'EOF'
model_reasoning_effort = "high"
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[shell_environment_policy]
inherit = "all"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
requires_openai_auth = true
EOF
```

**Step 7: Verify Installation**
```bash
# Activate environment
source .venv/bin/activate

# Test Python dependencies
python3 -c "import litellm; print('✓ LiteLLM installed')"

# Test Codex CLI
codex --version

# Verify environment variables
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓ OPENAI_API_KEY:', 'set' if os.getenv('OPENAI_API_KEY') else 'missing')"
```

**Step 8: Running Jobs in Background (Recommended)**

Since server tasks can take hours, use `tmux` or `screen`:

```bash
# Install tmux
sudo apt install -y tmux

# Create a new session
tmux new -s decompose

# Run your command
source .venv/bin/activate
python3 scripts/decompose_backward.py --batch --use-huggingface --dataset data/memory_set.jsonl --model gpt-5-mini --output-dir data/back_trs

# Detach from session: Press Ctrl+B, then D
# Reattach later: tmux attach -t decompose
# List sessions: tmux ls
```

**Differences from macOS:**
- Use `bash` instead of `zsh` (though both work)
- Paths are `/home/ubuntu/...` instead of `/Users/...`
- Use `apt` for system packages instead of `brew`
- Configure firewall if running LiteLLM proxy: `sudo ufw allow 8000`
- For long-running tasks, use `tmux`, `screen`, or `nohup`

## 1) Prepare Data (Split → Decompose → Build Memory)

**Step 1: Split dataset into memory/eval**
```bash
python3 scripts/split_before_decomposition.py
```
Creates `data/memory_set.jsonl` and `data/eval_set.jsonl` (plus ID lists).

**Step 2: Backward decomposition + memory build (automatic)**
```bash
python3 scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --output-dir data/back_trs
```
This automatically:
- Decomposes CI failures into problems
- Builds L1/L2/L3 memory files
- Saves to **`data/back_trs/`**:
  - `decomposed_issues.json` (decomposed problems)
  - `failure_memory.json` (L1 - concrete failures)
  - `repo_memory.json` (L2 - repair strategies)
  - `cross_memory.json` (L3 - universal patterns)

**Step 3: Forward decomposition + memory build (optional)**
```bash
python3 scripts/decompose_commits.py \
  --batch \
  --use-huggingface \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --output-dir data/fwr_trs
```
Similarly, this automatically builds forward memory and saves to **`data/fwr_trs/`**.

**Note:** 
- You do NOT need to run `build_memory_l1_l2_l3.py` separately
- Decomposition scripts handle everything in one command
- `data/back_trs/` = backward traces (CI failure → problem)
- `data/fwr_trs/` = forward traces (commit → problem)

### Running Decomposition on Ubuntu Server

**For specific repositories (e.g., CAMEL issues only):**

```bash
# SSH into server
ssh ubuntu@your-server-ip
cd /home/ubuntu/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Create filtered dataset for specific repo
python3 -c "
import json
camel_issues = []
with open('data/memory_set.jsonl') as f:
    for line in f:
        data = json.loads(line)
        repo_name = data.get('repo_name', '').lower()
        repo_owner = data.get('repo_owner', '').lower()
        if 'camel' in repo_name or 'camel' in repo_owner:
            camel_issues.append(data)
with open('data/camel_memory_set.jsonl', 'w') as f:
    for issue in camel_issues:
        f.write(json.dumps(issue) + '\n')
print(f'Created data/camel_memory_set.jsonl with {len(camel_issues)} issues')
"

# Run decomposition in background using tmux
tmux new -s decompose
python3 scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --dataset data/camel_memory_set.jsonl \
  --model gpt-5-mini \
  --output-dir data/back_trs
# Detach: Ctrl+B then D

# Or using nohup
nohup python3 scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --dataset data/camel_memory_set.jsonl \
  --model gpt-5-mini \
  --output-dir data/back_trs > decompose.log 2>&1 &

# Monitor progress
tail -f decompose.log
```

**Supported models for decomposition:**
- `gpt-5-mini` (recommended for production)
- `minimax2.5` (cost-effective alternative)
- `glm5.2` (requires GLM_API_KEY in .env)

## 2) Run Mini‑SWE‑Agent (evaluation runner)

Syntax
```bash
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos <comma repos> \
  --ablation <BASELINE|L1|L1+L2|L1+L2+L3> \
  --model <minimax2.5|gpt-5-mini|gpt-5.4-mini-2026-03-17> \
  --direction <backward|forward> \
  --workers <N>
```

Examples
```bash
# Baseline (no memory), backward, MiniMax
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,camel \
  --ablation BASELINE \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# Baseline, backward (3 repos)
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,flower,camel \
  --ablation BASELINE \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# Full memory L1+L2+L3, backward
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,camel \
  --ablation L1+L2+L3 \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# L1+L2, backward (3 repos)
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,flower,camel \
  --ablation L1+L2 \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# L1 only, backward (3 repos)
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,flower,camel \
  --ablation L1 \
  --model minimax2.5 \
  --direction backward \
  --workers 4
```
Notes
- Use one agent at a time (Mini‑SWE or Codex, not both concurrently).
- --workers controls parallel issues per run (default 1). Try 4 or 6 on bigger machines.
- Valid Mini‑SWE models: minimax2.5, gpt‑5‑mini, gpt‑5.4‑mini‑2026‑03‑17.

## 3) Run Codex (OpenAI Codex CLI agent)

Key points
- The wrapper auto‑configures the provider per model and writes ~/.codex on the fly.
- Auth banner prints: auth mode, provider, endpoint, and CODEX_HOME.
- Supported models: gpt‑5‑mini, gpt‑5.4‑mini‑2026‑03‑17, minimax/minimax‑m2.5.
- Aliases: passing minimax or minimax2.5 is routed to minimax/minimax‑m2.5 automatically.

### Quick Start (Ubuntu/macOS)

**Simple wrapper script (recommended):**
```bash
# Format: ./run_codex_simple.sh <model> <ablation> <direction> <issue-ids>

# Example 1: Baseline (no memory) with GPT-5 mini on specific issues
./run_codex_simple.sh gpt-5-mini baseline backward 416,407,408

# Example 2: Full memory (L1+L2+L3) with MiniMax
./run_codex_simple.sh minimax2.5 L1+L2+L3 backward 416,407,408

# Example 3: Single issue for debugging
./run_codex_simple.sh gpt-5-mini baseline backward 416
```

**On Ubuntu server (in background):**
```bash
# Using tmux (recommended)
tmux new -s codex
source .venv/bin/activate
./run_codex_simple.sh gpt-5-mini L1+L2+L3 backward 416,407,408
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t codex

# Or using nohup
nohup ./run_codex_simple.sh gpt-5-mini baseline backward 416,407,408 > codex.log 2>&1 &
tail -f codex.log
```

**View results:**
```bash
# Results are in: results/codex/<ablation>_<model>/<issue-id>/
ls -la results/codex/baseline_gpt-5-mini/416/

# View specific result
cat results/codex/baseline_gpt-5-mini/416/result.json | python3 -m json.tool

# View generated patch
cat results/codex/baseline_gpt-5-mini/416/patch.diff
```

Syntax
```bash
bash ./run_codex_direct.sh "<ids or empty>" <ablation> <direction> <model> [repo_slug] [dataset] [workers]
```

Examples
```bash
# All issues, ALL ablations, BOTH directions, GPT‑5‑mini, 4 workers
bash ./run_codex_direct.sh "" all both gpt-5-mini    data/eval_set.jsonl  4

# MiniMax M2.5 via OpenRouter, full memory, backward
f

# Snapshot model, baseline, backward, 6 workers
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17  data/eval_set.jsonl 6

# Filter issues by repo slug (use dataset to expand IDs)
bash ./run_codex_direct.sh "" L1 backward minimax/minimax-m2.5 agno-agi/agno data/eval_set.jsonl 4
```

Outputs
- results/codex/<ablation>_<model>/issue_id/ contains checkout, transcripts, patch.diff, result.json
- predictions.json is updated after each issue to avoid data loss on crashes
- For multi‑problem issues, Codex fixes one problem at a time in the same checkout and writes one unified patch.diff

## 4) Non‑OpenAI with Codex (MiniMax M2.5)

See codex/docs/reademe.md for how MiniMax M2.5 is wired via OpenRouter (OpenAI‑compatible) into Codex.

## 5) Troubleshooting

- GPT‑5 “max_tokens not supported” → Fixed: preflight uses Responses (max_output_tokens / max_completion_tokens) or Chat as needed.
- “Model metadata … fallback metadata” → Local client warning only; calls still hit minimax/minimax‑m2.5.
- “Model mismatch” FATAL → Provider returned a different model; correct the model string or provider, then rerun.
- ChatGPT login is not required; runs use API keys from .env.


### Mini‑SWE – 12 One‑Liners

Run ALL eval_issues for each model, ablation, and direction (workers=4, dataset=data/eval_set.jsonl).

# GPT‑5‑mini
```bash
bash ./run_miniswe_direct.sh "" BASELINE backward gpt-5-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" BASELINE forward  gpt-5-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward gpt-5-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  gpt-5-mini "" data/eval_set.jsonl 4
```
# GPT‑5.4‑mini‑2026‑03‑17
```bash
bash ./run_miniswe_direct.sh "" BASELINE backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" BASELINE forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
```
# MiniMax M2.5
```bash
bash ./run_miniswe_direct.sh "" BASELINE backward minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" BASELINE forward  minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  minimax2.5 "" data/eval_set.jsonl 4
```
### Codex – 12 One‑Liners

Run ALL eval_issues for each model, ablation, and direction (workers=4, dataset=data/eval_set.jsonl). The script auto‑configures provider and prints an auth banner.

# GPT ‑5 ‑mini
```bash
bash ./run_codex_direct.sh "" baseline backward gpt-5-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  gpt-5-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward gpt-5-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  gpt-5-mini "" data/eval_set.jsonl 4
```
# GPT ‑5.4 ‑mini ‑2026 ‑03 ‑17
```bash
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
```
# MiniMax M2.5 (via OpenRouter)
```bash
bash ./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
```
