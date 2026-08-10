# CI‑Repair Bench – Clean Setup & Run (Mini‑SWE + Codex)

This project runs two agents over a CI‑repair benchmark, with or without memory, across the supported models. One agent at a time.
- mini‑swe‑agent
- codex (OpenAI Codex CLI)

Supported agent models (exact list):
- **gpt-5.4-mini** (OpenAI GPT-5.4 - recommended, latest version)
- `minimax2.5` (MiniMax M2.5 via OpenRouter; resolves to `openrouter/minimax/minimax-m2.5`)

Ablations: BASELINE, L1, L1+L2, L1+L2+L3
Directions: backward (data/back_trs) or forward (data/fwr_trs)

Notes
- GLM 5.2 may be used only for decomposition/memory build, not as an agent model.
- predictions.json is updated incrementally after each issue so crashes do not lose patches.

## 0A) Mini-SWE-Agent-only Setup (Codex Not Required)

Use this setup when you want to run only Mini-SWE-Agent. It does **not**
install or require the Codex CLI, Node.js, a ChatGPT login, or Docker.

### Prerequisites

- Python 3.10 or newer
- Git
- Internet access for model API calls and cloning benchmark repositories

On Ubuntu, install the system prerequisites with:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### Create a standalone Mini-SWE environment

Run these commands from the repository root:

```bash
python3 -m venv .venv-miniswe
source .venv-miniswe/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e ./miniswe-agent
```

The editable Mini-SWE-Agent install includes the runner dependencies such as
LiteLLM, `datasets`, the memory backends, and embedding libraries. You do not
need `requirements-codex.txt` for Mini-SWE-Agent.

Activate this environment again whenever you open a new terminal:

```bash
source .venv-miniswe/bin/activate
```

### Configure the model API key

Create the local environment file:

```bash
cp -n .env.example .env
```

Then add the key for the model you will use:

```ini
# Required for --model gpt-5.4-mini
OPENAI_API_KEY=your-openai-api-key-here

# Required for --model minimax2.5
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Only one provider key is required when you run only one model. Never commit
the `.env` file.

### Verify the Mini-SWE installation

```bash
python -c "import minisweagent, litellm, datasets; print('Mini-SWE-Agent dependencies are installed')"
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py --help
test -f data/eval_set.jsonl && echo "Evaluation dataset found"
```

For a first API-backed smoke run, use one issue and one worker:

```bash
set -a
source .env
set +a

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '^(43)$' \
  --ablation baseline \
  --direction backward \
  --model gpt-5.4-mini \
  --workers 1
```

Replace `43` with an `instance_id` present in `data/eval_set.jsonl`. Baseline
runs do not require memory files. The L1, L1+L2, and L1+L2+L3 modes require
the appropriate files under `data/back_trs/` or `data/fwr_trs/`.

Mini-SWE-Agent clones each benchmark repository and runs its repair commands
locally. A particular benchmark project may therefore require its own tools
(for example `uv`, Poetry, Node.js, or Rust), even though those tools are not
required by the Mini-SWE runner itself. Repositories are cached under `repo/`;
set `MSWEA_REPO_CACHE_ROOT` to use a different cache directory.

## 0B) Combined Mini-SWE + Codex Setup

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

- Create `.env` file at repo root with your API keys:
```ini
# OpenAI (for GPT-5.4 models)
OPENAI_API_KEY=your-openai-api-key-here

# OpenRouter (for MiniMax M2.5)
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM-5.2 via Z.ai (optional, for decomposition/memory build)
GLM_API_KEY=your-glm-api-key-here
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL_NAME=glm-5.2

# HuggingFace (for dataset loading)
HUGGINGFACE_TOKEN=your-huggingface-token-here
```

> **Security**: Never commit `.env` file to git. It's already in `.gitignore`.

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
# Create .env file (copy API keys from secure location - DO NOT commit to git)
cat > .env << 'EOF'
# OpenAI API Key (for GPT-5.4)
OPENAI_API_KEY=your-openai-api-key-here

# OpenRouter API Key (for MiniMax M2.5)
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM-5.2 via Z.ai Platform (optional)
GLM_API_KEY=your-glm-api-key-here
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL_NAME=glm-5.2

# HuggingFace Token
HUGGINGFACE_TOKEN=your-huggingface-token-here
EOF

# Secure the .env file
chmod 600 .env
```

> **Important**: Replace placeholder values with your actual API keys. Keep `.env` secure and never commit it.

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
python3 scripts/decompose_backward.py --batch --use-huggingface --dataset data/memory_set.jsonl --model gpt-5.4-mini --output-dir data/back_trs

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
  --model gpt-5.4-mini \
  --output-dir data/back_trs
# Detach: Ctrl+B then D

# Or using nohup
nohup python3 scripts/decompose_backward.py \
  --batch \
  --use-huggingface \
  --dataset data/camel_memory_set.jsonl \
  --model gpt-5.4-mini \
  --output-dir data/back_trs > decompose.log 2>&1 &

# Monitor progress
tail -f decompose.log
```

**Supported models for decomposition:**
- `gpt-5.4-mini` (recommended - OpenAI GPT-5.4 latest)
- `minimax2.5` (cost-effective alternative)
- `glm5.2` (requires GLM_API_KEY in .env)

## 2) Run Mini‑SWE‑Agent (evaluation runner)

Complete the [Mini-SWE-Agent-only setup](#0a-mini-swe-agent-only-setup-codex-not-required)
first, then activate its environment and load `.env`:

```bash
source .venv-miniswe/bin/activate
set -a
source .env
set +a
```

### Direct runner

This is the recommended command when Mini-SWE-Agent has its own environment:

```bash
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '<regular expression over instance_id>' \
  --ablation <baseline|L1|L1+L2|L1+L2+L3> \
  --direction backward \
  --model <gpt-5.4-mini|minimax2.5> \
  --workers <N>
```

Examples:

```bash
# One issue, baseline (no memory), MiniMax through OpenRouter
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '^(43)$' \
  --ablation baseline \
  --model minimax2.5 \
  --direction backward \
  --workers 1

# Several issues with full backward memory, GPT-5.4 Mini
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '^(43|111|121)$' \
  --ablation L1+L2+L3 \
  --direction backward \
  --model gpt-5.4-mini \
  --workers 4
```

Results are written incrementally under:

```text
results/miniswe-agent/<ablation>_<model>/
```

### Convenience wrapper

The wrapper accepts a comma-separated issue list and can run several ablations
or both directions:

```bash
./run_miniswe_direct.sh \
  "43,111,121" \
  L1+L2+L3 \
  backward \
  gpt-5.4-mini \
  "" \
  data/eval_set.jsonl \
  4
```

Arguments, in order, are:

```text
issue_ids  ablation  direction  model  optional_repo_slug  dataset  workers
```

Pass an empty issue list (`""`) to use every ID in
`data/eval_issue_ids.json`. Pass `all` for the ablation or `both` for the
direction to run all supported combinations.

> The wrapper automatically activates `.venv-codex` when that directory
> exists. If you maintain separate Mini-SWE and Codex environments, use the
> direct runner above to guarantee that `.venv-miniswe` remains active.

Notes:

- Run Mini-SWE and Codex separately; concurrent jobs may contend for repository caches.
- `--workers` controls parallel issues. Start with `1`, then increase it based on CPU, memory, and API rate limits.
- Valid Mini-SWE model aliases are `gpt-5.4-mini` and `minimax2.5`.
- `baseline` ignores memory. Other ablations load backward or forward memory according to `--direction`.

## 3) Run Codex (OpenAI Codex CLI agent)

### Key Points
- Supported models: **`gpt-5.4-mini`** (recommended), `minimax/minimax-m2.5`
- Environment variables must be loaded from `.env` before running
- Use the wrapper script `./run_gpt54.sh` for easy setup

### Quick Start - Wrapper Script (Recommended)

The `run_gpt54.sh` script automatically loads `.env` and uses GPT-5.4-mini:

```bash
# Activate environment
source .venv-codex/bin/activate

# Run with wrapper (auto-loads .env)
./run_gpt54.sh                    # Uses defaults: baseline, backward, data/eval_set.jsonl, 1 issue
./run_gpt54.sh baseline backward  # Specify ablation and direction
./run_gpt54.sh L1+L2+L3 backward data/eval_set.jsonl 4  # Full memory, 4 workers
```

### Manual Commands (Load .env First)

**Before running any command, load environment variables:**
```bash
# Option 1: Source .env file
set -a
source .env
set +a

# Option 2: Export from .env
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
```

**Then run Codex:**
```bash
# Basic command with GPT-5.4-mini
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 1

# Full memory (L1+L2+L3)
bash ./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini "" data/eval_set.jsonl 4

# MiniMax model
bash ./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
```

### Running on Server

```bash
# Using tmux (recommended for long-running jobs)
tmux new -s codex
source .venv-codex/bin/activate
./run_gpt54.sh L1+L2+L3 backward data/eval_set.jsonl 4
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t codex

# Or using nohup
nohup ./run_gpt54.sh baseline backward data/eval_set.jsonl 4 > codex.log 2>&1 &
tail -f codex.log
```

### View Results

```bash
# Results are in: results/codex/<ablation>_<model>/<issue-id>/
ls -la results/codex/baseline_gpt-5_4-mini/416/

# View specific result
cat results/codex/baseline_gpt-5_4-mini/416/result.json | python3 -m json.tool

# View generated patch
cat results/codex/baseline_gpt-5_4-mini/416/patch.diff
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


### Mini‑SWE – Quick Commands

Run ALL eval_issues for each model, ablation, and direction (workers=4, dataset=data/eval_set.jsonl).

#### GPT-5.4-mini (Recommended)
```bash
bash ./run_miniswe_direct.sh "" BASELINE backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" BASELINE forward  gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini "" data/eval_set.jsonl 4
```

#### MiniMax M2.5
```bash
bash ./run_miniswe_direct.sh "" BASELINE backward minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" BASELINE forward  minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  minimax2.5 "" data/eval_set.jsonl 4
```
### Codex – Quick Commands

**Important**: Load environment variables before running: `set -a; source .env; set +a`

#### Using Wrapper Script (Easiest - Auto-loads .env)
```bash
./run_gpt54.sh baseline backward data/eval_set.jsonl 4
./run_gpt54.sh baseline forward  data/eval_set.jsonl 4
./run_gpt54.sh L1+L2+L3 backward data/eval_set.jsonl 4
./run_gpt54.sh L1+L2+L3 forward  data/eval_set.jsonl 4
```

#### Manual Commands with GPT-5.4-mini
```bash
# Load .env first: set -a; source .env; set +a
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini "" data/eval_set.jsonl 4
```

#### MiniMax M2.5 (via OpenRouter)
```bash
# Load .env first: set -a; source .env; set +a
bash ./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
```
